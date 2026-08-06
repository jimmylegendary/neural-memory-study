# ch14. Sleep-time Compute — 이 라인의 이름

## 14.1 Bridge-in: 전작이 남긴 문제

이 장에는 상속받을 open question이 없다.

Part II의 다른 장들은 전작이 명시적으로 남긴 미해결 문제를 인용으로 복원하며 시작한다. ch14는 그렇게 시작할 수 없다. Letta의 `Sleep-time Compute`(2504.13171)는 어떤 선행 연구의 open question도 인용하지 않고, 요약하지도 않는다. 참고문헌은 22개이며[References, pp.13–15], Related Work[Letta STC §2, p.3]는 미해결 질문이 아니라 **기법 계열**로 조직되어 있다 — test-time scaling, speculative decoding, 그리고 고전적 pre-computation.

계보에 가장 가까운 문장은 하나뿐이고, 그것은 문제의 계보가 아니라 **유비의 계보**다.

> 'sleep-time compute builds on the idea of pre-fetching in traditional operating systems, in the context of LLMs à la Packer et al. (2023), storing frequently used computational results to avoid higher latency at test-time.' [Letta STC §2 'Pre-computation', p.3]

Packer et al. 2023은 MemGPT다(→ ch13). 같은 저자들이고 같은 회사(Letta)다. 그런데 이 문장이 MemGPT에서 가져오는 것은 "운영체제의 pre-fetching처럼"이라는 비유이지 MemGPT가 남긴 문제가 아니다. MemGPT의 open question은 인용되지도, 바꿔 쓰이지도 않는다. 이 장이 §14.8에서 다시 확인하겠지만, E-경로 내부 인용은 이 한 건이 전부다 — RAG도, vector store도, graph memory도, 검색 문헌 어느 것도 참고문헌에 없다.

상속이 없는 자리를 논문은 **공백 제조**로 채운다. 논문은 test-time scaling 문헌 안에 있는 가정 하나를 이름 붙여 꺼낸다.

> 'These drawbacks are in part due to the fact that the current approach to applying test-time compute assumes that problems are stateless, i.e. queries (user queries at test-time) and the contexts (background information) required for answering them are provided to the model together at test-time.' [Letta STC §1, p.1]

이 문장이 지목하는 논문들(Snell et al. 2024, OpenAI 2024, DeepSeek-AI 2024, Brown et al. 2024)은 그 가정을 자기 open question으로 진술한 적이 없다. stateless 가정은 저자들이 그 문헌을 읽고 **부여한 성격 규정**이다. 그러므로 이 장의 bridge-in은 "전작이 남긴 문제를 받는다"가 아니라 "전작에게 문제를 만들어 준다"이다.

> **[평가]** 상속이 비어 있다는 것은 이 장의 결함이 아니라 관측이다. 그리고 그 관측은 이 책 전체에서 반복될 형태의 첫 사례다. 한 라인의 이름을 만든 논문이 그 라인의 조상을 갖지 않는다. 더 정확히는, 조상을 **다른 분야에서** 골랐다 — 논문이 자기 지적 부모로 세운 것은 캐시(Smith 1982), data cube(Gray 1997), speculative decoding(Leviathan et al. 2023, Stern et al. 2018, Cai et al. 2024)이다. 기억 연구도 아니고 학습 연구도 아니며, 제목이 빌려 쓴 생물학은 더더욱 아니다. 이 사실이 §14.8의 침묵 지도와 ch11의 판별식 논의로 그대로 이어진다.

ch13이 이 장에 넘겨주는 것은 따라서 문제가 아니라 **사실 두 가지**다. 첫째, 외부 기억을 운영체제 유비로 다루는 관행이 이미 있었다는 것. 둘째, 그 관행의 저자들이 이 논문의 저자들과 같다는 것. 이 장은 그 위에서, 상속 없이 시작한다.

---

## 14.2 문제의식

논문이 스스로 정의하는 문제는 비용의 크기가 아니라 **비용이 청구되는 시점**이다.

논문은 test-time scaling이 정확도를 사면서 지연과 달러를 낸다는 것을 출발점으로 놓되, 그 지불이 불가피한 물리량이 아니라 모델링 가정의 산물이라고 주장한다[Letta STC §1, p.1]. 가정은 §14.1에 인용한 stateless 가정이다. 그 가정 아래에서는 "여러 관련 질의가 문맥에 대해 비슷한 추론을 요구하면 모델은 매번 중복 계산을 다시 해야 하고, 그만큼 지연과 비용이 추가된다"[Letta STC §1, p.1].

반대 관측은 응용의 형태에서 온다.

> 'In reality, many LLM applications are inherently stateful, and work in conjunction with persisted, re-used context.' [Letta STC §1, p.1]

논문이 드는 사례는 세 가지다 — document QA, 공통 저장소 위에서 작동하는 coding agent, 대화 이력을 들고 있는 대화형 assistant. 셋 모두에서 "다음 사용자 입력이 오기 전에 이미 확보된 문맥(문서, 코드베이스, 대화 이력)이 존재한다"[Letta STC §1, p.1].

여기서 조작적 질문이 나온다. 프롬프트가 문맥과 질의의 쌍으로 분해되고 문맥이 질의보다 먼저 도착한다면, **문맥에만 쓴 계산은 test-time에 무엇을 사주는가**[Letta STC §3, p.3].

이 질문에 답하려면 먼저 두 국면을 분리해야 한다. 이 장이 소유하는 첫 번째 정의다.

> **정의 (sleep/wake 분리의 조작적 형태).** 하나의 서빙 경로를 두 국면으로 쪼갠다. **sleep 국면**은 질의 $q$가 아직 도착하지 않은 구간이며, 입력으로 문맥 $E_0$만을 가지고 예산 $B_s$를 쓴다. **wake 국면**은 $q$가 도착한 뒤의 구간이며, 예산 $B_t$를 쓴다. 두 국면은 세 가지가 다르다 — (i) 입력 가용성($q$의 유무), (ii) 지연 제약(wake만 사용자 대기 시간에 묶인다), (iii) 단가(wake 토큰이 sleep 토큰보다 비싸다고 가정된다). 이 세 차이 중 하나라도 없으면 분리는 아무것도 사주지 않는다.

sleep-time compute 자체의 일반 정의 — 어느 층을 갱신하는지 지정하지 않는 정의 — 는 ch01이 소유한다. 이 장이 도입하는 것은 그 정의를 **한 논문이 실제로 구현 가능한 형태로 좁힌 결과**다. Letta의 논문은 이 좁힘을 가장 단순한 방식으로 한다: sleep 국면의 산출물을 자연어 문자열 하나로 두고, wake 국면에서 그 문자열을 원래 문맥 자리에 끼워 넣는다.

> **[해설]** 독자의 어휘로 옮기면 이렇다. prefix cache는 "이 토큰들의 KV를 다시 만들지 않겠다"는 장치다. 같은 계산의 재사용이고, 무손실이며, 방향이 없다 — 어떤 질의가 오든 같은 KV가 유효하다. sleep 국면의 산출물은 다르다. "이 문맥에서 도출될 결론을 미리 도출해 두겠다"는 장치이고, 새로운 계산의 **선행 수행**이며, 손실적이고 방향이 있다 — 어떤 질의를 예상했느냐에 따라 유용성이 달라진다. 이 차이가 §14.6의 고예산 역전과 §14.7의 prefill 증가를 동시에 설명한다. 미리 뽑아 둔 결론은 맞을 때는 계산을 대신해 주고, 빗나갈 때는 프롬프트를 길게 만들 뿐이다.

문제의식의 마지막 조각은 평가에 있다. 논문은 자기가 정의한 문제를 기존 벤치마크로 시험하지 않는다. stateful reasoning을 재는 기존 벤치마크가 논문 안에 하나도 인용되지 않는다. 대신 저자들은 기존 과제를 잘라 만들었고, 그 절단이 §14.6의 모든 수치의 전제가 된다.

---

## 14.3 Core mechanism (통일 표기)

### 14.3.1 기제의 다섯 줄

먼저 짚어야 할 형식적 사실이 있다. **논문에는 번호가 붙은 수식이 하나도 없다.** 기제 전체가 [Letta STC §3](pp.3–4)의 산문 속 화살표 세 개로 진술된다. 따라서 이 절의 모든 출처 표기는 절·부록 위치이지 원문 식 번호가 아니다.

논문의 출발 선언은 프롬프트의 분해다. 이 책의 표기로 옮기면 원 문맥이 external store의 초기 상태가 된다.

$$
\text{prompt} \;=\; \big(E_0,\ q\big), \qquad E_0 \;:=\; c
\tag{14-1}
$$

여기서 $c$는 논문의 기호이고, 이 책은 $c$를 다른 뜻으로 예약해 두었으므로(§14.3.3) 본문에서는 쓰지 않는다. baseline은 식 (R)의 특수화다 — 원 문맥 전체가 프롬프트에 들어가고 검색 연산은 항등이며 wake 예산이 크다.

$$
\hat y \;=\; f_{B_t}\big(q;\ \Theta,\ W,\ \mathrm{ret}(E_0, q)\big),
\qquad \mathrm{ret}(E_0,q) \;=\; E_0
\tag{14-2}
$$

sleep 국면의 쓰기는 식 (U-E)의 특수화다. 그리고 이 특수화가 세 겹으로 이루어져 있다는 것이 이 장의 핵심 관측이다.

$$
E^{(j)} \;=\; \mathrm{wr}\big(E^{(j-1)},\ \hat c^{(j)}\big) \;=\; \hat c^{(j)},
\qquad
\hat c^{(j)} \;=\; S\big(E^{(j-1)};\ B_s/J\big),
\qquad j = 1,\ldots,J,\quad J \le 10
\tag{14-3}
$$

wake 국면의 읽기는 다시 식 (R)이되, 저장소가 이제 재표현을 들고 있고 wake 예산이 작다.

$$
\hat y \;=\; f_{B_t^{\mathrm{lo}}}\big(q;\ \Theta,\ W,\ \mathrm{ret}(E_1, q)\big),
\qquad \mathrm{ret}(E_1,q) \;=\; E_1 \;=\; \hat c^{(J)},
\qquad B_t^{\mathrm{lo}} \ll B_t
\tag{14-4}
$$

식 (14-3)과 (14-4)의 쌍이 논문의 주장 전체다. 같은 읽기 사상을, 훨씬 작은 wake 예산으로, 저장소가 중간 결과를 미리 들고 있기 때문에 쓸 수 있다는 것.

$B_s$를 키우는 손잡이는 두 개다. 비추론 모델에서는 병렬 sleep 궤적을 늘린다.

$$
\hat c_i \;=\; S\big(E_0;\ B_s/\kappa_{\parallel}\big),\ \ i=1,\ldots,\kappa_{\parallel},
\qquad
E_1 \;=\; \mathrm{concat}\big(\hat c_1,\ldots,\hat c_{\kappa_{\parallel}}\big),
\qquad \kappa_{\parallel}\in\{1,2,5,10\}
\tag{14-5}
$$

추론 모델(o1, o3-mini)에서는 API의 'reasoning effort' $\in$ {low, medium, high}를 돌린다[Letta STC §5.2, p.8; Fig. 8 범례, p.10]. 즉 이 경우 $B_s$는 토큰 수가 아니라 **벤더 쪽 서수**다.

마지막으로, 세 층 중 둘은 움직이지 않는다. 논문이 식으로 쓰지는 않지만 설정이 강제한다 — 모든 대상 모델이 폐쇄 API이고 논문 어디에도 training, fine-tuning, adapter, optimizer가 없다[Letta STC §4.2, p.5].

$$
\Theta_{k+1} = \Theta_k \quad \forall k,
\qquad
W \ \text{는 이 아키텍처에 존재하지 않는다}
\tag{14-6}
$$

### 14.3.2 정규형과 어긋나는 지점 다섯

식 (U-E)의 정규형은 $\mathrm{wr}$과 $S(c_k;B_s)$의 내부를 지정하지 않는다. 이 논문은 그 빈칸을 다음과 같이 채우고, 채우는 방식마다 결과가 달라진다.

**첫째, $\mathrm{wr}$은 파괴적 덮어쓰기다.** 정규형의 $\mathrm{wr}$은 append든 merge든 index-insert든 허용한다. 여기서는 참조 구현이 블록 값을 그대로 세팅한다 — `agent_state.memory.update_block_value(label=target_block_label, value=new_memory)`[Letta STC App. F Listing 1, p.22]. 부록 K의 서술도 같다: 함수가 "현재 문맥을 새 문자열로 교체한다"[Letta STC App. K, p.27]. 결과로 저장소에 이력이 없고 rollback이 없다. sleep 국면 안에서 잘못된 재작성은 복구 불가능하다.

**둘째, $S(\cdot;B_s)$는 단발이 아니라 반복이다.** 깊이 $J \le 10$이며, 각 단계는 현재 블록과 지정된 원본 블록 하나를 읽고, 모델이 `finish_rethinking`을 호출하면 끝난다[Letta STC App. K, p.27; Fig. 17, p.18]. 정규형은 $B_s$를 한 덩어리 예산으로 쓰지만 여기서는 **순차 사슬**로 소비된다. 이것이 §14.7의 sleep wall-clock 논의를 만든다.

**셋째, 서로 다른 두 개의 $\mathrm{wr}$이 이름 없이 공존한다.** 궤적 안에서는 덮어쓰기(식 14-3), 궤적 사이에서는 이어붙이기(식 14-5)다. 논문은 두 연산자를 쓰고 있다는 사실 자체를 언급하지 않는다. 이어붙이기 쪽에서는 상태 크기가 $\kappa_{\parallel}$에 선형으로 자라고, §14.6의 비단조성($\kappa_{\parallel}=5$가 $\kappa_{\parallel}=10$을 이긴다)이 그 성장의 첫 번째 가시적 대가다.

**넷째, $\mathrm{ret}$이 항등이다.** 정규형의 $\mathrm{ret}(E,q)$는 선택을 함의한다. 여기서는 저장소 전체가 원 문맥 자리에 통째로 끼워진다[Letta STC §3, p.4]. 인덱스도, 임베딩도, 유사도 점수도, top-$m$ 절단도 없다. 튜닝할 검색기가 없고 검색 실패 모드도 없다 — 그리고 그 말은 **E-경로가 통상 치르는 검색 비용이 이 논문에서 하나도 행사되지 않았다**는 뜻이기도 하다(→ ch10, ch15).

**다섯째, $E$는 문맥 범위이고 휘발성이다.** 문맥 하나당 저장소 하나이며, 지속되지 않고, 문맥 사이에서 병합되지 않고, 재통합되지 않는다. 논문 자신이 이것을 작업의 경계로 명시한다[Letta STC §7, p.13]. 그리고 sleep 라운드 첨자 $k$는 모든 실험에서 1을 넘지 않는다.

기제에 **없는 것**도 같은 무게로 기록해야 한다. 축출·만료·압축·용량 상한이 없다($J \le 10$이라는 재작성 횟수 제한이 유일한 구조적 경계다). 세션을 넘는 상태가 없다. 그리고 $\hat c$의 **검증이 없다** — 블록에 적힌 추론이 옳은지 확인하는 절차가 어디에도 없다. 유일한 시늉은 레벨 4 verbosity 프롬프트가 모델에게 "블록의 숫자를 쓸 거면 먼저 다시 계산해서 확인하라"고 지시하는 대목인데[Letta STC Fig. 16, p.17], 그것은 검증 비용을 wake 국면으로 되돌린다.

### 14.3.3 표기 대응표

표 14-1 — Letta `Sleep-time Compute`의 원 논문 기호와 이 책의 표기. 원문에 번호 붙은 수식이 없으므로 '원문 위치'는 절·부록·그림 위치다.

| 원 논문 기호 | 논문에서의 의미 | 이 책의 예약 충돌 | 이 책의 표기 |
|---|---|---|---|
| $c$ [§3, p.3] | 기존 문맥(코드베이스, 문서, 수학 문제의 앞부분) | 소문자 $c$는 Neural Memory에서 Omega rule window 길이로 예약되어 있고(→ NM), 대문자 chunk 크기 $C$와도 구분해야 한다 | $E_0$ — 원 문맥은 external store의 초기 상태다 |
| $c'$ [§3, p.4; Fig. 1 라벨 'Learned Context'] | sleep 국면이 만든 재표현 문맥 | 없음. 다만 'Learned'가 오도한다 — 이 논문에 gradient가 없다 | $\hat c$, 그리고 그 결과 상태로서 $E_1$ |
| $S(\cdot)$ [§3, p.4] | sleep 절차, $S(c)\to c'$ | $S_t$ = inner momentum(surprise) 버퍼(ch01 예약) | $S(\cdot;B_s)$. **판별 규칙**: momentum은 항상 시간 첨자 $S_t$, sleep 연산자는 항상 예산 인자 $(\cdot;B_s)$. 첨자 없는 맨 $S$는 쓰지 않는다 |
| $T_B(q,c)\to a$, $T_b(q,c')\to a$ [§3, pp.3–4] | 예산 $B$ 또는 $b$로 적용한 test-time 방법 | $T$는 sequence 길이 의미로 이미 $L$에 흡수됨 | 읽기 사상 $f(\cdot)$에 예산 첨자: $f_{B_t}$, $f_{B_t^{\mathrm{lo}}}$ |
| $B$, $b$ ($b \ll B$) [§3, p.4] | 큰/작은 test-time 예산 | 맨 $b$ 사용 금지(mini-batch 크기 의미와 충돌) | $B_t$, $B_t^{\mathrm{lo}}$ |
| $t = 10$ [§5.3, p.8] | test-time 생성 토큰이 sleep-time 생성 토큰보다 비싼 배수 | $t$ = 토큰 인덱스(ch01 예약) | $\kappa_{\mathrm{cost}}$ (장-국소) |
| $k$ [§5.2, p.8 / §5.1, p.7] | 두 가지로 쓰인다 — (i) 병렬 sleep 생성 개수, (ii) pass@$k$의 $k$ | $k$ = sleep 라운드 첨자(예약), $k_t$ = key 벡터 | (i) $\kappa_{\parallel}$ (장-국소), (ii) pass@$\kappa$ |
| $N$, $q_1\ldots q_N$ [§3, p.3] | 한 문맥을 공유하는 질의 수 | 없음 | $N_q$ |
| $p = (c,q)$ [§3, p.3] | 전체 사용자 프롬프트 | $p$ = $\ell_p$ 노름 차수 | 기호를 새로 만들지 않고 쌍 $(q, E)$를 그대로 쓴다 |
| $q$ [§3, p.3] | 사용자 질의, 자연어 문자열 | $q_t$ = query 벡터(항상 첨자) | $q$ 유지. **E-경로에서 $q$는 투영된 벡터가 아니라 토큰 문자열이다** |
| $a$ [§3, p.3] | 추론 흔적 뒤에 나오는 최종 답 | 없음 | $\hat y$ |

장-국소 기호 두 개를 추가로 선언한다. $J$ = sleep 국면의 재작성 깊이(상한 10). $\kappa_{\mathrm{cost}}$와 $\kappa_{\parallel}$은 이 장에서만 쓰며, Neural Memory가 Newton–Schulz 반복 횟수로 예약한 맨 $\kappa$(→ NM)와는 첨자로 구분된다.

### 14.3.4 이 장이 소유하는 두 정의

> **정의 (learned context $\hat c$).** learned context는 질의 $q$가 도착하기 전에 문맥 $E_0$만을 입력으로 예산 $B_s$를 써서 수행한 추론의 산출물이며, wake 국면에서 $E_0$을 **대체하여** 프롬프트에 놓이는 자연어 문자열이다. 이름의 'learned'는 gradient를 뜻하지 않는다 — 이 논문에는 backward pass가 없다(식 14-6). 학습되는 것은 파라미터가 아니라 **표현이 놓이는 공간**이고, 그 공간은 자연어다.

이 정정은 이 책의 재구성이 아니라 논문 본문의 진술과 일치한다. 논문은 [Letta STC §7]에서 스스로 이렇게 쓴다.

> 'Unlike traditional representation learning (Bengio et al., 2014), which typically operates in model parameter or activation space, we instead form representations in the space of natural language.' [Letta STC §7, p.13]

이 한 문장이 이 책의 3경로 분할에 대한 가장 좋은 1차 근거다. parameter space가 $\Theta$-경로, activation space가 $W$-경로, natural-language space가 $E$-경로다 — 그리고 그 삼분을 E-경로 저자가 직접 썼다(→ ch11). 'Learned Context'라는 오도하는 라벨은 Figure 1(p.2)의 그림 안에만 있고, 논문 본문은 정확하다.

> **정의 (상각 논증의 원형).** 상각 논증은 다음 형태의 논증이다 — sleep 국면의 비용은 문맥당 한 번 발생하고 wake 국면의 비용은 질의당 발생하므로, 한 문맥을 공유하는 질의 수 $N_q$가 커질수록 질의당 평균 비용이 식 (A)를 따라 떨어진다. 이 논증이 숫자로 성립하려면 세 값이 필요하다: 분자 $B_s$, wake 비용, 그리고 $N_q$.

논문이 쓰는 형태는 식 (A)에 wake/sleep 단가비를 넣은 것이다.

$$
\mathrm{Cost}_{\mathrm{avg}} \;=\; \kappa_{\mathrm{cost}}\cdot B_t^{\mathrm{lo}} \;+\; \frac{B_s}{N_q},
\qquad \kappa_{\mathrm{cost}} = 10
\tag{14-7}
$$

원문은 이렇게 쓴다(논문의 배수 기호는 표 14-1에 따라 $\kappa_{\mathrm{cost}}$로 옮긴다).

> 'we consider a simple linear model where tokens generated at test-time are a factor t the cost of the tokens at sleep-time. In our analysis, we set t = 10' [Letta STC §5.3, p.8]

이 식이 이 장의 나머지 절반을 지배한다. §14.5는 위 세 값 중 무엇이 논문에 있고 무엇이 없는지를 따진다.

---

## 14.4 어느 층을 언제 쓰는가

답부터 쓴다. **이 논문은 (U-E)만 실행한다. 그리고 정확히 한 번 실행한다.**

표 14-2 — 이 논문에서 움직이는 값과 움직이지 않는 값. "이 값은 누가 정하는가"에 전부 답한다.

| 대상 | 누가 정하는가 | 언제 | 이 논문에서 움직이는가 | 출처 |
|---|---|---|---|---|
| $\Theta$ — GPT-4o-mini, GPT-4o, o1, o3-mini, Claude Sonnet 3.7 Extended Thinking, DeepSeek-R1의 backbone | 모델 벤더가, 이 논문이 존재하기 전에 | outer-loop 사전·사후학습, 전부 논문 밖 | 아니오. 동결 | §4.2, p.5 |
| $W$ — fast weights | 아무도 | 해당 없음 | 아니오. **이 층이 사용된 아키텍처에 존재하지 않는다** | §7, p.13(쓰지 않는다고 명시) |
| $E$ — `rethink memory` 블록, 단일 가변 텍스트 블록 | 동결된 모델 자신이, 프롬프팅으로 | **sleep 국면.** 최대 10회 순차 `rethink_memory` 호출, `finish_rethinking`으로 종료 | **예. 유일한 가동부** | App. K, p.27; App. F Listings 1–2, p.22; Fig. 17, p.18 |
| sleep 시스템 프롬프트(Letta-Offline-Memory 페르소나, rethink/finish 프로토콜) | 저자가 손으로 | 고정 | 아니오 | Fig. 17, p.18; AIME 전용 부기 Fig. 18, p.18 |
| wake verbosity 프롬프트 5종(레벨 0–4) | 저자가 손으로 | 실험별 선택, 고정 | 아니오 | §5.1, p.6; App. A Figs. 12–16, pp.16–17 |
| $\kappa_{\parallel} \in \{1,2,5,10\}$ | 아무도 학습하지 않음. 손 스윕 | 실험 구성 | 스윕될 뿐 학습되지 않음 | §5.2, p.8; Fig. 7, p.10 |
| sleep 'reasoning effort' $\in$ {low, medium, high} | 손 스윕. effort→토큰 매핑은 벤더 내부이며 비공개 | 실험 구성 | 스윕될 뿐 학습되지 않음 | §5.2, p.8; Fig. 8, p.10 |
| $B_t$ — wake 예산 | 손 스윕. 모델별로 세 가지 다른 방식(verbosity 프롬프트 / API reasoning-effort / budget-forcing) | 실험 구성 | 스윕될 뿐 학습되지 않음 | §5.1, p.6 |
| $J \le 10$ — 재작성 호출 상한 | 아무도. 하드코딩된 상수 | 고정 | 아니오 | App. K, p.27 |
| $\kappa_{\mathrm{cost}} = 10$ — wake:sleep 단가비 | 아무도. 외부 벤더 문서에서 단언, 측정 아님 | 고정된 모델링 가정 | 아니오 | §5.3, p.8 + 각주 4 |
| Multi-Query 문항 생성기(o3-mini + Fig. 19 프롬프트) | 아무도. 동결된 벤더 모델 | **오프라인 데이터셋 구성** — 방법의 일부가 아님 | 아니오 | §4.1, p.4; App. C Fig. 19, p.19 |
| SWE-Features PR 군집기(claude-sonnet-3-7-20250219) — 무엇이 '관련 문맥'인지를 정한다 | 아무도. 동결된 벤더 모델 | **오프라인 벤치마크 구성** | 아니오 | App. D, p.20 |
| 질의 예측가능성 스코어러(Llama2-70B base, $\log P(q \mid E_0)$) | 아무도. 동결된 제3자 base 모델 | **사후 분석 도구** | 아니오 | §5.4, p.9 |

표 14-2에서 읽어야 할 것은 outer-loop 열이 **비어 있다**는 사실이다. 이 논문에서 학습되는 것은 없다. 학습 가능해 보이는 모든 대상은 (a) 동결된 벤더 모델이거나, (b) 손으로 쓴 프롬프트이거나, (c) 손으로 스윕한 하이퍼파라미터다. 움직이는 것은 텍스트 블록 하나뿐이고, 그것이 sleep 국면에 움직인다.

> **[해설]** 그래서 이 장이 Part II에서 E-경로의 정면 사례로 먼저 오는 것이 옳다. Part I의 T1–T5가 가르치는 모든 장치 — fine-tuning과 PEFT(→ ch03), distillation(→ ch04), RL(→ ch05), 합성 데이터(→ ch06), replay와 망각 대응(→ ch07) — 이 여기서는 **전부 부재한다**. E-경로는 backward pass 한 번 없이 끝까지 시연된다. 뒤의 $W$-경로·$\Theta$-경로 장들은 같은 발상을 gradient 값으로 사게 되고, 그 가격 차이가 Part III의 판정 재료다. $W$층이 실제로 어떻게 움직이는지는 이 책이 다시 설명하지 않는다 — 그 층의 갱신식은 이 책의 (U-W)와 글자 그대로 같으며, 전개는 Neural Memory 모노그래프가 갖는다(→ NM 식 (M)).

시간척도는 단일하다. 갱신 첨자 $k$가 1을 넘지 않는다. sleep은 문맥당 한 번 일어나고, 그 뒤 wake 읽기가 $N_q$번 일어나며, 그 사이에 문맥이 갱신되어 다시 sleep이 도는 일은 논문의 어떤 실험에도 없다. 순서로 쓰면 $E_0 \to E_1$(sleep) $\to$ 읽기(wake)뿐이며, 층 간 전이도 cascade도 교차도 없다.

이 단일성이 §14.5의 마지막 칸을 미리 결정한다. **반복 열화 $\rho$는 이 논문에서 미측정이 아니라 구성상 범위 밖이다.** 라운드가 하나뿐이면 라운드당 손실은 정의되지 않는다. 논문 자신이 이 경계를 명시한다.

> 'we make the simplifying assumption that interactions fall into two phases: sleep-time and test-time. However, real-world LLM use cases can be more complex, with multiple rounds of interaction and context modifications between rounds (e.g. multiple edits to a code-base).' [Letta STC §7, p.13]

독자의 1번 질문에 대한 답도 이 절에서 이미 나온다. **decode에서 바뀌는 것은 없다.** 커널도, 가중치 배치도, batching 형태도 그대로다. 프롬프트의 내용만 다르다. 그것이 E-경로가 오늘 당장 배포되는 이유이자, E-경로가 $\Theta$-경로처럼 용량을 올릴 수 없는 이유다. §14.7이 이 답을 서빙 스택의 각 층으로 전개한다.

---

## 14.5 비용 4종

표 14-3 — Letta `Sleep-time Compute`의 비용 4종. 네 칸이 전부 "논문에 없음"이다. 추정치를 넣지 않는다.

| 값 | 논문의 보고 | 대신 보고된 것 | 출처 |
|---|---|---|---|
| $B_s$ — sleep 계산 예산 | **논문에 없음** | 서수 손잡이만: $J \le 10$ 재작성 호출, $\kappa_{\parallel} \in \{1,2,5,10\}$, sleep reasoning effort $\in$ {low, medium, high} | App. K, p.27; §5.2, p.8; Figs. 7–8, p.10 |
| $L_w$ — wake 지연 기여 | **논문에 없음** | 질문당 생성된 test-time 토큰 수(지연의 대리값). Stateful GSM-Symbolic 약 0–600, Stateful AIME 약 1000–22500, SWE-Features 3000–10000 | Fig. 3, p.6; Fig. 4, p.7; Fig. 11, p.12 |
| $C_{\mathrm{cap}}$ — 상태 용량 | **논문에 없음** | 구조적 경계만: 재작성 $J \le 10$회, Letta core-memory 블록 배치(persona·human 읽기 전용, rethink 읽기/쓰기) | App. K, p.27; Fig. 17, p.18 |
| $\rho$ — 망각·열화율 | **논문에 없음** | sleep 라운드가 모든 실험에서 정확히 1회. 반복 통합이 구성상 범위 밖 | §7, p.13 |
| ($N_q$ — 참고) | 스윕값만 | $N_q \in \{1,2,5,10\}$, 합성 데이터셋 위에서 구성. 실제 워크로드의 $N_q$ 분포는 없음 | §5.3, pp.8–9; Fig. 9, p.11 |

### 14.5.1 $B_s$가 없다는 것

표 14-3의 첫 줄이 이 장에서 가장 무거운 사실이다. **상각 논증의 분자가 숫자로 존재하지 않는다.**

논문은 sleep-time 토큰 수도, FLOP 수도, 달러 액수도 보고하지 않는다 — 본문에도, 어떤 축에도, 어떤 표에도 없다. 모든 그림의 $x$축은 'Avg. Test Time Tokens / Question'이며, sleep 비용은 자기 축을 한 번도 갖지 못한다. 유일하게 sleep 비용이 등장하는 곳은 Figure 9(p.11)의 $x$축 'Total Inference Cost / Query'인데, 이 축은 식 (14-7)로 만들어진 **파생 합성량**이므로 $B_s$가 그 안에 $\kappa_{\mathrm{cost}} B_t$와 엉켜 있고 분리해 읽을 수 없다.

보고된 것은 서수뿐이다. 재작성 호출 상한 10, 병렬 궤적 개수 네 가지, 그리고 벤더가 정의를 공개하지 않은 reasoning effort 세 단계. 상한 10은 실제로 몇 번 호출되었는지를 말해 주지 않고, 논문은 실현된 평균 호출 수를 보고하지 않는다.

### 14.5.2 그래서 $N_q^*$를 계산할 수 없다

식 (14-7)에서 손익분기를 유도해 보면 무엇이 빠졌는지가 정확히 드러난다. 비교는 **같은 정확도에서** 이루어져야 한다 — 그것이 논문의 pareto 서술이 하는 일이다. sleep을 쓰는 팔이 wake 예산 $B_t^{\mathrm{lo}}$로 baseline의 $B_t$와 같은 정확도에 도달한다고 두면, sleep을 쓰는 쪽이 총비용에서 이기는 조건은

$$
\kappa_{\mathrm{cost}}\,B_t^{\mathrm{lo}} + \frac{B_s}{N_q} \;<\; \kappa_{\mathrm{cost}}\,B_t
\qquad\Longleftrightarrow\qquad
N_q \;>\; N_q^{*} \;=\; \frac{B_s}{\kappa_{\mathrm{cost}}\,\big(B_t - B_t^{\mathrm{lo}}\big)}
\tag{14-8}
$$

이다. 우변에 필요한 값은 셋이다. $\kappa_{\mathrm{cost}}$는 논문이 준다 — 다만 측정이 아니라 벤더 문서 인용이다(§14.5.4). $B_s$는 없다(§14.5.1). $B_t$와 $B_t^{\mathrm{lo}}$는 생성 토큰 축의 눈금으로만 존재하며, 논문에 결과 표가 없으므로 주석 없는 선 그래프에서 읽어야 한다.

따라서 **$N_q^{*}$는 이 논문의 1차 자료만으로 계산되지 않는다.** 논문이 주는 것은 $N_q \in \{1,2,5,10\}$ 네 점에서의 곡선과, 손익분기가 그 구간 어딘가에 있다는 정성적 진술이다. 그 진술은 Figure 9의 캡션에 있다 — "문맥당 질문 수가 적으면 총비용 관점에서 sleep-time compute이 덜 유리하다"[Letta STC Fig. 9 캡션, p.11]. 즉 논문 스스로 손익분기의 존재를 인정하되 그 위치를 숫자로 주지 않는다.

여기에 더 근본적인 공백이 겹친다. **실제 서빙에서 $N_q$의 분포를 보고한 논문이 이 책이 다루는 corpus 전체에 없다.** 상각 논증은 $N_q$를 안다고 가정하는데, 그 값은 워크로드의 성질이지 방법의 성질이 아니다. 이 논문은 $N_q$를 합성으로 구성해 네 값에서 재고, 그 네 값이 실제 트래픽에서 어떤 빈도로 나타나는지는 다루지 않는다(→ ch25).

### 14.5.3 비용모델이 생성 토큰만 센다

식 (14-7)의 두 항 모두 **생성된 토큰만** 센다. §14.3.4에 인용한 원문이 그렇게 말한다 — 'tokens generated at test-time'[Letta STC §5.3, p.8]. prefill 항, 즉 입력 토큰 항이 어느 쪽에도 없다.

그런데 이 방법의 기제 자체가 **입력을 길게 만드는 것**이다. $\hat c$는 $E_0$보다 길다(Figure 1, p.2의 두 문맥 상자를 나란히 보면 확인된다). 그리고 병렬 스케일링에서는 $\kappa_{\parallel}$개의 재표현이 이어붙어 프롬프트에 실리며, 그 값은 최대 10이고, 그 프롬프트는 **매 질의마다** 다시 처리된다[Letta STC §5.2, p.8]. 이 입력 비용은 5×에도, 13%에도, 18%에도, 2.5×에도 나타나지 않는다.

<!-- TODO-VERIFY: 그림 축 'Avg. Test Time Tokens / Question'이 프롬프트 토큰을 포함하는지 생성 토큰만 세는지. §5.3은 비용모델에 대해 'tokens generated at test-time'이라고 쓰지만 축 자체의 정의는 논문 어디에도 없다. 확인 방법: papers/2504.13171.txt에서 "Avg. Test Time Tokens" 및 "prompt tokens" 검색, 그리고 github.com/letta-ai/sleep-time-compute의 결과 아티팩트. 판정에 따라 5×가 decode-only 절약인지 총토큰 절약인지가 갈린다. -->

### 14.5.4 $\kappa_{\mathrm{cost}}=10$은 단언이다

상수 $\kappa_{\mathrm{cost}}$는 곱셈으로 결론에 들어간다. 식 (14-8)에서 $N_q^{*}$는 $\kappa_{\mathrm{cost}}$에 반비례한다 — $\kappa_{\mathrm{cost}}$를 1로 두면 손익분기 질의 수가 10배로 뛴다. 그만큼 무거운 상수인데, 논문이 제시하는 근거는 측정이 아니라 인용이다.

> 'Since at test-time, there are strict latency constraints, and latency optimized inference can be roughly 10× more expensive, we model the total cost of inference between both sleep-time and test-time, by up-weighing the cost of test-time tokens.' [Letta STC §5.3, p.8]

이 문장의 각주 4는 Databricks provisioned-throughput 문서 페이지를 가리킨다. 논문 안에서 이 비율을 재는 실험은 없고, $\kappa_{\mathrm{cost}}$를 흔들었을 때 2.5×가 어떻게 변하는지에 대한 민감도 분석도 없다.

### 14.5.5 네 칸의 공백이 뜻하는 것

> **[평가]** 비용 4종이 전부 absent인 것은 이 논문 한 편의 부주의가 아니다. 이 라인에 이름을 준 논문이 비용을 보고하지 않았고, 같은 경로의 후속 연구들이 그 관행을 그대로 물려받았다(→ ch15, ch26). 그 결과 "sleep-time compute이 싸다"는 명제는 **현재 문헌 상태에서 검증도 반증도 되지 않는다** — 비교할 수 있는 회계 단위가 없기 때문이다. 이 책은 비용이 침묵한 자리를 성능 주장으로 메우지 않는다(→ ch01). 그래서 표 14-3의 네 칸은 "효율적"이라는 서술 대신 "논문에 없음"으로 남는다.

이 공백을 이 책은 실험 X1로 다룬다. X1은 논문들이 각자의 축으로 흩어 놓은 값을 하나의 회계 단위로 옮기고, prefill 항을 포함했을 때 상각 논증이 어느 레짐에서 살아남고 어느 레짐에서 뒤집히는지를 결정론적으로 계산한다. **판정은 이 장에서 하지 않는다.** 통일 비용 회계의 판정은 ch25가 소유한다. 이 장이 하는 일은 판정에 필요한 세 값 중 두 값이 1차 자료에 없다는 사실을 확정하는 데까지다.

---

## 14.6 실험과 스케일

### 14.6.1 읽기 전에 알아야 할 형식

**논문에 결과 표가 없다.** 표는 정확히 하나 있고(Table 1, p.19), 그것은 데이터셋 통계다. 모든 정확도·F1·비용 수치는 주석 없는 선 그래프로 제시된다. 오차 막대도, 신뢰구간도, 분산 통계도 어디에도 없다. 그러므로 아래에 쓰는 수치는 **논문이 산문으로 진술한 값**과 **그림 축의 범위**로 한정된다. 곡선 위의 점을 눈으로 읽어 옮기지 않는다.

실행 횟수는 arm마다 다르다. o1·o3-mini·R1은 3회 평균, Claude 3.7 Sonnet은 10회 평균이며, 그 이유를 논문이 밝힌다 — "초기 실험에서 잡음이 더 많이 관측되어서"[Letta STC §5.1, p.6]. 즉 표본 수가 결과를 본 뒤에 arm별로 달라졌고, 분산은 보고되지 않아 독자가 그 조정의 타당성을 판단할 재료가 없다.

### 14.6.2 벤치마크는 저자가 만들었다

> **정의 (Stateful 벤치마크 구성).** Stateful 벤치마크 구성이란, 기존의 단일 프롬프트 과제를 문장 경계에서 **문맥 절반과 질의 절반으로 잘라** 만든 변형 벤치마크를 말한다. 자르기는 원 과제의 난이도를 보존하되, "문맥이 질의보다 먼저 도착한다"는 조건을 인공적으로 성립시킨다. 이 구성은 sleep/wake 분리를 측정 가능하게 만드는 동시에, 그 분리가 실제 워크로드에서 성립하는지를 **평가 밖으로 밀어낸다**.

표 14-4 — Letta `Sleep-time Compute`의 네 벤치마크. 전부 저자 제작이다.

| 벤치마크 | 크기 | 구성 방식 | 출처 |
|---|---|---|---|
| Stateful GSM-Symbolic P1 | 5000 examples | GSM-Symbolic P1(GSM8K에 절 1개를 더한 변형)을 의문문 앞에서 절단 | §4.1, p.4; Fig. 2, p.5 |
| Stateful GSM-Symbolic P2 | 2500 examples | GSM-Symbolic P2(절 2개 추가), 같은 절단 | §4.1, p.4 |
| Stateful AIME | 60 questions | AIME 2024 + AIME 2025 합본. 구두점으로 분리한 statement 중 마지막 하나를 질의로, 나머지를 문맥으로 | §4.1, p.4; App. J, pp.27–28 |
| Multi-Query GSM-Symbolic P1 | 12043 questions / 1095 contexts / 1095 original / 10948 generated | 템플릿당 1개 인스턴스 표집 후, o3-mini가 "같은 난이도, 같은 추론 단계 수"의 Q/A 쌍을 추가 생성 | Table 1, p.19; §4.1, p.4; App. C Fig. 19, p.19 |
| Multi-Query GSM-Symbolic P2 | 5497 questions / 500 contexts / 500 original / 4997 generated | 같은 방식 | Table 1, p.19 |
| SWE-Features | 33 examples (Aider-AI/aider 18 + comfyanonymous/ComfyUI 15) | `.py`/`.js` 파일 3개 이상을 수정한 PR 수집. gpt-4o-mini가 제목·본문을 심사하고, claude-sonnet-3-7-20250219가 PR을 군집화해 형제 PR을 문맥으로 삼는다 | §6, p.12; App. D, pp.19–20 |

표 14-4에 이 장의 첫 번째 의무 caveat이 이미 들어 있다. **5×·13%·18%·2.5×는 전부 이 표의 벤치마크에서 나온 값이다.** 논문 안에 기존의 stateful reasoning 벤치마크는 하나도 없다. 그 숫자들 뒤에 독립적인 평가 기준이 없다는 뜻이다.

Multi-Query는 기반 데이터셋의 약 5분의 1만 쓴다 — P1의 5000 예제 중 1095개 문맥, P2의 2500개 중 500개 문맥이며, 이유는 "템플릿당 한 인스턴스를 표집한다"이다[Letta STC Table 1, p.19; §4.1, p.4].

### 14.6.3 논문이 보고하는 수치

**5× (test-time compute 감소).** 초록의 표현은 "같은 정확도에 도달하는 데 필요한 test-time compute을 Stateful GSM-Symbolic과 Stateful AIME에서 약 5× 줄인다"이다[Letta STC Abstract, p.1]. 본문은 이 배율에 레짐 제한을 붙인다 — "낮은 test-time 예산에서, sleep-time compute의 성능이 baseline보다 유의하게 좋으며, baseline이 5배 많은 test-time 토큰으로 내는 성능과 비슷한 성능에 도달한다"[Letta STC §5.1, p.6]. 즉 5×는 **낮은 예산 구간의 수평 이동**이다.

**13% (Stateful GSM-Symbolic).** "pareto 곡선을 바깥으로 밀어 비슷한 test-time 예산에서 성능을 최대 13%까지 개선한다"[Letta STC §5.2, p.8; Fig. 7, p.10]. 이 값은 $\kappa_{\parallel}$ 스윕에서 나온다. 저자는 "더 어려운 과제와 더 강한 모델에서 이득이 가장 크다(예: P2 + gpt-4o)"고 덧붙인다.

**18% (Stateful AIME).** "sleep-time에 계산을 키우면 대체로 pareto 곡선이 바깥으로 이동하며 성능이 최대 18%까지 개선된다"[Letta STC §5.2, p.8]. 다만 내부 참조가 어긋난다 — [Letta STC §5.2]는 이 결과를 'Figure 26'으로 지목하는데, 그 그림의 캡션은 'Scaling sleep-time compute on Stateful AIME2025'(App. M, p.31)로 **연도별 부분집합**이다. 60문항 합본의 스케일링 그림은 Figure 8(p.10)이다.

**2.5× (질의당 평균 비용 감소).** "문맥당 질문이 10개일 때 단일 질의 baseline 대비 질의당 평균 비용을 최대 2.5× 줄일 수 있다"[Letta STC §5.3, pp.8–9; Fig. 9, p.11]. 조건은 Multi-Query GSM-Symbolic, $N_q=10$, $\kappa_{\mathrm{cost}}=10$, 그리고 생성 토큰만 세는 식 (14-7)이다.

**pass@$\kappa$ 대비 우위.** "모든 과제와 모든 모델에서, 같은 test-time 토큰 예산일 때 sleep-time compute이 pass@$k$ 병렬 스케일링을 일관되게 능가한다"[Letta STC §5.1, p.7; Figs. 5–6, pp.8–9]. 저자는 자기 baseline의 강점을 명시한다 — pass@$k$는 "test-time에 정답 검증기에 oracle로 접근한다는 비현실적 가정"을 하며, 그 가정을 하지 않는 방법이 그것을 이긴다면 유의미한 개선이라는 것이다.

**문맥만 주는 통제.** 부록 I는 질의를 지운 채 문맥만 준 baseline('ablate question')을 세워, 데이터셋의 질문이 문맥에서 자명하게 추측되지 않음을 보인다[Letta STC App. I, p.26; Figs. 21–22, pp.26–27].

> **[해설]** 위 두 항목은 이 논문이 잘한 일이다. oracle 우위를 가진 baseline을 세우고 그 우위를 본문에 밝힌 것, 그리고 "질문이 문맥에서 그냥 보이는 것 아니냐"는 가장 명백한 반론에 대한 통제를 부록에 실은 것. 이 책은 두 설계를 신뢰할 만한 것으로 본다. 아래의 caveat 목록은 이 두 항목을 지우지 않는다.

**SWE-Features.** "낮은 test-time 예산에서 sleep-time compute이 성능을 개선하며 test-time 토큰을 최대 약 1.5× 줄인다. 다만 test-time 예산이 높으면 test-time compute만 쓰는 쪽이 더 나을 수 있다"[Letta STC §6, p.12]. F1 축은 약 0.30–0.55, test-time 토큰 축은 3000–10000이다[Fig. 11, p.12].

**질의 예측가능성.** "질문이 문맥으로부터 더 예측 가능해질수록 sleep-time compute과 표준 test-time compute의 정확도 격차가 벌어진다"[Letta STC §5.4, pp.9–10]. 프로토콜은 Llama2-70B base 아래의 $\log P(q \mid E_0)$를 5분위로 나누고, GPT-4o-mini에 verbosity 0(가장 낮은 wake 예산)을 적용한 것이다. 정확도 차 축은 P1이 0.0–0.5, P2가 0.00–0.40이다[Fig. 10, p.11].

### 14.6.4 논문 자신이 보고하는 불리한 결과

**이득이 높은 test-time 예산에서 역전한다. 두 과제 모두에서.** Stateful GSM-Symbolic에서는 이렇다.

> 'However, at the test-tome [원문 오타] compute budgets, the test-time compute only baseline slightly outperforms sleep-time compute. We hypothesize that this may be because the standard test-time compute only has the content relevant to the specific question, so there is less distracting information in the prompt.' [Letta STC §5.1, p.6]

SWE-Features에서도 같은 역전이 나오고, 저자는 정밀도 쪽에서 그것을 관측한다 — "높은 test-time 예산 설정에서는 표준 test-time compute이 정밀도가 더 높고 재현율은 비슷하다"[Letta STC §6, p.12]. 두 경우 모두 **가설이 제시되고 시험되지 않는다.** 그리고 초록은 이 레짐 제한을 달지 않는다.

**sleep 스케일링이 비단조다.** "성능을 개선하는 병렬 에이전트 수에는 한계가 있는 듯하며, 병렬 생성 5개가 대체로 10개보다 낫다"[Letta STC §5.2, p.8]. 13%라는 헤드라인이 바로 이 스윕에서 나온다. 최적점이 구간 내부에 있고, 그것을 찾는 규칙도 임계도 제시되지 않는다.

**가장 강한 추론 모델이 거의 이득을 보지 못한다.** "모든 모델에서 유의한 test-time·정확도 pareto 이동을 관측했으나, o1은 예외로 이득이 제한적이다"[Letta STC §5.1, p.6; Fig. 4, p.7]. 설명은 제시되지 않는다.

**질의 수가 적으면 순손실이다.** "문맥당 질문 수가 적으면 총비용 관점에서 sleep-time compute이 덜 유리하다"[Letta STC Fig. 9 캡션, p.11].

### 14.6.5 이 책이 확인한 불리한 사실

**Stateful AIME의 구성에 수동 개입이 있고, 문맥이 비어 있는 인스턴스가 포함된다.**

> 'There are a couple of edge cases where the question is posed in e.g. the second to last statement rather than the last statement. In these cases, we manually rearrange the statements to ensure the query being used corresponds to the question. In a few cases, there is only one statement in the problem. In these cases, the context is empty.' [Letta STC App. J, pp.27–28]

LaTeX 그림도 수동으로 제거되었고, 제거 후에도 문제가 풀리는지를 수동으로 확인했다고 같은 부록이 밝힌다. 여기서 무거운 것은 **빈 문맥**이다. 문맥이 비어 있으면 sleep 국면이 읽을 것이 없으므로 sleep-time compute은 구성상 아무것도 할 수 없다. 그런 인스턴스가 몇 개인지 논문은 밝히지 않는다. 분모는 60이다 — 한 문항이 정확도 1.67점이다.

**Multi-Query 데이터셋의 90.9%가 모델 생성 문항이다.** P1은 12043개 중 10948개, P2는 5497개 중 4997개이며 두 비율 모두 90.9%다[Letta STC Table 1, p.19]. 생성된 답의 정확성을 검증했다는 보고는 논문 어디에도 없다. 그리고 논문 자신의 예시(Figure 20, p.20)를 보면 생성 문항 중 여럿이 문맥에서 숫자 하나를 옮겨 적으면 답이 되는 단일 조회형이다('How many stacking rings are on the tower?'). 상각 논증의 분모 $N_q$가 바로 그 문항들로 채워져 있다.

**모델 선택 기준이 측정 대상과 상관된다.** "각 데이터셋에서 우리는 test-time compute을 적게 썼을 때 성능이 나쁘지만 test-time compute을 키우면 개선되는 모델을 평가한다"[Letta STC §4.2, p.5]. 이것은 test-time-compute 대 정확도 기울기가 가파른 모델을 고른다는 뜻이고, 가파른 기울기는 정확히 5× 수평 이동을 크게 보이게 만드는 성질이다.

**baseline arm의 프롬프트가 공개되지 않았다.** 부록 A에 실린 다섯 개 verbosity 프롬프트(레벨 0–4)는 전부 sleep 산출물의 존재를 전제한다 — 모델에게 "rethink memory 블록을 확인하라", "rethink memory 블록에 이미 있는 것은 다시 계산하지 말라"고 지시한다[Letta STC App. A, Figs. 12–16, pp.16–17]. 대응하는 baseline arm의 프롬프트는 실려 있지 않다. 논문 텍스트만으로는 두 arm이 $\hat c$의 유무에서**만** 다른지 확인할 수 없다.

<!-- TODO-VERIFY: 벤더링된 텍스트 추출본에서 레벨 2(Figure 14, p.16)와 레벨 3(Figure 15, p.17) verbosity 프롬프트가 문자 단위로 동일하다. 사실이면 GPT-4o/4o-mini의 다섯 개 compute 동작점 중 둘이 같은 프롬프트로 만들어진 것이다. 추출 아티팩트일 가능성이 있으므로 본문 단정으로 쓰지 않는다. 확인 방법: arXiv 2504.13171 게시 PDF의 "Figure 15: Prompt for level 3 verbosity" 블록을 Figure 14와 문자 비교. -->

**SWE-Features는 테스트를 돌리지 않고 채점한다.** "PR을 GitHub에서 긁어 온 것이라 평가에 쓸 만한 직관적인 테스트가 없다. 대신 예측된 수정 파일 집합과 정답 수정 파일 목록을 비교해 F1을 보고한다"[Letta STC §6, p.12; App. D, p.21]. 에이전트는 Docker 안에서 테스트를 작성하라는 지시를 받지만 채점은 그것을 무시한다. 즉 지표는 **동작하는 코드**가 아니라 **파일 집합을 맞히는 것**에 보상을 준다. 그리고 이것이 이 방법의 이득이 가장 작고(약 1.5×) 고예산에서 역전하는 바로 그 지표다.

**같은 모델 계열이 과제를 정의하고 또 푼다.** claude-sonnet-3-7-20250219가 PR을 군집화해 어떤 선행 PR이 대상 PR의 '관련 문맥'인지를 정하고[Letta STC App. D, p.20], Figure 11에서 평가되는 에이전트도 Claude 3.7 Sonnet이다[Fig. 11 범례, p.12]. 무엇이 유용하게 관련된 문맥인지를 그것을 이용해야 할 모델이 정한다. 이에 대한 통제는 보고되지 않는다.

**같은 부록이 프롬프트 교란을 밝히고 ablation하지 않는다.** "현재 단계와 전체 단계를 명시적으로 알려 주면 에이전트 성능이 특히 저예산 설정에서 유의하게 개선된다는 것을 발견했다"[Letta STC App. D, p.21]. 저예산 구간은 정확히 이 방법이 이긴다고 주장되는 구간이다.

**지연이 한 번도 측정되지 않았다.** wall-clock, TTFT, ms/token 수치가 논문에 0개다. 동기 진술 — "답을 얻기 위해 잠재적으로 수 분을 기다리고 질의당 수십 달러까지 든다"[Letta STC §1, p.1] — 은 OpenAI 가격 페이지를 가리키는 각주로 지지되고, "지연 최적화 추론이 대략 10× 비싸다"는 진술은 Databricks 문서 페이지를 가리키는 각주로 지지된다. 초록의 문제 제기('high latency and inference cost')는 전적으로 생성 토큰 대리값 위에 서 있다.

**내부 참조 오류가 여럿이다.** [Letta STC §5.2]의 18% 지목이 부분집합 그림을 가리키는 것(위), [Letta STC §4.1]과 App. C가 'Table C'를 지시하는데 캡션은 'Table 1'인 것, [Letta STC §6]이 sleep arm 에이전트가 "test-time 국면에 더 많은 파일을 탐색했다"고 쓰는데 문맥상 sleep 국면인 것, 그리고 역전을 진술하는 [Letta STC §5.1] 문장에 'test-tome' 오타가 있는 것.

### 14.6.6 스케일 상한

**모델 규모 상한은 없다.** 모든 대상 모델이 파라미터 수 비공개 폐쇄 API다[Letta STC §4.2, p.5]. 논문 전체에서 파라미터 수가 진술된 모델은 Llama2-70B 하나이고, 그것은 예측가능성 분석의 도구이지 대상이 아니다[§5.4, p.9]. 방법이 모델 무관한 프롬프팅이므로, 규모에 대한 주장은 이 논문으로 확인되지도 반증되지도 않는다.

**학습 규모는 0이다.** 어떤 종류의 학습도 일어나지 않는다.

**평가 규모의 최대치**는 Multi-Query GSM-Symbolic P1의 12043 문항 / 1095 문맥이며, 그중 10948개(90.9%)가 생성 문항이다.

**문맥 길이 상한이 보고되지 않는다.** 문맥 길이도, 프롬프트 길이도, $|\hat c|$도 논문 어디에도 없다. GSM-Symbolic 문맥은 문장 몇 개이고(Fig. 2, p.5), SWE-Features 문맥은 PR 제목·본문·패치와 에이전트의 저장소 탐색이다(App. D, p.20). **긴 문맥 레짐이 한 번도 행사되지 않는다** — 동기 예시가 코드베이스와 문서 QA인 논문에서 이것은 큰 공백이다.

**현실성 상한**은 SWE-Features다. 저장소 2개에서 온 PR 33건이고, 테스트 없이 파일 집합 F1으로 채점된다. 이것이 유일한 비수학·비합성 평가이며, 동시에 이득이 가장 작고 고예산에서 뒤집히는 설정이다. 논문이 동기로 든 시나리오와 대조하면 — 큰 저장소 위의 coding assistant는 PR 33건으로, 대화 이력 위의 assistant는 대화 세션 0건으로 평가되었다.

> **[평가]** 위 목록의 길이가 이 논문의 기여를 지우지 않는다. 시간 비대칭을 이용한다는 발상 자체는 옳다 — 문맥이 질의보다 먼저 도착하는 워크로드가 실재하고, 그 시간 동안 노는 계산 자원이 실재하며, 그것을 쓰면 wake 국면의 생성 토큰이 준다는 것은 관측된 사실이다. 생성 토큰 축의 5×는 관측이다. 이 책이 반박하는 것은 그 관측이 아니라, **그 축이 비용 축과 같다는 암묵 등식**이다. 생성 토큰이 줄었다는 관측과 총비용이 줄었다는 결론 사이에는 prefill 항, 측정되지 않은 $B_s$, 단언된 $\kappa_{\mathrm{cost}}$, 그리고 미지의 $N_q$ 분포가 놓여 있다. 그 네 가지를 통과해야 등식이 성립한다.

---

## 14.7 Systems/serving 함의

독자의 1번 질문 — 그래서 decode에서 뭐가 바뀌는가 — 에 대한 답은 이 논문에서 유난히 깨끗하다. **decode 커널은 아무것도 바뀌지 않는다.**

**기제는 프롬프트 토큰 치환이고 그 이상이 아니다.** 논문의 표현은 "test-time에 $c$ 대신 새 문맥 $c'$을 제공할 수 있다"이다[Letta STC §3, p.4]. 서빙 스택의 변경은 논문 어디에도 기술되지 않는다. wake 국면의 요청은 종류상 평범한 요청과 같고, 다른 것은 프롬프트의 내용뿐이다.

**따라서 상태는 KV cache와 애플리케이션 텍스트 저장소에 내려앉고, 가중치에는 절대 내려앉지 않는다.** 결과로 **shared-weight batching이 보존된다.** ch01의 Rosetta 표가 말하는 대로, per-user fast-weight나 per-user 가중치 delta가 생기면 가중치를 공유하는 배치가 깨지고 grouped-GEMM 계열의 우회가 필요해진다(→ ch01, ch19; per-request $W$ 상태의 decode 비용 전개는 → NM 식 (M) 이후 장들). E-경로는 그 대가를 치르지 않는다. 이 논문은 batching도, KV cache도, prefix caching도 한 번 언급하지 않지만, 기제가 프롬프트 치환이라는 사실만으로 이 결론이 따라 나온다.

**$\hat c$는 한 문맥의 $N_q$개 질의에 걸친 공유 prefix다.** prefix caching이 있는 스택에서 상각은 경제적 비유가 아니라 문자 그대로의 KV 재사용이 된다.

> **[해설]** 여기서 논문의 비용모델이 양쪽으로 어긋난다. 식 (14-7)은 생성 토큰만 세므로 공유 prefix의 KV 재사용을 **절약으로 잡지 못하고**, 동시에 길어진 prefix의 prefill을 **비용으로도 잡지 못한다**. 즉 논문의 회계는 자기 방법의 비용을 과소평가하고 자기 방법의 절약도 과소평가한다. 두 오차의 부호는 반대이지만 크기를 비교할 재료가 논문에 없으므로, 상쇄 여부는 이 논문으로 판정되지 않는다.

**wake prefill이 오르고, 논문은 얼마나 오르는지 한 번도 경계 짓지 않는다.** 병렬 스케일링에서는 $\kappa_{\parallel}$개의 이어붙인 재표현이 매 질의의 프롬프트에 실린다(최대 10개)[Letta STC §5.2, p.8]. $|\hat c|$가 보고되지 않으므로 prefill 증가분은 논문에서 계산되지 않는다. 이것이 이 논문의 가장 큰 systems 공백이다 — **decode 토큰 절약을 주장하면서 경계 없는 prefill 토큰 비용을 측정하지 않는다.**

**sleep 작업은 배치 가능하고, 지연에 둔감하며, throughput으로 최적화되는 decode 잡이다.** 서빙 함대의 언어로 옮기면, 이 논문의 제품은 latency-critical decode를 background decode로 바꾸는 것이고, 논문은 그 전환에 10:1의 가격을 붙였다[Letta STC §5.3, p.8]. interactive tier의 일을 batch tier로 옮기는 낯익은 모양이다. 다만 그 10:1은 이 함대에서 측정된 값이 아니라 벤더 문서에서 인용된 값이다(§14.5.4).

**sleep의 wall-clock은 임계 경로 밖이어도 공짜가 아니다.** sleep 쓰기는 문맥당 최대 10회의 **순차** 모델 턴 사슬이다 — 각 호출이 직전 블록을 읽고 덮어쓴다[Letta STC App. K, p.27]. 여기에 병렬 궤적 $\kappa_{\parallel}$까지 곱하면, 최악의 구성은 첫 질의가 도착하기 전에 문맥 하나당 100회의 모델 턴이다. 함대가 단위 시간에 몇 개의 문맥을 예열할 수 있는지가 이 값으로 정해진다. 논문은 실제로 사용된 턴 수도, 처리율 수치도 보고하지 않는다.

**트래픽이 계층의 어디에 내려앉는가.** 애플리케이션 텍스트 저장소(Letta memory 블록 값)[Letta STC App. F Listing 1, p.22] → 토크나이저 → prefill → KV cache. HBM 상주 가중치도 아니고, optimizer 상태도 아니며, per-user 가중치 샤드도 아니다. 이것이 $\Theta$-경로와의 가장 날카로운 대비다 — E-경로의 쓰기 트래픽은 데이터베이스로 들어가는 텍스트이고, $\Theta$-경로의 쓰기 트래픽은 디바이스 메모리로 들어가는 가중치 delta다. **memory-device 기회를 만드는 쪽은 후자다**(→ ch26, ch29). 이 장에서 memory-centric 논증을 더 밀지 않는다. 이 논문에 그 논증을 지탱할 바이트 수치가 하나도 없기 때문이다.

**입장 제어 신호가 하나 제안되어 있고, 제안된 형태로는 작동하지 않는다.** 논문은 $\log P(q \mid E_0)$를 sleep-time compute이 이득을 낼 문맥의 신호로 제시하고, 그것으로 결과를 5분위로 나눈다[Letta STC §5.4, p.9; Fig. 10, p.11]. 계산 자체는 싸다 — 작은 동결 base 모델의 forward 한 번이면 된다.

> **[평가]** 그러나 이 신호는 sleep 결정을 게이팅할 수 없다. 신호가 $q$를 입력으로 요구하는데, $q$는 sleep 시점에 정의상 존재하지 않는다. 논문은 할당 정책을 future work로 명명하면서 이 순환을 관측하지 않는다. 서빙에서 이 신호를 쓰려면 $q$가 아니라 **$E_0$만으로 계산되는 대리 지표**로 바꿔야 하고, 그 변환은 논문에 없다.

---

## 14.8 한계와 bridge-out

### 14.8.1 논문 자신이 남긴 문제

논문의 [Letta STC §7](p.13)은 네 가지를 남긴다.

1. **어느 문맥이 예측 가능한 질의를 갖는가, 그리고 계산을 어떻게 배분하는가.** "흥미로운 향후 방향은 어떤 문맥이 예측 가능한 질문을 가질지 식별하고, 서로 다른 문맥과 질의에 걸쳐 sleep-time과 test-time 사이에 추론 계산을 최적으로 배분하는 것이다." 이 논문에는 **언제 잠들지를 결정하는 컨트롤러가 없다.**
2. **두 국면 가정을 넘어서기.** "실제 LLM 사용 사례는 더 복잡할 수 있고, 여러 라운드의 상호작용과 라운드 사이의 문맥 수정이 있다(예: 코드베이스에 대한 여러 번의 편집). 또한 sleep-time의 길이 자체도 상호작용마다 크게 다를 수 있다(사용자 타이핑 사이의 짧은 간격부터 며칠의 비활동까지)." 이것이 정확히 $\rho$의 질문이며, 저자들이 미해결이라고 명시한다.
3. **자연어 공간의 표현학습으로 밀 수 있는가.** §14.3.4에 인용한 문장이 여기서 나오고, 이 책의 3경로 분할의 1차 근거가 된다.
4. **$\hat c$를 합성 학습 데이터로 쓸 수 있는가.** "향후 연구는 sleep-time compute의 출력 자체를 합성 데이터의 한 형태로 쓰는 것을 탐색할 수 있다." 이것은 E-경로에서 $\Theta$-경로로의 명시적 인계다 — $\hat c$가 (U-$\Theta$)의 학습 집합 $\mathcal{R}_k$가 되는 구성이다. 저자들이 제안하고, 시험하지 않는다.

논문이 질문으로 세우지 않았으나 이 책이 미해결로 기록하는 것도 있다. 고예산 역전의 기제(가설만 있고 시험 없음), $\kappa_{\parallel}$ 비단조성의 임계, o1 예외의 원인, 실서빙의 $N_q$ 분포, $\kappa_{\mathrm{cost}}=10$의 타당성, 그리고 $\hat c$의 **검증·만료·축출·버전 관리** — 이 마지막 묶음은 논문 어디에서도 다루어지지 않는다. 쓰기 검증 단계도, 용량 정책도, staleness 처리도 없다.

### 14.8.2 경로 간 인용 여부

참고문헌은 22개다[References, pp.13–15]. 전수 확인 결과는 다음과 같다.

- **$W$-경로 인용: 0건.** fast weight, test-time training, 선형 attention, state-space model, recurrent memory, weight-space test-time 상태 — 어느 것도 없다. 'state'라는 단어는 'stateful application'의 뜻으로만 나온다. KV cache는 한 번도 언급되지 않는다.
- **$\Theta$-경로 직접 인용: 0건.** LoRA도, PEFT도, 모델 편집(ROME/MEMIT)도, 지속학습도, catastrophic forgetting도, knowledge distillation도 없다. 가장 가까운 이웃은 합성 데이터 두 편(Yang et al. 2024, Gunasekar et al. 2023)과 Bansal et al. 2024인데, 셋 다 [Letta STC §7]의 future work 문단에서 sleep 산출물의 **소비자**로만 등장하지 경쟁 기제로 등장하지 않는다. 다른 두 공간을 이름으로 부르는 유일한 문장(§14.3.4 인용)이 그 공간의 어떤 연구도 인용하지 않고 2014년 리뷰 하나만 단다.
- **생물학·CLS 인용: 0건.** McClelland도, hippocampal replay도, sleep consolidation 신경과학도, dreaming 문헌도 없다. 이 논문의 중심 은유인 'sleep'에 생물학 인용이 **하나도** 붙어 있지 않다. [Letta STC §1](p.2)은 이 용어를 순전히 조작적으로 도입한다 — "모델과의 상호작용 사이, 그렇지 않았으면 놀고 있었을 sleep-time에 추론을 수행한다." 단어는 **유휴**를 위해 선택된 것이지 consolidation을 위해 선택된 것이 아니다.
- **E-경로 내부 인용: 정확히 1건.** Packer et al. 2023(MemGPT), 같은 저자·같은 회사, [Letta STC §2]의 OS/pre-fetching 유비에서. RAG도, vector database도, graph memory도, 검색 문헌 어느 것도 참고문헌에 없다 — $\mathrm{ret}$을 항등으로 두는 설계 선택은 검색 문헌이 반대할 만한 선택인데도 그렇다.
- **논문이 실제로 교전하는 것:** test-time scaling(Snell 2024, OpenAI 2024, DeepSeek-AI 2024, Brown 2024, Muennighoff 2025), speculative decoding(Leviathan 2023, Stern 2018, Cai 2024, DeepSeek-V3), 고전적 systems 선계산(Smith 1982 캐시 메모리, Gray 1997 data cube).

이 네 건의 침묵을 두 종류로 갈라야 한다. 어떤 침묵은 **연대가 강제한다** — 인용될 수 없는 것을 인용하지 않은 것은 계보 사실이지 선택이 아니다. 어떤 침묵은 **선택된다** — 인용할 수 있었는데 하지 않은 것이다. 이 논문의 발표 시점은 2025년 4월이고, 위 네 묶음은 **전부 후자다.** $W$-경로의 주류 계보(→ ch16, ch17), $\Theta$-경로의 쓰기 primitive인 LoRA(2106.09685)와 모델 편집(2202.05262, 2210.07229)(→ ch19), 망각 대응의 원형인 EWC(1612.00796)(→ ch07), CLS 이론과 생물학적 sleep 실험(→ ch08, ch23), 그리고 E-경로 내부의 RAG(2005.11401)(→ ch10)와 Memorizing Transformers(2203.08913)(→ ch12) — 어느 것도 이 논문보다 뒤에 나오지 않았다. 인용 가능한 상태로 전부 존재했다.

> **[평가]** 이 라인의 이름을 만든 논문은 자기를 캐시와 speculative decoding의 후손으로 놓았지 기억 연구나 학습 연구의 후손으로 놓지 않았다. 그리고 뒤에 'sleep-time compute'이라는 이름을 쓴 $W$-경로·$\Theta$-경로의 논문들은, 자기를 인용한 적 없고 생물학을 인용한 적도 없는 논문에서 그 이름을 물려받았다. 이름의 충돌은 공유된 이론적 약속이 아니다. 이 관측이 ch11의 판별식과 ch27의 판정에서 다시 쓰인다.

### 14.8.3 다음 장이 받아가는 것

ch15는 이 장이 세운 (U-E)의 특수화를 프로덕션 시스템 쪽으로 가져간다. 이 장에서 항등이었던 $\mathrm{ret}$이 거기서는 진짜 검색기가 되고, 이 장에서 파괴적 덮어쓰기 하나였던 $\mathrm{wr}$이 거기서는 연산자 집합으로 갈라지며, 이 장에서 문맥과 함께 죽던 $E$가 거기서는 세션을 넘어 지속된다. 그리고 이 장이 라운드 1로 고정해 둔 $k$가 거기서는 늘어나기 시작한다.

> **[평가]** 그런데 ch15가 다루는 논문들은 **이 논문을 인용하지 않는다.** 같은 경로의 후속 연구가 그 경로에 이름을 준 논문을 보지 않는다. §14.8.2가 보인 침묵이 바깥으로만 향한 것이 아니라 안쪽에서도 되돌아온다는 뜻이다. 그 단절 자체가 ch15의 재료이며, 연대가 강제한 침묵이 아니라 선택된 침묵이라는 점에서 ch11·ch27이 다시 쓴다.

이 장이 채우지 못하고 넘기는 것은 표 14-3의 네 칸이다. 미리 말해 두면 **ch15도 그 칸을 채우지 못한다.** 채우려는 시도가 이 책의 실험 X1(비용)과 X2(용량)이고, 그 위에서의 판정은 ch25와 ch26이 소유한다.

---

## 요약

- Letta의 `Sleep-time Compute`(2504.13171)는 상속받은 open question 없이 시작한다. 참고문헌 22개이며, 계보 진술은 같은 저자의 MemGPT에 대한 운영체제 pre-fetching 유비 하나뿐이다[§2, p.3]. 논문은 문제를 물려받는 대신 test-time scaling 문헌에 stateless 가정을 부여해 공백을 제조한다[§1, p.1].
- 기제는 (U-E) 하나이며, 네 겹으로 특수화되어 있다: $\mathrm{wr}$은 파괴적 덮어쓰기, $S(\cdot;B_s)$는 깊이 $J\le10$의 반복 자기 재작성, $\mathrm{ret}$은 항등, $E$는 문맥 범위이고 휘발성이다. $\Theta$는 동결이고 $W$는 존재하지 않는다. 논문에는 번호 붙은 수식이 하나도 없다.
- learned context $\hat c$의 'learned'는 gradient를 뜻하지 않는다. 논문 본문이 스스로 정확하게 쓴다 — 표현이 형성되는 공간이 parameter도 activation도 아닌 자연어라는 것이고, 그 문장이 이 책 3경로 분할의 1차 근거다[§7, p.13].
- 비용 4종이 **전부 논문에 없다.** $B_s$·$L_w$·$C_{\mathrm{cap}}$·$\rho$ 어느 것도 숫자로 보고되지 않는다. 상각 논증의 분자 $B_s$가 없으므로 손익분기 $N_q^{*}$를 식 (14-8)로 계산할 수 없고, 논문은 $N_q \in \{1,2,5,10\}$ 네 점의 곡선과 "질의 수가 적으면 불리하다"는 정성 진술만 준다[Fig. 9 캡션, p.11].
- 비용모델 (14-7)은 생성 토큰만 센다. prefill 항이 없는데 이 방법의 기제 자체가 입력을 길게 만드는 것이며, $\kappa_{\parallel}=10$에서는 열 개의 재표현이 매 질의마다 prefill된다. $\kappa_{\mathrm{cost}}=10$은 벤더 문서 인용이고 민감도 분석이 없다.
- 5×·13%·18%·2.5×는 전부 저자 제작 벤치마크(Stateful GSM-Symbolic, Stateful AIME, Multi-Query GSM-Symbolic)에서 나온 값이다. Stateful AIME 60문항에는 문맥이 빈 인스턴스가 개수 미상으로 포함되고 수동 재배열이 있었으며[App. J, pp.27–28], Multi-Query의 90.9%가 모델 생성 문항이고 생성 답의 검증은 보고되지 않았다[Table 1, p.19].
- 이득은 높은 test-time 예산에서 역전한다. Stateful GSM-Symbolic과 SWE-Features 두 과제 모두에서이며, 기제는 제시되지 않았다[§5.1, p.6; §6, p.12]. sleep 스케일링도 비단조여서 $\kappa_{\parallel}=5$가 10을 이긴다[§5.2, p.8].
- 서빙 관점에서 decode 커널은 아무것도 바뀌지 않는다. 상태가 가중치에 내려앉지 않으므로 shared-weight batching이 보존되고, 그 대신 $\Theta$-경로가 만드는 종류의 memory-device 기회도 생기지 않는다. 대가는 경계 없는 wake prefill 증가이며 논문은 그것을 측정하지 않는다.

## 자가 점검 체크리스트

- [ ] 이 논문이 갱신하는 층이 (U-E) 하나뿐이고, 그 (U-E)가 정규형과 어긋나는 지점 다섯 개를 말할 수 있는가($\mathrm{wr}$ 파괴적 덮어쓰기 / $S(\cdot;B_s)$ 반복 / 두 번째 $\mathrm{wr}$인 concat / $\mathrm{ret}$ 항등 / $E$ 휘발).
- [ ] 식 (14-8)을 쓰고, 우변 세 값 중 어느 것이 논문에 있고 어느 것이 없는지 지목할 수 있는가. 없는 값 때문에 $N_q^{*}$가 계산되지 않는 이유를 한 문장으로 말할 수 있는가.
- [ ] "생성 토큰이 5× 줄었다"와 "비용이 5× 줄었다" 사이에 놓인 네 가지 — prefill 항, 미보고 $B_s$, 단언된 $\kappa_{\mathrm{cost}}$, 미지의 $N_q$ 분포 — 를 열거할 수 있는가.
- [ ] 5×·13%·18%·2.5×가 각각 어느 벤치마크, 어느 레짐의 값인지 말할 수 있는가. 그리고 이득이 역전하는 레짐을 두 과제 모두에서 지목할 수 있는가.
- [ ] 'learned context'에서 무엇이 학습되지 **않는지**, 그리고 논문 본문이 그 사실을 어디에서 스스로 밝히는지 말할 수 있는가.
- [ ] 이 논문이 $W$-경로·$\Theta$-경로·생물학을 각각 몇 건 인용하는지 말할 수 있는가. 그 답이 왜 ch11의 재료가 되는지 설명할 수 있는가.
- [ ] **Rosetta**: prefix cache와 $\hat c$의 차이를 재사용의 종류로 설명할 수 있는가(같은 계산의 재사용 대 새 추론의 선행 수행). 그리고 이 논문에서 shared-weight batching이 깨지지 않는 이유를, per-user 가중치 delta가 생기는 경우와 대조해 말할 수 있는가.

