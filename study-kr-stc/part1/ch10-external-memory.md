# ch10. 외부 기억과 검색 비용

> **이 장의 목표** — (1) $E$층의 세 구현 형태(text·vector·graph)를 구분하고 각각의 $\mathrm{ret}(E,q)$가 무엇을 계산하는지 말한다. (2) RAG를 식 (R)과 (U-E)로 환원해, 이 논문이 **읽기**만 완성하고 **쓰기**를 비워 두었다는 사실을 수식으로 확인한다. (3) $\mathrm{ret}(E,q)$의 비용을 검색·prefill·쓰기 세 항으로 분해하고, 그중 상각식 (A)의 분모에 들어가는 항이 하나뿐임을 계산한다. (4) chunk 크기 $C$와 회수 폭 $K_{\text{ret}}$이 서빙 비용의 어디를 각각 움직이는지 자기 config 숫자로 판정한다.
>
> **왜 필요한가** — Part II의 $E$-경로 네 장(ch12 Memorizing Transformers, ch13 RAG→MemGPT, ch14 Letta `Sleep-time Compute`, ch15 Mem0·Zep·ReasoningBank·SCM)이 전부 ch01이 정의하고 이 장이 형태와 비용을 채우는 $E$ 위에서 돈다. 특히 ch14의 상각 논증은 식 (A)를 쓰는데, (A)의 분자에 들어갈 항과 분모를 통과하지 못하는 항을 가르는 것이 이 장의 §10.5다. ch25(통일 비용 회계)는 그 분해를 회계 단위로 그대로 쓴다.
>
> **NM과의 관계** — Neural Memory에는 이 장에 대응하는 장이 **없다**. NM은 $W$층만 다뤘고 $E$는 v2에서 신설한 층이다. 따라서 이 장은 NM에서 아무것도 넘겨받지 않고, 대신 NM이 세운 회계 도구를 빌려 쓴다: prefill/decode 비대칭과 state 트래픽 회계는 → NM ch10 §10.2·§10.4, KV cache와 고정 크기 state의 crossover 산수는 → NM ch10 §10.3, chunk 크기를 roofline 독립변수로 놓는 관점은 → NM ch09·ch10. 이 장이 더하는 것은 세 가지다 — 모델 **밖** 저장소의 형태 분류, 검색 연산의 비용 구조, 그리고 상각으로 사라지지 않는 항의 식별.

<!-- TODO-VERIFY: RAG의 게재 학회. vendored plate에는 arXiv v4 스탬프만 있고 venue 줄이 없다. 확인 방법: grep -in 'neurips\|advances in neural\|conference on neural' papers/stc/2005.11401.txt, 또는 arXiv 2005.11401v4 초록 페이지의 comments 필드. 확정 전까지 본문에 학회명을 쓰지 않는다. -->

## 10.1 $E$는 무엇으로 만들어지는가 — text·vector·graph

$E$는 하나의 자료구조가 아니라 세 구현 형태의 총칭이며, 형태를 가르는 것은 저장 방식이 아니라 $\mathrm{ret}(E,q)$가 무엇을 계산하는가다.

> **정의.** $E$ 자체의 정의는 ch01이 소유한다 — 모델 **밖**에 텍스트·벡터·그래프로 저장되는 기억이다(→ ch01). 이 장이 소유하는 것은 그 **구현 형태**이고, 형태는 세 가지다. **text store**는 레코드를 원문 문자열로 저장하고, 질의와 레코드의 어휘 겹침(BM25 등) 또는 전체 스캔으로 읽는다. **vector store**는 레코드를 고정 길이 실수 벡터로 바꿔 저장하고, 질의 벡터와의 유사도 상위 몇 개를 읽는다. **graph store**는 레코드를 노드로, 레코드 사이의 관계를 엣지로 저장하고, 순회로 읽는다. 세 형태는 배타적이지 않다 — 실제 시스템은 같은 레코드를 둘 이상의 형태로 동시에 색인한다.

이 장이 다루는 RAG는 vector store와 text store를 겹쳐 쓴다. 저장된 것은 원문 텍스트이고("A key feature of our memory is that it is comprised of raw text rather distributed representations" [RAG §5]), 검색은 그 텍스트의 embedding 위에서 일어난다. graph store 형태의 실제 시스템은 ch15가 소유하므로 여기서는 형태만 정의하고 넘어간다.

> **[해설]** 세 형태의 차이는 독자에게 익숙한 언어로는 **접근 패턴의 차이**다. text store는 inverted index 조회 — 정수 posting list의 순차 읽기와 교집합. vector store는 고차원 벡터 배열의 무작위 접근. graph store는 포인터 추적. 셋 중 어느 것도 GEMM이 아니고, 셋 다 arithmetic intensity가 낮다. 독자가 커널 최적화에서 익힌 직관 — tiling으로 재사용률을 올린다 — 은 이 세 형태 어디에도 그대로 이식되지 않는다. 그리고 셋 다 HBM이 아니라 host DRAM 또는 NVMe에 산다. $E$는 정의상 가속기 밖의 층이며, 이 위치가 §10.5의 비용 분해를 강제한다.

$C_{\text{cap}}$의 단위에는 처음부터 주의가 필요하다. RAG의 $E$는 전역이고 읽기 전용이며 모든 사용자가 같은 것을 본다. 따라서 이 논문의 $C_{\text{cap}}$은 **corpus당** 값이다 [RAG App. C]. 반면 이후 $E$-경로 논문들이 보고하는 상태량은 **사용자당·세션당**이다(→ ch13·ch15). 단위가 다른 두 수를 나란히 놓으면 비교가 성립하지 않는다. 이 장은 corpus당 값만 다룬다.

## 10.2 RAG — 읽기 경로에 붙은 $E$

RAG는 $E$를 일급 객체로 만든 논문이고, 동시에 $E$에 **누가 쓰는가**를 통째로 비워 둔 논문이다.

> **정의.** **RAG**(retrieval-augmented generation)는 사전학습된 seq2seq generator에 사전학습된 dense index를 붙이고, 검색된 레코드를 잠재변수로 두어 그 위에서 주변화한 우도를 최대화하도록 generator와 질의 encoder를 함께 fine-tuning하는 방법이다 [RAG Abstract, §1]. arXiv:2005.11401.

통일 표기로 환원하면 RAG는 식 (R)에서 $W$ 자리가 비어 있는 형태다.

$$
\hat y \;=\; f\big(q;\ \Theta,\ \varnothing,\ \mathrm{ret}(E,q)\big)
\tag{10-1}
$$

이 식이 말하는 것은 하나다 — RAG에서 test time에 움직이는 것은 파라미터가 아니라 **뽑혀 나온 레코드의 정체**뿐이다. 논문에는 fast weight가 없다. corpus 전체에서 $W=\varnothing$이 가장 깨끗하게 성립하는 사례이므로, $W$-경로 장들은 RAG를 영점으로 쓸 수 있다.

주변화를 어디서 하느냐로 두 변형이 갈린다. RAG-Sequence는 답 전체에 대해 레코드 하나가 책임지고(주변화가 토큰 곱 **밖**), RAG-Token은 토큰마다 다른 레코드가 책임질 수 있다(주변화가 토큰 곱 **안**) [RAG §2.1]. 이 논문의 표시 수식에는 번호가 없으므로 이 책은 절 위치로만 인용한다.

$E$ 쪽 갱신을 (U-E)에 넣어 보면 퇴화가 눈에 보인다.

$$
E_{k+1} \;=\; \mathrm{wr}\big(E_k,\ \hat c_k\big),
\qquad
\hat c_k \;=\; S\big(c_k;\ B_s\big) \;=\; c_k,
\qquad
B_s = 0
\tag{10-2}
$$

식 (10-2)가 말하는 것은 RAG의 $S(\cdot;B_s)$가 추론을 전혀 하지 않는다는 것이다. 저장되는 물건은 원문 chunk 그 자체이고, 요약도 파생 사실도 없다 [RAG §3, §5]. 그리고 $\mathrm{wr}$을 실행하는 주체가 모델이 아니다 — 쓰기는 corpus 파이프라인이나 사람이 하고, 그것도 레코드 삽입이 아니라 **index 전체 교체**로만 한다 [RAG §4.5]. 논문 자신의 표현으로 이 기억은 "human-readable"이고 "human-writable"이다 [RAG §5]. 빠진 세 번째 성질이 machine-writable이고, $E$-경로의 이후 연구 전부가 그 빈칸을 채우는 작업이다(→ ch13·ch14).

index 전체 교체가 실제로 작동한다는 증거는 하나 있다. 2016년 12월과 2018년 12월 사이에 재임자가 바뀐 세계 지도자 82명을 "Who is {position}?" 템플릿으로 물었을 때, index와 세계가 맞으면 70%(2016 index/2016 지도자)와 68%(2018/2018)를 맞히고, 어긋나면 12%(2018 index/2016 지도자)와 4%(2016 index/2018 지도자)로 떨어진다 [RAG §4.5]. 이 값들이 이 장에서 유일하게 $\rho$에 가장 가까운 측정치다 — 다만 이것은 반복 sleep 라운드의 열화율이 아니라 저장소 신선도의 함수다. 82항목 템플릿 프로브라는 규모도 함께 기록해 둔다.

> **[해설]** 독자의 어휘로 번역하면, RAG는 세계 지식이 바뀔 때 **배포 아티팩트를 건드리지 않는다**. checkpoint를 다시 쓰지 않고, 서빙 가중치를 교체하지 않으며, 사용자별 delta가 생기지 않는다. 그래서 shared-weight batching이 그대로 살아 있다 — 배치 안의 모든 요청이 같은 $\Theta$와 같은 index를 쓰고, 요청마다 다른 것은 프롬프트에 실려 들어오는 토큰뿐이다. Rosetta 표의 "모델 재배포 ↔ (U-$\Theta$) 1회" 행이 여기서 왜 결정적인지가 드러난다: $E$-경로는 그 행을 아예 밟지 않는다.

$E$를 붙여서 무엇을 사는가에 대한 이 논문의 답은 **용량**이다. RAG-Sequence는 Natural Questions에서 44.5 EM을 기록하는데, 파라미터 수가 가장 가까운 T5-large(770M)는 같은 벤치에서 28.9 EM이고 [RAG App. G], 훨씬 큰 T5-11B의 closed-book 점수도 34.5에 그친다 [RAG Table 1]. 즉 21M개 텍스트 레코드가 [RAG §3], 수십억 파라미터를 더 쌓아서 얻었을 지식을 대신 들고 있다.

이 비교에는 논문이 직접 밝힌 제한이 셋 붙는다. 첫째, T5 수치는 재실행이 아니라 발표된 값을 옮긴 것이다 [RAG App. G]. 둘째, RAG 쪽 생성기는 BART-large가 상한이고([RAG §2.3]은 400M, [RAG App. G]는 406M으로 두 값을 인쇄한다), 수십억 파라미터 생성기 위에서도 이 우위가 남는지는 논문이 답하지 않는다. 셋째, 검색기는 Natural Questions와 TriviaQA에 검색 감독을 받은 DPR로 초기화되어 있고, 그 둘이 헤드라인 SotA를 주장하는 바로 그 벤치다 — 논문 자신이 이 사실을 적는다 [RAG §4.1]. $\Theta$의 용량 상한을 다룬 ch09의 논지가 여기서 반대편에서 확인된다 — $E$-경로의 창립 논거는 "$\Theta$에 다 넣지 말고 밖에 두라"이며, 그 대가로 지불하는 것이 §10.5의 세 항이다.

이 논문은 generator 가중치를 "parametric memory", index를 "non-parametric memory"라 불렀다 [RAG §2.2, §2.3]. 이 이분법이 문헌에 들어온 자리가 여기다.

> **[평가]** 이 책은 그 이분법을 쓰지 않고 $\Theta$와 $E$로 바꿔 쓴다. 이유는 정확도다 — 이후 경로들은 $W$라는 세 번째 항을 갖는데, 그 항은 "parametric"에도 "non-parametric"에도 들어가지 않는다. fast weight는 파라미터의 모양을 하고 있으므로 "non-parametric"이 아니고, 배포 아티팩트에 속하지 않으므로 논문이 뜻한 "parametric"도 아니다. 두 칸짜리 어휘로는 세 층을 셀 수 없다. ch01이 세운 3층 표기가 이 장 이후 전 장의 기준이다.

## 10.3 chunking — 저장 단위를 정하는 결정

검색은 레코드 단위로 일어나므로, corpus를 어떻게 자르느냐가 검색기 성능보다 먼저 결정되는 변수다.

> **정의.** **chunking**은 corpus를 검색 단위(레코드)로 자르는 결정이다. 단위 하나의 크기를 $C$로 쓴다. $C$는 예약 기호이며, NM에서는 시퀀스를 나누는 단위였고 이 장에서는 corpus를 나누는 단위다 — 둘 다 "한 번에 묶어 다루는 토큰 수"라는 같은 역할을 한다. 용량은 어떤 경우에도 $C_{\text{cap}}$으로 쓰고 맨 $C$로 쓰지 않는다. §10.5가 쓰는 $C_{\text{search}}$·$C_{\text{prefill}}$·$C_{\text{write}}$도 같은 규약을 따르는 비용 항이며, 첨자 없는 $C$만이 chunk 크기다.

RAG의 chunking은 단순하다. 2018년 12월 Wikipedia dump의 각 문서를 겹치지 않는 100-word chunk로 잘라 총 $n = 21\mathrm{M}$개 레코드를 만든다 [RAG §3]. 즉 $C = 100$ words이고, 이 선택은 선행 open-domain QA 연구의 관행을 따른 것이다 [RAG §3].

$C$가 정하는 것은 세 가지이고, 셋의 방향이 서로 반대다.

1. **레코드 수** $n \propto 1/C$. $C$를 절반으로 줄이면 색인 대상이 두 배가 되고, 벡터 저장 용량 $C_{\text{cap}}$도 두 배가 된다.
2. **회수당 프롬프트 질량** $K_{\text{ret}}\cdot C$. 이것이 §10.5의 두 번째 비용 항이며, $C$에 정비례한다.
3. **레코드의 자족성**. $C$가 작으면 문장이 경계에서 잘리고 대명사의 선행사가 다른 chunk에 남는다. 검색이 정답 chunk를 뽑아도 그 chunk만으로 답이 나오지 않는 실패가 여기서 나온다. 세 번째 항목은 논문이 보고한 실패가 아니라 chunking 정의로부터의 연역이다 — RAG는 겹치지 않는 절단을 쓰므로 [RAG §3] 경계에서 잘린 문장을 복구할 경로가 정의상 없다. 이 실패를 계측한 논문은 corpus에 없다.

세 방향이 반대이므로 $C$에는 최적값이 있고, 그 최적값은 corpus와 질의 분포에 의존한다.

> **[평가]** RAG는 $C=100$을 고정하고 그 knob을 스윕하지 않는다. 논문이 스윕한 것은 회수 폭 $K_{\text{ret}}$뿐이다(Figure 3, $K_{\text{ret}}\in[10,50]$) [RAG §4.5]. 그런데 §10.5에서 보듯 서빙 비용의 두 번째 항은 $K_{\text{ret}}$과 $C$의 **곱**에 비례한다. 즉 논문은 곱의 한 인자만 움직여 보고 다른 인자는 관행값에 두었다. $E$-경로 전체가 이 관행값을 물려받았고, 이후 어느 논문도 $C$를 비용 축으로 놓지 않는다. 원문 전체에서 chunk 크기가 인쇄되는 자리는 [RAG §3]의 한 문장뿐이며, 100 words 이외의 값이 등장하는 실험은 없다.

systems 접점 하나로 이 절을 닫는다. $C$는 독자가 아는 tile 크기와 다르다. FlashAttention의 tile은 계산되는 함수를 바꾸지 않지만, chunking은 **검색 대상 집합 자체를 바꾼다**. $C$를 바꾸면 같은 질의에 다른 레코드가 뽑히고, 프롬프트 내용이 달라지고, 답이 달라진다. $E$층에서 $C$는 성능 knob이 아니라 semantic knob이다.

## 10.4 embedding과 유사도 검색

vector store의 $\mathrm{ret}(E,q)$는 두 개의 encoder와 하나의 top-$K_{\text{ret}}$ 연산으로 완전히 결정된다.

> **정의.** **embedding**은 텍스트를 고정 길이 실수 벡터로 보내는 함수다. 저장 쪽 함수를 $\phi_d$, 질의 쪽 함수를 $\phi_q$로 쓰고, 출력 폭을 $d_e$로 쓴다($d$는 모델 폭으로 예약되어 있으므로 구분한다). **유사도 검색**은 질의 벡터에 가장 가까운 저장 벡터 $K_{\text{ret}}$개를 찾는 연산이며, 가까움을 내적으로 재면 maximum inner product search(MIPS)다. $K_{\text{ret}}$은 회수 폭을 가리키는 이 장의 기호다 — 예약 기호 $k$는 sleep 라운드 첨자이므로 회수 개수에 절대 쓰지 않는다.

RAG의 검색기는 이 형태 그대로다 [RAG §2.2].

$$
\mathrm{ret}(E,q) \;=\; \underset{e\,\in\,E}{\mathrm{top}\text{-}K_{\text{ret}}}\ \ \exp\!\big(\phi_d(e)^{\top}\phi_q(q)\big)
\tag{10-3}
$$

이 식이 말하는 것은 검색이 학습 가능한 대상이 되었다는 것이다. 어느 레코드가 정답인지에 대한 감독 없이, 과제 loss의 gradient가 주변화를 통과해 $\phi_q$까지 흘러간다 [RAG §2.4]. 그러나 두 encoder의 지위는 대칭이 아니다. $\phi_q$는 $\Theta$의 일부로 학습되고, $\phi_d$는 DPR에서 가져와 **동결된다** — 논문의 문장은 "we ... keep the document encoder (and index) fixed, only fine-tuning the query encoder and the BART generator"이다 [RAG §2.4].

동결의 이유가 곧 $E$층의 첫 번째 구조적 제약이다. $\phi_d$를 한 번 바꾸면 저장된 $n$개 벡터를 **전부** 다시 계산해야 한다. 논문도 그렇게 쓴다 — document encoder를 훈련 중에 갱신하는 것은 index를 주기적으로 다시 만들어야 해서 비싸다는 것이다 [RAG §2.4]. 비용은 주장되고 측정되지 않는다. 반면 $\phi_q$는 공짜로 바꿀 수 있다. 읽기 함수의 두 반쪽이 서로 다른 가격표를 달고 있다는 이 비대칭은 $E$층 설계 전체를 지배한다.

검색 자체의 비용 구조는 exhaustive 형태에서 가장 선명하다. 전수 MIPS는 $n$개 벡터를 전부 읽어 내적을 계산하므로 $2 n d_e$ FLOPs를 쓰고 **저장된 벡터 배열 전체**를 트래픽으로 소비한다. 원소당 $b$ bytes라 하면 arithmetic intensity는

$$
I \;=\; \frac{2 n d_e}{n d_e b} \;=\; \frac{2}{b}\ \ [\text{FLOP/byte}]
\tag{10-4}
$$

이다. 식 (10-4)가 말하는 것은 검색의 intensity가 **레코드 수에도 벡터 폭에도 의존하지 않는다**는 것이다. 오직 저장 정밀도만이 정한다: fp32면 0.5, 8비트면 2 FLOP/byte. bf16 GEMV의 1 FLOP/byte와 같은 자릿수이며, 독자가 아는 decode GEMV가 이미 bandwidth-bound라는 사실(→ NM ch10 §10.6)이 그대로 이 연산에도 적용된다. 전수 검색은 계산이 아니라 **대역폭 문제**다.

식 (10-4)에는 반직관적인 따름정리가 하나 붙는다. index를 압축하면 트래픽은 줄지만 arithmetic intensity는 **올라간다** — $b$가 작아지기 때문이다. RAG가 보고한 CPU 메모리 100 GB → 36 GB 압축은 [RAG App. C] index가 상주하는 바이트를 2.8배(이 책의 나눗셈) 줄인다는 뜻이고, 동시에 그 위의 검색을 대역폭 한계에서 조금 더 떼어 놓는다는 뜻이다. 벡터 양자화가 $E$층에서 표준이 된 이유는 저장 공간이 아니라 이 두 효과의 결합이다.

ANN이 이 그림을 바꾸는 방식은 트래픽의 양이지 성질이 아니다. RAG는 FAISS에 Hierarchical Navigable Small World 근사를 얹어 방문 벡터 수를 $n$보다 훨씬 작게 줄이고 [RAG §3], 논문은 MIPS가 "sub-linear time"에 근사적으로 풀린다고 쓴다 [RAG §2.2]. 그러나 방문 순서는 질의에 따라 달라지는 그래프 순회이므로 접근이 무작위다. 읽는 양은 줄고 접근 지역성은 나빠진다.

systems 접점은 위치다. RAG는 index 벡터를 CPU에 둔다 — "doing Maximum Inner Product Search with FAISS is sufficiently fast on CPU, so we store document index vectors on CPU" [RAG App. C]. 무압축 약 100 GB, FAISS 압축 후 36 GB이고, 8비트로 저장할 수 있다고 부록이 덧붙인다 [RAG App. C, App. G]. 즉 $E$는 HBM 밖 host DRAM에 사는 read-mostly 구조이고, GPU가 400M 파라미터 생성기를 [RAG §2.3] 붙들고 있는 동안 CPU가 그래프를 걷는다. 독자의 회계로 옮기면 TTFT 안에 **프롬프트 길이가 아니라 index 크기에 비례하는 host-side 항**이 하나 들어온 것이다. 논문은 이 항을 한 번도 측정하지 않는다.

보고된 이 footprint와 식 (10-4)의 $n d_e b$는 같은 수가 아니며, 이 장은 둘을 구분해 쓴다. App. G는 index가 21M개 벡터, 총 $15.3\mathrm{B}$개 값으로 이루어진다고 인쇄한다 [RAG App. G] — 즉 $n d_e = 1.53\times10^{10}$이다. 이 곱을 fp32로 저장하면 벡터 배열 자체는 $n d_e b \approx 61$ GB이고(이 책의 곱셈), 8비트로 저장하면 15 GB다. 논문이 보고하는 값은 무압축 약 100 GB, 압축 후 36 GB로 양쪽 다 그보다 크다 [RAG App. C]. 차이의 이유는 회계 대상이 다르다는 것이다 — 보고값은 "all of Wikipedia"에 대한 **index 전체의 CPU 상주 메모리**이고, 여기에는 벡터 배열 외에 [RAG §3]이 얹은 HNSW 그래프와 저장된 원문이 함께 들어간다 [RAG §3, App. C]. 논문은 이 100 GB를 항목별로 분해하지 않는다. 따라서 이 장은 논문 수치를 인용할 때는 100 GB/36 GB를 그대로 쓰고, 식 (10-4)를 돌릴 때는 벡터 배열만 세는 이 책의 예시 구성을 쓴다. 폭 $d_e$ 자체는 인쇄하지 않는다 — 곱 $n d_e$를 논문이 직접 인쇄하고 아래 계산이 곱만으로 전부 성립하므로, 폭을 확정할 이유가 없다.

## 10.5 $\mathrm{ret}(E,q)$의 비용 — 세 항으로의 분해

검색은 공짜가 아니고, 비용이 한 곳에 있지도 않다. 질의 하나가 $E$층에 지불하는 값은 세 항의 합이다.

$$
C_E(q) \;=\; \underbrace{C_{\text{search}}}_{\text{① 검색 자체}} \;+\; \underbrace{C_{\text{prefill}}}_{\text{② prefill 증가}} \;+\; \underbrace{\frac{C_{\text{write}}}{N_q}}_{\text{③ 쓰기}}
\tag{10-5}
$$

식 (10-5)가 말하는 것은 세 항이 서로 다른 자원을 서로 다른 주기로 먹는다는 것이다. 셋을 하나씩 본다.

**① 검색 자체.** §10.4가 유도한 항이다. host DRAM 위의 벡터 연산이고, bandwidth-bound이며, 크기는 index 크기와 ANN 파라미터가 정한다. 주기는 **질의마다**다. 이 항은 GPU를 쓰지 않지만 TTFT의 앞쪽에 직렬로 들어간다 — 질의를 embedding하고, host에서 그래프를 걷고, $K_{\text{ret}}$개 레코드를 다시 가속기로 보낸 뒤에야 prefill이 시작된다. 이 왕복이 프롬프트 길이가 아니라 index 크기에 따라 늘어난다는 것이 $E$층이 서빙 경로에 넣는 새 항의 성질이다.

**② prefill 증가.** 이 장이 소유하는 정의다.

> **정의.** **prefill 증가**는 $\mathrm{ret}(E,q)$의 결과를 프롬프트에 실어 보냄으로써 생기는 추가 prefill 비용이다. 크기는 회수된 토큰 질량 $K_{\text{ret}}\cdot C$가 정하고, 발생 주기는 **질의마다**이며, 어떤 상각으로도 줄지 않는다.

RAG에서 이 항의 형태는 노골적이다. 입력과 회수된 레코드를 결합하는 방법이 "we simply concatenate them"이고 [RAG §2.3], 회수 폭은 open-domain QA 테스트에서 RAG-Token 15개, RAG-Sequence 50개, MS-MARCO와 Jeopardy에서 양쪽 10개다 [RAG App. A]. 즉 질의 하나가 최대 50개의 (질의 + 100-word 레코드) 컨텍스트를 generator에 통과시킨다.

decode 쪽 배수도 여기서 갈린다. RAG-Token은 주변화된 전이확률을 표준 beam decoder에 그대로 꽂으므로 decode 루프는 하나지만 매 스텝이 $K_{\text{ret}}$-원 혼합이고, 따라서 $K_{\text{ret}}$개의 KV cache를 동시에 들고 있어야 한다 [RAG §2.5]. RAG-Sequence의 Thorough Decoding은 레코드마다 beam search를 한 번씩 돌린 뒤, 그 레코드의 beam에 없는 가설마다 forward pass를 추가로 돌린다 — 논문 자신이 "For longer output sequences, |Y| can become large, requiring many forward passes"라고 쓴다 [RAG §2.5]. 생성 과제에서는 이 비용 때문에 Fast Decoding을 채택했고, 그 이유는 Thorough Decoding이 성능을 올리지 못했기 때문이다 [RAG App. A].

> **[해설]** 오늘날 실무의 형태는 조금 다르다. $K_{\text{ret}}$개 레코드를 각각 독립 컨텍스트로 돌리는 대신 하나의 프롬프트로 이어 붙인다. 그러면 컨텍스트 길이가 $K_{\text{ret}}\cdot C$만큼 늘고, prefill FLOPs는 그 길이에 비례하며(dense 모델에서 파라미터 $P$당 $\approx 2PT$), attention 항은 길이의 제곱으로 붙고, KV cache는 길이에 비례해 커진다. 두 형태는 비용의 배분이 다를 뿐 **질의마다 새로 지불한다**는 성질에서 같다.

> **[해설]** prompt cache 관점에서 한 번 더 본다. 회수 결과는 질의마다 달라지므로 프롬프트의 그 구간은 캐시 hit이 나지 않는다. 앞에 고정 system prompt를 두면 거기까지만 hit이고, 회수 블록이 시작되는 지점부터 끝까지는 전부 miss다. 즉 $E$-경로는 prefix cache를 늘려 주는 것이 아니라 **prefix cache가 덮는 구간의 경계를 앞으로 밀어낸다**. Rosetta 표의 "prefix cache ↔ $\hat c$" 행을 조심해서 읽어야 하는 이유가 이것이다 — 둘 다 재사용이 목적이지만, cache는 같은 계산의 재사용이고 회수 결과는 매번 다른 내용이다.

**③ 쓰기 비용.** RAG에서 $\mathrm{wr}$은 corpus 파이프라인이 하는 일이다: $n$개 chunk를 동결된 $\phi_d$로 한 번씩 embedding하고 FAISS index를 만든다 [RAG §3]. 문맥당 $B_s$는 0이다 — 저장 전에 어떤 추론도 하지 않기 때문이다(식 (10-2)). 그리고 이 비용은 corpus 스냅샷당 한 번 지불되고 배포 수명 전체에 걸쳐 나뉜다. 논문은 이 항의 시간도 FLOPs도 보고하지 않는다.

**세 항을 상각식에 넣으면 이 장의 결론이 나온다.** 식 (A)의 형태로 다시 쓰면

$$
C_{\text{avg}} \;=\; \underbrace{C_{\text{search}} + C_{\text{prefill}}}_{C_{\text{wake}}} \;+\; \underbrace{\frac{C_{\text{write}}}{N_q}}_{C_{\text{sleep}}/N_q},
\qquad
\lim_{N_q\to\infty} C_{\text{avg}} \;=\; C_{\text{search}} + C_{\text{prefill}} \;>\; 0
\tag{10-6}
$$

이 극한이 말하는 것은 하나다 — **상각은 ③만 지운다. ①과 ②는 $N_q$를 아무리 키워도 남는다.** 이것은 회계 방식의 선택이 아니라 주기의 정의다. ③은 문맥당 한 번 지불되므로 그 문맥을 공유하는 질의 수로 나뉘고, ②는 질의당 한 번 지불되므로 나눌 것이 없다.

RAG는 이 구조의 극단에 있다. ③이 corpus당 한 번이라 $N_q$가 사실상 배포 수명 전체의 질의 수이고, 그래서 손익분기 $N_q^\ast$를 묻는 것 자체가 공허하다. 상각 논증이 필요해지는 것은 ③이 **문맥당** 항으로 옮겨간 뒤다 — 그것이 $E$-경로의 이후 논문들이 한 일이다(→ ch13·ch14). 그때 ②는 그대로 남는다. 이 장은 여기까지만 확정하고 판정은 Part III로 넘긴다(→ ch25).

> **[해설]** 세 항이 batching에 거는 압력도 정리해 둔다. 좋은 소식부터 — $E$는 전역이고 $\Theta$는 fine-tuning 후 고정이므로, 배치 안의 모든 요청이 같은 가중치와 같은 index를 쓴다. shared-weight batching은 온전하다. 나쁜 소식은 요청 하나의 유효 시퀀스 수가 1이 아니라 $K_{\text{ret}}$이라는 것이다. continuous-batching 스케줄러가 보는 요청의 토큰 비용을 사용자의 프롬프트가 아니라 **검색 하이퍼파라미터**가 정하고, 운영자가 $K_{\text{ret}}$ 다이얼을 돌리면 재학습 없이 SLA와 정확도를 맞바꿀 수 있다 [RAG §4.5]. 이 다이얼의 품질 수익은 단조가 아니다 — RAG-Sequence는 $K_{\text{ret}}$을 50까지 올릴수록 좋아지지만 RAG-Token은 10에서 정점을 찍는다 [RAG Figure 3].

> **[평가]** RAG는 세 항 중 어느 것도 측정하지 않는다. 지연·처리량·FLOPs·index 빌드 시간이 논문 어디에도 없고, 부록이 보고하는 비용은 메모리 footprint뿐이다 [RAG App. C]. 회수 폭을 바꾸면 "performance and runtime"이 영향을 받는다고 명시해 놓고도 [RAG §4.5], Figure 3의 세 패널은 전부 $K_{\text{ret}}$을 품질 축에만 대응시킨다 [RAG Figure 3]. 서빙 비용을 정하는 유일한 knob을 세 번 스윕하면서 비용 축을 한 번도 그리지 않은 것이다. $E$-경로의 이후 비용 주장은 전부 이 침묵 위에 세워진다.

## 10.6 학습된 검색기가 항상 이기지는 않는다

비용을 지불할 가치가 있는지는 검색기가 실제로 이기는지에 달려 있고, 이 논문의 표에는 지지 않는 반례가 들어 있다.

**FEVER에서 BM25가 학습된 검색기를 이긴다.** dev 세트 ablation에서 BM25로 갈아 끼운 RAG-Token-BM25가 FEVER-3-way 75.1, FEVER-2-way 91.6을 기록하고, 학습된 검색기를 쓰는 RAG-Token은 74.5와 90.6에 그친다 [RAG Table 6]. 논문도 인정한다 — "For FEVER, BM25 performs best, perhaps since FEVER claims are heavily entity-centric and thus well-suited for word overlap-based retrieval" [RAG §4.5]. 파라미터가 하나도 없는 어휘 겹침 검색기가, 이 논문의 중심 기여인 학습된 검색기를 한 과제군에서 이긴다.

**"모든 과제에서 개선된다"는 문장도 자기 표에서 정확히 성립하지 않는다.** 논문은 동결 검색기 ablation을 두고 "learned retrieval improves results for all tasks"라고 쓰지만 [RAG §4.5], 같은 표의 MS-MARCO Bleu-1에서 RAG-Token-Frozen과 RAG-Token은 둘 다 49.4로 **동점**이다 [RAG Table 6]. 개선이 아니라 무승부다.

**한 과제군에서는 기제가 통째로 무너진다.** story generation에서는 검색기가 붕괴해 입력과 무관하게 같은 레코드를 뽑고, 그러면 generator가 레코드를 무시하는 법을 배워서 "the RAG model would perform equivalently to BART"가 된다 [RAG App. H]. 논문은 이것을 부록에 수치 없이, 데이터셋 상세 없이, 수정 시도 없이 적는다. 이 실패 양식에는 이름이 붙어 있다 — **retrieval collapse**이며, $E$-경로가 처음 보고한 읽기 쪽 실패다.

반대 방향의 사실도 함께 기록해 둔다. 회수 품질은 절대적이지 않다 — FEVER gold evidence 기준으로 1위 레코드가 정답 문서인 비율이 71%, 상위 10개 안에 있는 비율이 90%다 [RAG §4.4]. 그런데 NQ에서는 정답이 회수된 어떤 레코드에도 없을 때조차 11.8%를 맞힌다 [RAG §4.1]. 즉 $\Theta$와 $E$는 서로를 부분적으로 대체한다.

> **[평가]** 세 사실을 §10.5의 비용 축에 얹으면 다음이 따라 나온다. 항 ①과 ②는 검색기의 품질과 무관하게 **항상 지불된다**. FEVER에서는 그 비용을 내고 산 학습된 검색기가 무료 baseline에 지고, story generation에서는 지불한 비용 전부가 버려진다. "검색을 붙이면 좋아진다"는 명제는 과제 의존적이고, 비용은 과제 의존적이지 않다. 이 비대칭이 $E$-경로 평가의 기본 구도이며, ch15에서 품질 1위가 반복적으로 "기억 시스템 없음"으로 나오는 현상의 조상이 여기에 있다.

## (state, update, cost) 정리

이 장의 개념을 세 층 프레임에 배치하면 RAG가 어느 칸을 채우고 어느 칸을 비웠는지가 한눈에 보인다.

표 10-1 — $E$층 개념의 (state, update, cost) 배치. 수치는 전부 [RAG] 보고값이고 합산·차감만 이 책의 산술이며, 빈칸은 "논문에 없음"이다.

| 대상 | 층 | 갱신식 | 시간척도 | 비용 |
|---|---|---|---|---|
| chunk embedding index (21M 레코드, $C=100$ words) | $E$ | (U-E). 단 $\hat c_k = c_k$ 항등, $B_s = 0$ | corpus 스냅샷마다 (실증은 2016→2018 2회) | $C_{\text{cap}}$ = 36 GB(압축) / 약 100 GB(무압축), **corpus당** [RAG App. C] |
| generator + 질의 encoder $\phi_q$ | $\Theta$ | (U-$\Theta$)의 **모양만**. $\mathcal{R}$이 사람이 라벨한 고정 corpus라 sleep 라운드가 아니다 | 배포 전 1회 | 논문 표기 "trainable 626M" [RAG App. G]. 단 그 합에는 같은 문장이 학습하지 않는다고 밝힌 $\phi_d$ 110M이 들어 있어, 실제로 움직이는 것은 generator 406M + $\phi_q$ 110M이다 |
| 저장 encoder $\phi_d$ | $\Theta$ (동결) | 없음 | — | 110M [RAG App. G]. 바꾸면 $n$개 벡터 전량 재계산 [RAG §2.4] |
| fast weights | $W$ | 없음 ($W=\varnothing$) | — | 0 |
| $\mathrm{ret}(E,q)$ | 읽기 (R) | — | 질의마다 | ① host DRAM 대역폭 + ② prefill $K_{\text{ret}}\cdot C$ 토큰. 둘 다 **논문 미측정** |
| index 교체 | $E$ | $\mathrm{wr}$을 사람·파이프라인이 실행 | 세계가 바뀔 때 | 신선도 실패 시 68% → 4% [RAG §4.5] |

표에서 읽어야 할 것은 두 가지다. 첫째, $B_s = 0$ 칸이 RAG를 sleep-time compute 판별식에서 탈락시킨다 — 질의 전에 예산을 쓰는 오프라인 추론이 없다(→ ch11). 둘째, 비용 열에서 실제로 채워진 칸은 $C_{\text{cap}}$ 하나뿐이고 그것도 단위가 corpus당이다. $B_s$·$L_w$·$\rho$는 이 논문에 없다.

## Worked micro-example — 질의 하나가 지불하는 세 항

식 (10-5)의 세 항을 실제 숫자로 채워, ②가 ③을 몇 자릿수로 지배하는지 암산으로 확인한다. **아래 계산은 전부 이 책의 예시 계산이며 논문의 보고값이 아니다.** 논문에서 오는 값과 이 예제가 고르는 값을 먼저 갈라 둔다.

**논문에서 오는 값**: $n = 21\mathrm{M}$ 레코드, $C = 100$ words, $K_{\text{ret}} = 50$(RAG-Sequence, open-domain QA 테스트), index 값 개수 $n d_e = 1.53\times10^{10}$, 저장 encoder $\phi_d$ = 110M, 보고된 CPU 메모리 무압축 약 100 GB / 압축 후 36 GB [RAG §3, App. A, App. C, App. G].
**이 예제가 고르는 값**(논문에 없다): 벡터를 fp32로 저장($b = 4$ bytes/elem), 1 word ≈ 1.3 tokens, generator는 8B dense 모델이라 prefill FLOPs $\approx 2PT$, 가속기 실효 처리량 125 TFLOP/s(peak 312 TFLOP/s × MFU 40%), host DRAM 대역폭 100 GB/s, 질의 자체는 15 words.

전제를 하나로 고정한다. 트래픽 항은 **벡터 배열만** 세고, 그 크기는 위 두 값에서 직접 나온다: $n d_e b = 1.53\times10^{10}\times4 = 61$ GB. 논문이 보고하는 약 100 GB는 §10.4에서 밝힌 대로 HNSW 그래프와 원문까지 포함한 index 전체의 상주 메모리이므로 이 예제의 트래픽 항과 회계 대상이 다르다 [RAG §3, App. C]. 아래 3단계는 61 GB 쪽으로 계산하고, 100 GB를 쓰면 같은 값이 1.6배 커진다 — 결론이 되는 자릿수는 바뀌지 않는다.

**1단계 — 프롬프트 질량.** 회수 질량 $= K_{\text{ret}}\cdot C = 50\times100 = 5{,}000$ words $\approx 6{,}500$ tokens. 질의 자체는 $15\times1.3\approx20$ tokens. 프롬프트가 20 → 6,520 tokens, 즉 **326배**가 된다.

**2단계 — 항 ②, prefill.** $2PT = 2\times8\!\times\!10^9\times6{,}520 \approx 1.04\times10^{14}$ FLOP = **104 TFLOP**. 검색 없이 질의만 넣으면 $2\times8\!\times\!10^9\times20 = 3.2\times10^{11}$ = 0.32 TFLOP. 시간으로는 $104/125 \approx 0.84$ s 대 2.6 ms다. TTFT가 밀리초에서 초로 넘어간다.

**3단계 — 항 ①, 검색.** 전수 MIPS의 계산량은 $2 n d_e = 2\times1.53\!\times\!10^{10} = 3.06\times10^{10}$ = 30.6 GFLOP뿐이다. 같은 가속기면 0.24 ms — 계산량으로는 무시할 수준이다. 그런데 같은 연산이 벡터 배열 61 GB를 훑으므로 100 GB/s DRAM에서 **0.61 s**가 걸린다. 계산 시간의 2,500배다. 식 (10-4)가 말한 것이 숫자로 확인된다: 검색은 FLOPs가 아니라 대역폭 문제이고, ANN은 바로 이 0.6초를 깎으려고 존재한다. 논문은 그 결과를 "sufficiently fast on CPU"라고만 쓰고 값을 주지 않으므로 [RAG App. C], 여기서 멈춘다.

**4단계 — 항 ③, 쓰기.** index를 만들려면 저장된 전체 텍스트를 $\phi_d$에 한 번 통과시켜야 한다. 레코드당 $C=100$ words $\approx 130$ tokens이므로 총량은 $21\!\times\!10^6\times130 = 2.73\times10^9$ tokens이고, $2\times1.1\!\times\!10^8\times2.73\!\times\!10^9 \approx 6.0\times10^{17}$ FLOP = **600 PFLOP**. 같은 가속기 한 장이면 $6.0\!\times\!10^{17}/1.25\!\times\!10^{14} = 4{,}800$ s ≈ 1.3시간이다(FAISS/HNSW 그래프 빌드는 여기 포함되지 않았고 논문도 보고하지 않는다).

**5단계 — 나눗셈 하나.** ③이 질의당 prefill 아래로 내려가는 지점은

$$
N_q^\ast \;=\; \frac{C_{\text{write}}}{C_{\text{prefill}}} \;=\; \frac{6.0\times10^{17}}{1.04\times10^{14}} \;\approx\; 5.8\times10^{3}
\tag{10-7}
$$

질의 약 5,800개다. 배포 수명 동안 $N_q = 10^9$개를 처리하면 ③은 질의당 $6.0\times10^{17}/10^9 = 0.6$ GFLOP로 줄어 ②의 $1.7\times10^{5}$분의 1이 된다.

**읽어야 할 결론.** ③은 다섯 자릿수 아래로 사라지고 ②는 104 TFLOP에 그대로 남는다. $N_q$를 더 키워도 이 그림은 바뀌지 않는다 — 식 (10-6)의 극한이 그것이다. 반대로 ②를 줄이는 방법은 $K_{\text{ret}}$이나 $C$를 줄이는 것뿐이고, 둘 다 품질 knob이다(§10.3, §10.5).

> **[평가]** 위 절대치 중 논문에서 오는 것은 $n$·$C$·$K_{\text{ret}}$·$n d_e$·footprint·$\phi_d$ 크기뿐이고 나머지는 이 예제가 고른 값이다. 따라서 이 계산이 단정하는 것은 **비율과 순서**다 — ②가 ③을 다섯 자릿수 차이로 지배하고, ②만 $N_q$에 무감하다. 절대치는 각자의 스택 값으로 갈아 끼워야 하는 하한·방향성이다. 이 책은 자체 계산에서 비율·crossover·순서만 단정문으로 쓰고 절대치는 그렇게 쓰지 않는다.

## 요약

- $E$의 구현 형태는 text·vector·graph 세 가지이고, 형태를 가르는 것은 저장 방식이 아니라 $\mathrm{ret}(E,q)$가 무엇을 계산하는가다. 셋 다 GEMM이 아니고 셋 다 arithmetic intensity가 낮다.
- RAG는 식 (R)에서 $W=\varnothing$인 형태이며, 읽기 함수 $\mathrm{ret}(E,q)$를 학습 가능한 일급 객체로 만든 논문이다. 반면 (U-E)는 퇴화해 있다 — $\hat c$가 항등사상이고 문맥당 $B_s = 0$이며 $\mathrm{wr}$을 실행하는 주체가 모델이 아니다.
- chunk 크기 $C$는 레코드 수 $n$·프롬프트 질량 $K_{\text{ret}}\cdot C$·레코드의 자족성을 서로 반대 방향으로 움직이는 semantic knob이다. RAG는 $C=100$ words를 관행값으로 고정하고 스윕하지 않는다.
- 읽기 함수의 두 반쪽은 가격이 다르다. 질의 encoder $\phi_q$는 공짜로 바꿀 수 있지만 저장 encoder $\phi_d$를 바꾸면 $n$개 벡터를 전량 재계산해야 하고, 그래서 RAG는 $\phi_d$를 동결한다 [RAG §2.4].
- $\mathrm{ret}(E,q)$의 비용은 세 항이다: ① 검색 자체(host DRAM 대역폭), ② prefill 증가($K_{\text{ret}}\cdot C$ 토큰), ③ 쓰기. 상각식 (A)의 분모를 통과하는 것은 ③뿐이고, ①과 ②는 $N_q\to\infty$에서도 남는다.
- prefill 증가는 질의마다 발생하며 상각되지 않는다. 회수 결과가 질의마다 다르므로 prompt cache로도 회수되지 않는다.
- 학습된 검색기가 항상 이기지는 않는다. FEVER에서 BM25가 이기고(75.1/91.6 대 74.5/90.6), 논문도 "For FEVER, BM25 performs best"라고 인정하며 [RAG Table 6, §4.5], story generation에서는 retrieval collapse로 기제 자체가 무너진다 [RAG App. H].
- RAG는 지연·처리량·FLOPs·index 빌드 시간을 하나도 보고하지 않는다. 비용 축이 비어 있는 채로 $E$-경로가 시작되었고, 이후 논문들의 비용 주장은 그 침묵을 물려받았다.

## 자가 점검 체크리스트

- [ ] $E$의 세 구현 형태를 각각 한 문장으로 정의하고, 각 형태의 $\mathrm{ret}$이 계산하는 것을 말할 수 있다.
- [ ] RAG를 식 (R)과 (U-E)로 각각 환원해 쓸 수 있고, 어느 자리가 왜 비어 있는지($W=\varnothing$, $\hat c$ 항등, $B_s=0$) 설명할 수 있다.
- [ ] $C$(chunk 크기)와 $C_{\text{cap}}$(용량)을 혼동하지 않고, $K_{\text{ret}}$을 sleep 라운드 첨자 $k$와 구분해 쓸 수 있다.
- [ ] 식 (10-5)의 세 항을 자기 서빙 config 숫자로 채우고, (10-6)에서 $N_q$로 나뉘는 항이 하나뿐인 이유를 주기의 정의로 설명할 수 있다.
- [ ] $\phi_q$와 $\phi_d$의 갱신 비용이 왜 비대칭인지 말할 수 있고, 그것이 $E$층 설계에 거는 제약을 하나 들 수 있다.
- [ ] FEVER에서 BM25가 이긴다는 사실을 수치와 함께 인용할 수 있고, 그 사실이 비용 논증에 무엇을 더하는지 말할 수 있다.
- [ ] Rosetta: "vector DB 조회 ↔ $\mathrm{ret}(E,q)$" 행이 왜 완전한 대응이고, "prefix cache ↔ $\hat c$" 행이 왜 부분적 대응인지(회수 결과는 매 질의 달라진다) 각각 한 문장으로 설명할 수 있다.

## 다음 장으로

이 장은 $E$층의 구현 형태를 셋으로 가르고 그 위의 읽기 비용을 세 항으로 갈랐다. 그러나 정작 RAG는 이 책의 주제가 아니다 — 문맥당 $B_s = 0$이고 저장 전에 어떤 추론도 하지 않으므로, "질의 전에 예산을 써서 어느 층의 상태를 바꾼다"는 조건을 만족하지 않는다. 이 장이 남기는 질문은 두 개다. 무엇이 $E$에 쓸 내용을 만드는가($S(c;B_s)$를 항등에서 떼어내면 무엇이 되는가), 그리고 그 판정을 무슨 기준으로 하는가. Part II의 첫 장인 ch11이 두 번째 질문을 먼저 처리한다 — 판별식을 명시하고 논문 대장 전체를 통과·불통과로 가른 뒤, 통과한 것들을 $E$·$W$·$\Theta$ 세 경로로 나눈다. 첫 번째 질문은 그 분류 위에서 ch12·ch13이 이어받는다.

이 장이 다루지 않은 것도 지목해 둔다. RAG의 평가설계에 붙는 나머지 불리한 사실들 — baseline 선택, 데이터셋 구성, 비교 조건의 대칭성 — 은 이 장이 아니라 ch13(RAG에서 MemGPT로)이 소유한다. 이 장은 $E$층의 **형태**와 **비용 분해**를 세우는 개념 장이므로, 평가설계 쟁점은 같은 논문을 다시 다루는 ch13에 모아 한 번에 처리한다.
