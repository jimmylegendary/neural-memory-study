# 참고문헌

## 수록 원칙

이 목록에 실리는 문헌은 두 종류이고, 두 종류를 구획으로 나눈다. **§90.1–§90.3은 corpus 29편**이다.
원문을 확보해 전문을 대조한 문헌이며, 본문의 사실 서술과 수치 인용은 전부 여기서만 나온다.
**§90.5는 본문이 이름으로만 지목하는 문헌**이다. 계보를 설명하거나 어떤 논문이 무엇을 인용하지
않았는지를 지적할 때 이름이 필요해 등장할 뿐, 이 책은 이들을 근거로 쓰지 않는다. 두 구획을 섞지
않는 것이 이 책의 정직성 규칙이다 — 대조하지 않은 문헌에서 사실을 옮기지 않는다.

각 항목은 다음 순서로 적는다.

> 저자 · *제목* · **arXiv ID 또는 venue** · 연도 · **이 책의 담당 장**

규약 넷을 못 박는다.

- **저자가 다섯 명 이상이면 제1저자(또는 공동 제1저자)만 적고 "외"를 붙인다.** 전원 명단은 원문
  표제부에 있다.
- **venue는 원문에 인쇄된 것만 적는다.** 원문에 venue 표기가 없으면 "venue 미기재"로 적는다.
  다른 경로로 알려진 게재 사실이 있더라도 그 원문에 인쇄되어 있지 않으면 여기에 적지 않는다 —
  EWC가 대표 사례다(§90.3).
- **이 책이 읽은 판(version)을 병기한다.** arXiv 문헌은 판마다 표·수치가 달라지므로, 본문의 인용은
  전부 여기 적힌 판을 가리킨다.
- **담당 장은 그 문헌을 정면으로 다루는 장이다.** 괄호 안은 그 문헌을 근거로 쓰는 다른 장이다.
  전수 분류의 근거와 판정은 부록 A와 ch11에 있다.

각 장이 본문에서 쓰는 인용 표지(`[SEAL §4.2]` 형식)는 그 장이 첫 등장 시 고정한다. 표지가 장마다
다를 수 있으므로, 이 목록은 표지가 아니라 ch11의 전수 분류표가 쓰는 짧은 이름으로 배열한다.

---

## 90.1 corpus — A군: 판별식 통과 (9편) + 경계 사례 (1편)

판별식 네 조건을 모두 만족하는 문헌 9편과, 조건 (1)만 만족하지 않아 통과군에 세지 않는 경계 사례
1편(MemGPT)이다. 조건별 판정은 부록 A 표 A-1에 있다.

**Letta `Sleep-time Compute`** — Kevin Lin, Charlie Snell 외 (Letta; UC Berkeley).
*Sleep-time Compute: Beyond Inference Scaling at Test-time.*
**arXiv:2504.13171**, 2025-04-17. venue 미기재.
→ **ch14** (ch01, ch11, ch24, ch25, ch27, ch28)

**MemGPT** — Charles Packer 외 (UC Berkeley).
*MemGPT: Towards LLMs as Operating Systems.*
**arXiv:2310.08560**, v1 2023-10-12 / 이 책이 읽은 판 v2 2024-02-12. venue 미기재.
→ **ch13** (ch11, ch24)

**Mem0** — Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, Deshraj Yadav.
*Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory.*
**arXiv:2504.19413**, 2025-04-28. venue 없음.
→ **ch15** (ch11, ch26, ch27)

**Zep** — Preston Rasmussen 외.
*Zep: A Temporal Knowledge Graph Architecture for Agent Memory.*
**arXiv:2501.13956**, 2025-01-20. venue 없음.
→ **ch15** (ch11, ch27)

**ReasoningBank** — Siru Ouyang 외 (UIUC; Google Cloud AI Research; Yale).
*ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory.*
**arXiv:2509.25140**, v1 2025-09 / 이 책이 읽은 판 v2 2026-03-16. venue 없음.
→ **ch15** (ch05, ch11, ch27)

**SCM** — Saish Shinde (Clyrai IP Studio), 단독 저자.
*SCM: Sleep-Consolidated Memory with Algorithmic Forgetting for Large Language Models.*
**arXiv:2604.20943v1**, 2026-04-22. venue 없음 — 저자 스스로 "a research preview"라고 적는다.
→ **ch15** (ch11 §11.4.1, ch27)

**Multi-Timescale (Memini)** — Andreas Pattichis, Constantine Dovrolis.
*Continual Knowledge Updating in LLM Systems: Learning Through Multi-Timescale Memory Dynamics.*
**arXiv:2605.05097**, v3 2026-06-24. **ICML 2026 Workshop "Continual Adaptation at Scale:
Towards Sustainable AI"** — 부록 포함 9쪽의 workshop 논문이다.
→ **ch15** (ch11, ch27)

**`Do Language Models Need Sleep?`** — Sangyun Lee, Sean McLeish, Tom Goldstein, Giulia Fanti
(Carnegie Mellon University; University of Maryland).
*Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference.*
**arXiv:2605.26099**, v3 2026-06-05. 원문이 스스로 "Preprint."라고 표기하며 venue는 없다.
→ **ch17** (ch11, ch24, ch26, ch29)

**SEAL** — Adam Zweiger, Jyothish Pari 외 (Improbable AI Lab, CSAIL MIT).
*Self-Adapting Language Models.*
**arXiv:2506.10943**, 이 책이 읽은 판 v2 2025-09-18. **NeurIPS 2025** (원문 1쪽 각주).
→ **ch20** (ch05, ch06, ch11, ch19, ch27)

**`Language Models Need Sleep`** — Ali Behrouz, Farnoosh Hashemi, Adel Javanmard, Vahab Mirrokni
(Google; Cornell).
*Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories.*
**arXiv:2606.03979**, v2 2026-07-10. venue 없음 — 원문 각주는 2025년 9월부터 OpenReview에
공개되어 있었다고만 적는다.
→ **ch21** (ch04, ch11, ch18, ch24, ch27)

---

## 90.2 corpus — B군: 판별식 불충족 (9편)

이 이름 아래에서 자주 함께 거론되지만 네 조건 중 하나 이상에서 탈락하는 문헌이다. 탈락한다는 것은
이 범주가 아니라는 뜻이지 쓸모없다는 뜻이 아니며, 각 문헌이 실제로 증거가 되는 자리를 담당 장이
지목한다.

**Nested Learning** — Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni
(Google; Columbia).
*Nested Learning: The Illusion of Deep Learning Architecture.*
**arXiv:2512.24695v1**, 2025-12-31. **NeurIPS 2025** — 노트는 "A version published at
NeurIPS 2025"로 기록한다.
→ **ch18** (ch11, ch21, ch24, ch28)

**Memory Caching** — Ali Behrouz 외 (Google).
*Memory Caching: RNNs with Growing Memory.*
**arXiv:2602.24281v1**, 2026-02-27. venue 미기재.
→ **ch18** (ch11, ch26, ch29)

**Memory Layers** — Vincent-Pierre Berges, Barlas Oğuz 외 (Meta FAIR).
*Memory Layers at Scale.*
**arXiv:2412.09764v2**, 2024-12-20(arXiv 스탬프) / 2024-12-23(본문 Date 줄). venue 미기재.
→ **ch09 · ch19** (ch11, ch26, ch29)

**Generative Adapter** — Tong Chen 외 (University of Washington; Microsoft Research).
*GenerativeAdapter: Contextualizing Language Models in Parameters with a Single Forward Pass.*
**arXiv:2411.05877**, 2024-11-08. venue 미기재 — 원고는 conference 양식이나 학회명을 적지 않는다.
→ **ch16** (ch11, ch24, ch26)

**LoRA** — Edward Hu, Yelong Shen 외 (Microsoft).
*LoRA: Low-Rank Adaptation of Large Language Models.*
**arXiv:2106.09685v2**, 2021-10-16. venue 미기재.
→ **ch03 · ch19** (ch09, ch11, ch20, ch21, ch26)

**ROME** — Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov (MIT CSAIL; Northeastern;
Technion – IIT).
*Locating and Editing Factual Associations in GPT.*
**arXiv:2202.05262**, 이 책이 읽은 판 v5 2023-01-13. **NeurIPS 2022**.
→ **ch19** (ch11, ch22)

**MEMIT** — Kevin Meng, Arnab Sen Sharma 외 (MIT CSAIL; Northeastern; Technion – IIT).
*Mass-Editing Memory in a Transformer.*
**arXiv:2210.07229v2**, 2023-08-01. **ICLR 2023** (전 페이지 running header).
→ **ch19** (ch11, ch22)

**RAG** — Patrick Lewis 외 (Facebook AI Research; UCL; NYU).
*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*
**arXiv:2005.11401**, 이 책이 읽은 판 v4 2021-04-12. venue 미기재 — vendored 원문 1,450줄
어디에도 게재 표기가 없다.
→ **ch10 · ch13** (ch11, ch25, ch26)

**Memorizing Transformers** — Yuhuai Wu, Markus N. Rabe, DeLesley Hutchins, Christian Szegedy.
*Memorizing Transformers.*
**arXiv:2203.08913**, 2022-03-16. **ICLR 2022** (running header).
→ **ch12** (ch11, ch26)

---

## 90.3 corpus — C군: 판별식의 대상이 아닌 배경·제약 문헌 (10편)

판별식이 판정할 대상이 아니라 다른 장들이 근거로 쓰는 재료다. 지속학습의 원형 셋, 제약·이론 넷,
생물 실증 둘, 그리고 $\Theta$-경로의 반증 후보 하나로 이루어진다.

**EWC** — James Kirkpatrick 외 (DeepMind).
*Overcoming catastrophic forgetting in neural networks.*
**arXiv:1612.00796**, 이 책이 읽은 판 v2 2017-01-25. **venue 미기재** — vendored 원문은 arXiv
preprint 판이고 게재 표기·DOI가 없다. 이 문헌에 venue를 붙여 인용하는 것은 이 책에서 결함이다.
→ **ch07 · ch22** (ch08, ch18)

**Deep Generative Replay (DGR)** — Hanul Shin, Jung Kwon Lee, Jaehong Kim, Jiwon Kim
(MIT; SK T-Brain).
*Continual Learning with Deep Generative Replay.*
**arXiv:1705.08690**, 이 책이 읽은 판 v3 2017-12-12. **NIPS 2017** (원문 1쪽 각주에 인쇄됨).
→ **ch07 · ch22** (ch06)

**GEM** — David Lopez-Paz, Marc'Aurelio Ranzato (Facebook AI Research).
*Gradient Episodic Memory for Continual Learning.*
**arXiv:1706.08840**, 이 책이 읽은 판 v6 2022-09-13. **NIPS 2017** (원문 1쪽 각주에 인쇄됨).
→ **ch07 · ch22** (ch18, ch25, ch30)

**model collapse** — Matthias Gerstgrasser, Rylan Schaeffer, Apratim Dey, Rafael Rafailov 외
(Harvard; Stanford).
*Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and
Synthetic Data.*
**arXiv:2404.01413v2**, 2024-04-29. venue 미기재.
→ **ch06 · ch22** (ch20, ch28, ch30)

**How much do LMs memorize** — John X. Morris 외 (FAIR at Meta; Cornell).
*How much do language models memorize?*
**arXiv:2505.24832**, 이 책이 읽은 판 v3 2025-06-18. venue 미기재.
*원문의 날짜 표기가 두 개이고 하루 어긋난다 — 본문 Date 줄은 2025-06-19, arXiv 스탬프는
2025-06-18이다. 이 책은 두 값을 인쇄된 대로 병기한다.*
→ **ch09 · ch22** (ch26, ch29)

**LoRA as Knowledge Memory** — Seungju Back, Dongwoo Lee 외 (KAIST; Samsung SDS).
*Understanding LoRA as Knowledge Memory: An Empirical Analysis.*
**arXiv:2603.01097v5**, 2026-07-29. **ICML 2026, PMLR 306** (원문 1쪽 각주).
→ **ch09 · ch19** (ch26)

**Rate–Distortion Memory Compaction** — Ashwin Gerard Colaco, Nada Lahjouji
(University of California, Irvine).
*What to Keep, What to Forget: A Rate–Distortion View of Memory Compaction in LLMs and Agents.*
**arXiv:2607.08032v1**, 2026-07-09. venue 미기재 — ACM 양식으로 조판되어 있으나 학회명이 없다.
→ **ch22** (ch11 §11.7, ch26, ch27)

**Can a LM Learn Facts Continually** — Charles O'Neill (Baseten), 단독 저자.
*Can a Language Model Learn Facts Continually in Its Weights?*
**arXiv:2607.11020v2**, 2026-07-14. venue 없음 — 표제는 "Preprint, July 2026"이다.
→ **ch22** (ch27, ch30)

**PLOS Sleep** — Ryan Golden, Jean Erik Delanois, Pavel Sanda 외 (UCSD; Institute of Computer
Science of the Czech Academy of Sciences).
*Sleep prevents catastrophic forgetting in spiking neural networks by forming a joint synaptic
weight representation.*
**PLoS Computational Biology 18(11): e1010628**, 2022-11-18. arXiv preprint 없음.
→ **ch08 · ch23** (ch07)

**PAD** — Nicolas Deperrois, Mihai A. Petrovici, Walter Senn 외 (University of Bern; Heidelberg
University).
*Learning cortical representations through perturbed and adversarial dreaming.*
**eLife 2022;e76384**, DOI 10.7554/eLife.76384, 2022-04-06 게재. arXiv ID 없음(2021-09-09 preprint).
→ **ch08 · ch23** (ch06)

<!-- TODO-VERIFY: PAD의 eLife 권(volume) 번호. 이 책이 옮겨 적은 인용 블록은 "eLife 2022;0:e76384"이고
     권 자리가 0으로 인쇄되어 있다. 확인 방법: PAD 원문 papers/stc/STC-T13.txt 에서 "eLife 2022;" 문자열을
     검색해 각 면주에 인쇄된 권 번호를 확인한다. 확인 전에는 권 번호를 인쇄하지 않는다. -->

---

## 90.4 이 책이 근거로 삼는 이 책 자신의 산출물

corpus의 어느 논문도 상태 용량을 바이트로 보고하지 않고, 비용 보고의 단위가 논문마다 다르다.
그래서 이 책은 비교에 필요한 두 회계를 스스로 만들었다. 이들은 외부 문헌이 아니므로 서지 항목이
아니라 **책 안의 위치**로 인용한다.

표 90-1 — 이 책이 스스로 만든 회계·감사 산출물과 그것을 싣는 장

| 산출물 | 무엇을 만드는가 | 위치 |
|---|---|---|
| 통일 비용 회계 | 논문별로 단위가 다른 비용 보고를 하나의 축에 재배치하고, 상각 논증의 유효 범위를 정한다 | ch25 |
| 상태 용량 회계 | 세 경로의 상태량을 바이트로 환산해 자릿수를 비교하고, dtype이 정하는 정보 밀도를 고정한다 | ch26 |
| $B_s$ 축 재배치 | 보고된 $B_s$–품질 점들을 통일 축에 놓고, 대다수 논문이 $B_s$를 보고하지 않는다는 사실을 정량화한다 | ch28 |
| 반복 consolidation 열화 | 자기생성 $\mathcal{R}$로 라운드를 반복할 때의 $\rho$를 소형 재현으로 측정한다 | ch29, ch30 |

이 네 산출물의 수치는 **등급 규칙**을 달고 인용된다 — 비율·crossover·순서·bound 분류는 본문
단정문으로 쓰고, 절대치는 하한 또는 방향성으로만 쓴다. 등급 규칙 자체는 Neural Memory 모노그래프의
정직성 계약을 승계한 것이다.

**Neural Memory 모노그래프** — 이 책의 전작. $W$층 하나를 master equation (M) 위에서 해부한다.
이 책의 식 (U-W)는 (M)과 글자 그대로 같고, $\Theta$·$W$·$S_t$·$\ell$·$\mathcal{L}$·$\eta_t$·
$\alpha_t$·$\beta_t$·$\mathcal{M}(\cdot;W)$·$C$(chunk)의 의미는 전부 거기서 승계한다.
본문 참조 형식은 "→ NM ch12 §12.3"이다.

---

## 90.5 본문이 이름으로만 지목하는 문헌

아래 문헌은 원문 대조를 거치지 않았다. 본문에 등장하는 이유는 하나뿐이다 — **어떤 논문이 누구를
읽었고 누구를 읽지 않았는지**를 적으려면 읽히지 않은 쪽의 이름이 필요하기 때문이다. 이 문헌들에서
사실이나 수치를 옮겨 오지 않는다.

**$W$-경로의 주류 계보** — ch11 §11.7이 `Do Language Models Need Sleep?`의 침묵을 지적하며
나열하는 네 편이다. 네 편 모두 이 책의 corpus가 아니다.

표 90-2 — 이름으로만 지목하는 $W$-경로 주류 계보 네 편

| 문헌 | 식별자 | 연도 | 본문에서의 자리 |
|---|---|---|---|
| Titans: Learning to Memorize at Test Time | arXiv:2501.00663 | 2024-12-31 | ch11 §11.7, ch17, ch18 |
| It's All Connected (Miras) | arXiv:2504.13173 | 2025-04-17 | ch11 §11.7, ch18 |
| Atlas: Learning to Optimally Memorize the Context at Test Time | arXiv:2505.23735 | 2025-05-29 | ch11 §11.7, ch18 |
| TNT: Improving Chunkwise Training for Test-Time Memorization | arXiv:2511.07343 | 2025-11-10 | ch11 §11.7 |

이 네 편은 Neural Memory 모노그래프가 정면으로 다룬 대상이다. 이 책이 이들을 다시 읽지 않는 것은
분업이며, $W$-경로의 기제 상세가 필요하면 그쪽으로 넘긴다(→ NM).

**CLS 이론의 원전** — McClelland, McNaughton, O'Reilly. *Why there are complementary learning
systems in the hippocampus and neocortex.* **Psychological Review 102(3): 419–457**, 1995.
ch08이 CLS를 도입할 때, 그리고 ch11 §11.6이 "이름은 생물에서 왔고 논증은 오지 않았다"를 적을 때
등장한다 — 이 라인에 sleep이라는 이름을 준 논문의 참고문헌에 이 원전이 없다는 사실이 그 서술의
근거다.

**agent memory 갈래** — ReasoningBank이 자기 목소리로 진단하는 선행 연구로 ch15에 이름이 나온다
(AWM, Synapse). 이 책은 이들을 읽지 않았고, ReasoningBank의 진단을 그 논문에 귀속시켜 옮기기만 한다.

<!-- TODO-VERIFY: AWM·Synapse의 arXiv ID와 연도. 확인 방법: papers/stc/2509.25140.txt 의 참고문헌
     목록(저자 성 알파벳순 구간)에서 "Agent workflow memory"와 "Synapse: Trajectory-as-exemplar"를
     검색해 원문이 인쇄한 식별자를 옮긴다. 원문이 arXiv ID 대신 venue만 인쇄했으면 §수록 원칙의
     "원문에 인쇄된 것만" 규약대로 venue를 적고 arXiv ID 자리는 비운다.
     확인 전에는 식별자를 인쇄하지 않는다. -->

**서지 항목이 아닌 이름들** — 본문에는 문헌이 아니라 도구·모델·데이터셋의 이름도 등장한다
(vLLM, NetworkX, GPT-4.1, Llama 3.1, PEER, BM25, DPR 등). 이들은 참고문헌이 아니라 그것을 쓴
논문의 실험 설정이므로, 출처는 언제나 그 논문의 위치로 병기한다. 별도 항목을 두지 않는다.

---

## 90.6 담당 장 역색인

문헌에서 장을 찾는 것이 §90.1–§90.3이고, 장에서 문헌을 찾는 것이 이 표다. 굵은 글씨가 그 장이
정면으로 다루는 문헌이고, 나머지는 근거로 쓰는 문헌이다.

표 90-3 — 장 → 그 장이 쓰는 문헌 역색인

| 장 | 문헌 |
|---|---|
| ch01 | Letta STC, `LM Need Sleep`, `Do LMs Need Sleep?` (세 층의 필요를 보이는 예로만) |
| ch02 | — (Neural Memory ch02 위에서 시작) |
| ch03 | **LoRA** |
| ch04 | `LM Need Sleep`, SEAL (distillation 실무의 실제 사용례로만) |
| ch05 | SEAL, ReasoningBank (RL 루프의 실제 사용례로만 — §05.7) |
| ch06 | **model collapse**, SEAL, DGR, PAD |
| ch07 | **EWC**, **DGR**, **GEM**, PLOS Sleep |
| ch08 | **PLOS Sleep**, **PAD**, EWC |
| ch09 | **How much do LMs memorize**, **Memory Layers**, **LoRA as Knowledge Memory** |
| ch10 | **RAG** |
| ch11 | corpus 29편 전부 |
| ch12 | **Memorizing Transformers** |
| ch13 | **MemGPT**, RAG |
| ch14 | **Letta STC** |
| ch15 | **Mem0**, **Zep**, **ReasoningBank**, **SCM**, **Multi-Timescale** |
| ch16 | **Generative Adapter** |
| ch17 | **`Do LMs Need Sleep?`** |
| ch18 | **Nested Learning**, **Memory Caching**, GEM |
| ch19 | **LoRA**, **ROME**, **MEMIT**, Memory Layers, LoRA as Knowledge Memory |
| ch20 | **SEAL** |
| ch21 | **`LM Need Sleep`**, Nested Learning |
| ch22 | **Can a LM Learn Facts Continually**, **Rate–Distortion**, EWC, DGR, GEM, How much do LMs memorize, ROME, MEMIT |
| ch23 | **PLOS Sleep**, **PAD** |
| ch24 | ch11의 분류 결과 — 통과 9편 + 경계 사례 1편 |
| ch25 | Letta STC, GEM, RAG + 이 책의 통일 비용 회계 |
| ch26 | Memory Layers, LoRA, How much do LMs memorize, Rate–Distortion + 이 책의 상태 용량 회계 |
| ch27 | 통과 9편 전부 + Can a LM Learn Facts Continually |
| ch28 | Letta STC, Nested Learning, model collapse |
| ch29 | `Do LMs Need Sleep?`, Memory Caching, Memory Layers |
| ch30 | model collapse, GEM, Can a LM Learn Facts Continually |
