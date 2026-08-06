# 목차

"sleep-time compute"이라는 이름 아래 모인 문헌을 하나의 판별식으로 걸러 내고, 남는 세 경로($E$·$W$·$\Theta$)를 하나의 비용 회계로 옮긴 뒤, 그 위에서 경로별로 판정한다. 척추 논지는 **이름이 기제보다 넓다**이며 정식 판정은 ch24와 ch27이 낸다. 페이지 번호는 빌드가 채운다.

## 앞부분 (front matter)

- **서문** — 이 책이 왜 있는가, Neural Memory 모노그래프와의 관계, 누구를 위한 책인가, 이 책의 방법 네 가지(판별식·통일 회계·정직성 규칙·자기 실험의 등급화), v1을 폐기한 이유, 읽는 순서.
- **목차** — 이 문서.
- **Part I 도입** — 자를 만드는 열 장이 어떤 순서로 놓이는가.
- **Part II 도입** — 판별식을 corpus에 대는 열세 장의 8절 격자와 경로별 서사.
- **Part III 도입** — 판정 일곱 장의 전개와 증거 등급 계약.

## Part I — 배경: 자를 만든다

세 층·세 갱신식·비용 4종을 세우고, $\Theta$-경로를 읽는 데 필요한 training 실무를 독자의 inference 어휘로 건설한다. 모든 개념은 (state, update, cost) 객체로 도입되고, 각 장은 worked micro-example로 닫는다.

- **ch01 · 오리엔테이션 — 세 층, 세 경로, 그리고 이름의 문제** — $\Theta$·$W$·$E$와 식 (R)·(U-W)·(U-$\Theta$)·(U-E)·(A), 판별식, 비용 4종, Rosetta 확장분, 이 책이 답하는 여덟 질문의 지도.
- **ch02 · training 레짐 지도** — pre-train·SFT·RL·continual을 "무엇이 언제 움직이는가"로 정의하고 (U-$\Theta$)의 어느 인자를 채우는지 대응시킨다.
- **ch03 · fine-tuning과 PEFT** — full FT·adapter·LoRA·QLoRA를 $\Theta$의 부분 갱신으로 도입한다. rank $r$과 적용 행렬 집합 $\mathcal{S}$가 여기서 정의된다.
- **ch04 · distillation** — logit·feature·self-distillation과 upward distillation. sleep 라운드의 교사가 누구인가라는 질문의 출처.
- **ch05 · RL for LLM** — RLHF·DPO·RLVR과 reward hacking. 보상으로 $\Theta$를 옮기는 경로와 그 검증 비용.
- **ch06 · 데이터를 만드는 법 — synthetic과 붕괴** — $\mathcal{R}_k$를 만드는 $\mathrm{gen}(\cdot;B_s)$의 실체, 자기생성 데이터의 제약과 model collapse.
- **ch07 · 망각과 replay** — catastrophic forgetting, rehearsal, EWC, GEM, 성능 행렬 $\Pi$와 replay 혼합비 $\omega$. $\rho$의 정체가 여기서 정해진다.
- **ch08 · CLS 이론과 생물학적 sleep** — 왜 두 시스템인가, 왜 오프라인인가. 이 라인이 빌려 쓴 은유의 출처(결산은 ch23).
- **ch09 · 기억 용량** — bits-per-parameter, memory layer의 용량 실측, 저랭크 쓰기의 용량. $C_{\text{cap}}$의 상한.
- **ch10 · 외부 기억과 검색 비용** — RAG, vector·graph store, $\mathrm{ret}(E,q)$의 비용 분해. $E$층의 형태와 값.

## Part II — 계보: 판별식을 대 본다

각 장은 8절 고정 구조를 따른다: bridge-in(전작이 남긴 open question) → 문제의식 → 통일 표기 core mechanism과 대응표 → 어느 층을 언제 쓰는가 → 비용 4종 → 실험과 스케일 → systems/serving 함의 → 한계와 bridge-out. 이 부는 아무것도 판정하지 않는다.

- **ch11 · 분기 장 — 판별식과 전수 분류** — 네 조건과 다섯 단계 적용 절차, corpus 전수 분류(통과·불충족·배경), 이름과 기제의 불일치, 경로 간 침묵 지도, 강제된 침묵과 선택된 침묵의 구분.
- **ch12 · Memorizing Transformers** — $E$층의 초기형. 학습 중 캐시 적재라는 쓰기와 그 한계.
- **ch13 · RAG에서 MemGPT로** — $\hat c$가 항등사상인 지점에서 벗어나는 과정, 그리고 MemGPT가 판별식 (1)을 만족하지 않는다는 사실.
- **ch14 · Letta `Sleep-time Compute`** — 이 라인의 이름을 만든 논문. learned context, 상속 없는 bridge-in, 자체 제작 벤치, 높은 예산에서의 이득 역전.
- **ch15 · Mem0·Zep·ReasoningBank·SCM** — 프로덕션 $E$-경로 네 시스템의 $\mathrm{wr}$ 형태 비교, 그리고 품질 1위가 반복적으로 "기억 없음"이라는 사실.
- **ch16 · Generative Adapter — 경첩** — 산출물은 $\Theta$ 모양인데 갱신식은 (U-W)이고 시계는 wake다. 분류가 모양이 아니라 (갱신식, 시계)의 쌍으로 이루어지는 이유.
- **ch17 · `Do Language Models Need Sleep?`** — $W$-경로의 유일한 정면 사례. KV cache를 비우기 전 recurrence를 오프라인에 추가로 도는 절차.
- **ch18 · Nested Learning과 Memory Caching — 연속체는 다리인가** — 갱신 주파수 연속체가 표기의 다리인지 기제의 다리인지 판정하고, $W$층 용량 천장을 읽는다.
- **ch19 · LoRA와 모델 편집** — $\Theta$에 쓰는 두 primitive(저랭크 증분과 직접 편집)와 각각의 상한. ROME 자신의 유보를 포함한다.
- **ch20 · SEAL** — 모델이 자기 학습 데이터를 만들어 $\Theta$를 갱신하는 두 시간척도 절차.
- **ch21 · `Language Models Need Sleep`** — wake $W$ → sleep $\Theta$의 순서, Knowledge Seeding과 Dreaming. corpus에서 bridge-in이 교과서적으로 성립하는 거의 유일한 사례.
- **ch22 · 반증 축 — 가중치에 사실을 계속 넣을 수 있는가** — $\Theta$-경로의 falsifier 후보와 rate–distortion 제약, 그리고 Part I에서 이송된 방법론 결함들.
- **ch23 · 생물학이 실제로 보인 것** — 은유의 담보를 확인한다. sleep 실험과 dreaming 실험이 통제군 대비 실제로 보인 것과 보이지 않은 것.

## Part III — 판정

Part II가 확립한 사실과 이 책의 실험(X1–X4) 위에서 결론을 낸다. 각 장은 판정할 질문 → 증거 → 분석 → 판정 → 반증 조건의 다섯 절이며, 마지막 절은 생략되지 않는다.

- **ch24 · 무엇이 실제로 sleep-time compute인가** — 3분할의 확정과, 이름과 기제의 간격이 일시적 지체인가 구조인가에 대한 판정.
- **ch25 · 통일 비용 회계** — 세 경로를 하나의 축에 올린다. 상각식 (A)에서 빠진 항, $\kappa$ 민감도, 레짐 간 순서(X1).
- **ch26 · 상태 용량과 정보 밀도** — 경로 간 자릿수, dtype이 정하는 정보 밀도, 사용자 수에 대한 스케일링(X2). corpus에서 바이트로 보고한 논문이 0편인 자리.
- **ch27 · 유망한가 — 경로별 판정** — 원 질문에 답한다. 경로마다 다른 답과 그 조건, 그리고 $E$-경로 시스템 평가 결함의 묶음 표.
- **ch28 · scaling law — $B_s$를 축으로** — $B_s$를 독립변수로 놓은 점들을 통일 축에 재배치하고, 보고 실태 전수 감사로 law의 성립 가능성을 판정한다(X3).
- **ch29 · system·infra와 memory-device 기회** — 배포 형태의 변화, shared-weight batching이 깨지는 지점, 웜 계층 delta 스테이징. memory-centric 논증을 펴는 유일한 자리.
- **ch30 · 한계와 반증 조건** — 이 책의 판정을 뒤집을 관측을 구체적으로 적는다. 반복 consolidation의 열화율(X4)과 남은 공백.

## 뒷부분 (back matter)

- **부록 A · 표기 대응 총람** — 각 논문의 원 표기 → 이 책의 기호. 원문 수식 번호 병기.
- **부록 B · 판별식 전수 분류표** — corpus 전편의 통과·불충족·배경 판정과 각 판정이 걸린 조건.
- **부록 C · 의무 caveat 총람** — 논문에 불리한 사실의 전체 목록과 그것을 담는 장.
- **부록 D · 실험 X1–X4 재현** — 설정·상수 출처·고정 시드·등급 계약과, 각 실험이 반박하지 **않는** 것.
- **참고문헌** — vendored 원문 위치 기준.
- **용어 색인(glossary)** — 각 개념의 정의 소유 장으로 연결.
