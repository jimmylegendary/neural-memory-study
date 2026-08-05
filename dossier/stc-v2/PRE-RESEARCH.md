# PRE-RESEARCH — Sleep-Time Compute v2 (S0 종합)

**버전**: v1.0 (2026-08-06, S0 종료판)
**기반**: `notes/stc-v2/*.json` 28편, `papers/`·`papers/stc/` vendored 원문,
`dossier/stc-v2/S0-DESIGN.md`(프레임), `experiments/stc/X1·X2`(실험 warrant)
**지위**: Part I–III의 장 배정·bridge 사슬·척추 논지는 이 문서가 정한다.

---

## 1. S0의 결론 한 문단

통일 판별식을 corpus에 적용한 결과, **"sleep-time compute"이라는 이름 아래 모인 문헌의
상당수가 어떤 층도 오프라인에 갱신하지 않는다.** 이름이 기제보다 넓게 쓰이고 있다.
판별식을 통과한 것만 남기면 세 경로($E$, $W$, $\Theta$)가 나오는데, 이 셋은 서로를 거의
인용하지 않고, 비용을 바이트로도 FLOP으로도 보고하지 않아 서로 비교할 수 없으며,
$E$-경로에서는 품질 1위가 반복적으로 "기억 시스템 없음"이다. 따라서 "유망한가"는 현재 문헌
상태에서 **그대로 답할 수 없는 질문**이고, 답하려면 (a) 무엇이 실제로 이 범주에 속하는지
판별하고 (b) 통일 회계를 세우는 두 단계가 선행해야 한다. 이 책이 그 둘을 하고, 그 위에서
경로별로 판정한다.

---

## 2. 판별식과 전수 분류

### 2.1 판별식

> 논문 $P$가 sleep-time compute인가?
> **(1)** 질의 $q$가 도착하기 **전에**, **(2)** 예산 $B_s>0$을 써서, **(3)** $\Theta$·$W$·$E$
> 중 **어느 한 층의 상태를 실제로 바꾸고**, **(4)** 그 변화가 **이후 질의**에 쓰이는가.
> 넷을 모두 만족해야 한다. 하나라도 빠지면 인접 문헌이지 이 범주가 아니다.

### 2.2 분류 결과

**A. 판별식 통과 — 세 경로**

| 논문 | 갱신 층 | 시간척도 | 비고 |
|---|---|---|---|
| Letta `Sleep-time Compute` (2504.13171) | $E$ | 질의 전 1회 | 이 라인의 이름. `wr`은 파괴적 덮어쓰기 |
| MemGPT (2310.08560) | $E$ | 대화 중 | 쓰기가 오프라인이 아님 — **경계 사례**, §2.3 |
| Mem0 (2504.19413) | $E$ | 메시지쌍마다 | `wr` ∈ {ADD, UPDATE, DELETE, NOOP} |
| Zep (2501.13956) | $E$ | 에피소드마다 + 주기적 community refresh | 이중 시간 그래프 |
| ReasoningBank (2509.25140) | $E$ | 과제 종료 후 | `wr` = 순수 합집합, 정리·감쇠 없음 |
| SCM (2604.20943) | $E$ | 대화 후 | **제목은 Θ를 가리키나 가중치 불변** |
| Multi-Timescale (2605.05097) | $E$ | 이벤트마다 | 유일 실험에서 LLM 미사용 |
| `Do LMs Need Sleep?` (2605.26099) | $W$ | KV cache 비우기 전 $N$회 재귀 | W-경로의 유일한 정면 사례 |
| SEAL (2506.10943) | $\Theta$ | 두 시간척도 | 자기 생성 self-edit |
| `LM Need Sleep` (2606.03979) | $\Theta$ (←$W$) | sleep 라운드 | wake $W$ → sleep $\Theta$ 순서 |

**B. 판별식 불충족 — 인접 문헌이지 sleep-time compute이 아님**

| 논문 | 왜 아닌가 | 이 책에서의 역할 |
|---|---|---|
| Nested Learning (2512.24695) | 모든 갱신이 (U-W). $\Theta$는 아무것도 안 움직이는 $f=0$ 끝점. **연속체는 표기의 다리이지 기제의 다리가 아니다** | ch18 — 연속체 주장의 검증 |
| Memory Caching (2602.24281) | $B_s=0$. 세그먼트 경계는 갱신이 아니라 체크포인트 저장. 갱신식이 아니라 **읽기식**을 바꾼 논문 | ch18 — W층 용량 천장 |
| Memory Layers (2412.09764) | 배포 후 어떤 층도 안 바뀜. pre-training 1회 | ch09 — $\Theta$층 **용량 실측 공급원** |
| Generative Adapter (2411.05877) | 갱신은 (U-W), 시계는 wake. 산출물만 $\Theta$ 모양 | ch16 — **경첩 장** |
| LoRA (2106.09685) | 쓰기 primitive이지 오프라인 절차가 아님 | ch03·ch19 — 쓰기 도구 |
| ROME / MEMIT | 직접 편집. **ROME은 스스로 "실용적 방법으로 의도되지 않았다"고 명시** | ch19 — 쓰기 도구와 그 상한 |
| RAG (2005.11401) | $\hat c$가 항등사상. $S(c;B_s)$가 추론을 하지 않음 | ch10 — $E$의 조상 |
| Memorizing Transformers (2203.08913) | 쓰기가 학습 중 캐시 적재 | ch12 — $E$의 초기형 |
| EWC / DGR / GEM | 지속학습 일반. 질의 전 오프라인 개념 없음 | ch07 — 망각 대응의 원형 |
| model collapse (2404.01413) | $\mathcal{R}$이 자기생성일 때의 제약 | ch06 — **반증 축** |
| 용량·이론 4편 | 기제가 아니라 제약 | ch09·ch22 |
| 생물학 3편 | 은유의 출처 | ch08·ch23 |

### 2.3 경계 사례의 처리

MemGPT는 쓰기가 **대화 중**에 일어나므로 "질의 전"을 엄밀히 만족하지 않는다. 그러나 그 쓰기의
산출물이 **이후 질의**에 쓰이고, Letta STC가 이 계보를 자기 조상으로 지목하므로 $E$-경로의
출발점으로 서술하되 **판별식 (1)을 만족하지 않는다는 사실을 명시**한다(ch13 의무 caveat).

---

## 3. 계보와 bridge 사슬

```
     생물학·CLS (ch08, ch23)          용량·이론 (ch09, ch22)
        │ 은유만 흘려보냄                    │ 제약을 건다
        │ (아래 셋 중 누구도 인용하지 않음)      │
   ┌────┴──────────────────────────────────┴────┐
   │              ch11 분기 장 — 판별식             │
   └────┬─────────────┬──────────────┬───────────┘
        │             │              │
   E-경로          W-경로          Θ-경로
   ch12 MemTrans   ch16 GenAdapter  ch19 LoRA·편집
   ch13 RAG·MemGPT ch17 NeedSleep?  ch20 SEAL
   ch14 Letta STC  ch18 NL·MemCache ch21 LM Need Sleep
   ch15 Mem0·Zep·RB                 ch22 반증
```

**bridge 사슬의 실제 상태 — 이것이 발견이다.**

- ch14(Letta STC)의 bridge-in은 **상속받을 open question이 없다.** 참고문헌 22개이고,
  계보 진술은 같은 저자의 MemGPT에 대한 유비 하나뿐이다. 이 장은 **상속 없는 시작**으로 쓰고
  그 사실 자체를 서술한다.
- ch15(Mem0·ReasoningBank)는 **Letta STC를 인용하지 않는다.** 같은 경로의 후속 논문이 이 라인의
  이름을 만든 논문을 보지 않는다. 사슬이 끊긴 자리를 [평가] 블록으로 명시한다.
- ch17(`Do LMs Need Sleep?`)은 다른 두 경로를 각각 한 문장씩 언급하면서
  **자기 경로의 주류 계보(Titans/Atlas/Nested Learning)에 침묵한다.**
- ch21(`LM Need Sleep`)은 Nested Learning의 "오프라인 통합 미구축" 공백을 **명시적으로 상속한다.**
  corpus 전체에서 bridge-in이 교과서적으로 성립하는 거의 유일한 사례다.
- ch18(Nested Learning)도 상속 진술이 없다("none-stated").

### 3.1 경로 간 침묵 지도 (참고문헌 전수 확인 결과)

| 논문 | $E$ 인용 | $W$ 인용 | $\Theta$ 인용 | 생물 인용 |
|---|---|---|---|---|
| Letta STC | 자기 계보만 | ✗ | ✗ | ✗ (제목에 빌려 씀) |
| MemGPT | ○ | ✗ | ✗ | ✗ |
| Mem0 | ○ | 1건(문제 증거로만) | ✗ | 수사적 |
| Zep | ○ | ✗ | ✗ | ✗ |
| ReasoningBank | ○ | ✗ | ✗ | ✗ ("sleep" 0회) |
| `Do LMs Need Sleep?` | 한 문장 | 자기 주류 계보 ✗ | 한 문장 | — |
| Nested Learning | ✗ (Letta STC 8개월 선행인데 없음) | ○ | EWC만 | 절단 인용 |
| Memory Caching | ✗ | ○ | ✗ | ✗ ("sleep" 0회) |
| LoRA | ✗ | ✗ | ✗ | ✗ (51개 중 0/0/0) |
| ROME | ✗ | ✗ | 다른 Θ 갈래도 ✗ | ✗ |
| Memory Layers | ✗ | ✗ | ✗ | ✗ (44개 중 0) |
| model collapse | ✗ | ✗ | ✗ | ✗ (양방향 전무) |
| 기억용량(2505.24832) | ✗ | ✗ | ✗ | ✗ |
| rate–distortion | **○ 전면** | 확인됨 | 확인됨 | — |

연대가 강제한 침묵(ROME·MEMIT·EWC·DGR·PLOS는 세 경로보다 앞섬)과 **선택된 침묵**(Letta STC,
Nested Learning, Mem0, ReasoningBank)을 구분해 서술한다. 전자는 계보 사실이고 후자가 발견이다.

---

## 4. 정직성 의무 caveat (확정)

`style/STC-STYLE-NOTATION.md` §6.4를 이 표로 갱신한다. 해당 장 본문에 없으면 결함이다.

| caveat | 의무 장 |
|---|---|
| Letta STC의 5×·13%·18%·2.5×는 저자 제작 벤치(Stateful GSM-Symbolic/AIME, Multi-Query)에서 나온 값이고, Stateful AIME은 문맥이 비어 있는 인스턴스를 포함하며 그 개수를 밝히지 않는다 | ch14 |
| Letta STC의 비용모델은 생성 토큰만 센다. prefill 항이 없고, $\kappa_{\text{cost}}=10$은 벤더 문서 인용이며 민감도 분석이 없다 | ch14, ch25 |
| Letta STC의 이득은 높은 test-time 예산에서 **역전**한다(두 과제 모두). 기제 미제시 | ch14 |
| Multi-Query 데이터셋의 90.9%가 모델 생성 문항이고 생성 답의 검증이 보고되지 않았다 | ch14 |
| Mem0의 자기 표에서 full-context(72.90%)가 Mem0(66.88%)를 이긴다. 논문도 인정한다 | ch15 |
| Mem0의 graph 확장은 single-hop·multi-hop에서 **더 나쁘다**. 지연 3.22×, 토큰 2.05× | ch15 |
| Zep은 어떤 ablation도 없다. DMR에서 자명한 full-conversation baseline이 이미 비교대상을 이기고, 남은 마진 0.4%p에 신뢰구간·시드·유의검정이 없다 | ch15 |
| Zep은 knowledge-update에서 **퇴행**한다(gpt-4o-mini 76.9→74.4) — invalidation 기제가 존재하는 바로 그 범주에서 | ch15 |
| ReasoningBank은 검색 경험이 늘수록 단조 악화(49.7→46.0→45.5→44.4)하고, 출하 기본값이 최소 비영값이다. 일반화 헤드라인은 29 인스턴스(1과제=3.4점) | ch15 |
| ReasoningBank의 판정기 정확도는 72.7%이고, 견고성 실험은 실제 판정기 편향 대신 대칭 잡음으로 대체했다 | ch15 |
| SEAL의 "GPT-4.1 능가"는 세 열 중 한 열에서만 성립한다 | ch20 |
| `LM Need Sleep`은 Semantic Reward를 빼면 AIME-25가 **오르는데**(69.2 vs 69.0) "모든 구성요소가 기여한다"고 쓴다. 기제는 pre-trained Llama/Qwen 위의 graft | ch21 |
| Nested Learning의 Table 6은 캡션의 "모든 구성요소가 기여" 주장을 스스로 반증한다(inner-projection q 제거 시 ppl 12.19 < Hope 12.24). 지속학습 헤드라인은 restructuring을 분리하는 통제군이 없다 | ch18 |
| Memory Caching의 Table 5 한 행이 000/000/000으로 비어 있는데 캡션은 "모든 설계 선택이 긍정적으로 기여"라고 단정한다 | ch18 |
| Generative Adapter의 초록 헤드라인 수치(31.5)가 본문 표 어디에도 없다 | ch16 |
| MEMIT의 zsRE Score 50.7은 Specificity 26.6에 지배되며, 이는 편집하지 않은 GPT-J의 27.0보다 **낮다** | ch19 |
| ROME은 스스로 "지식 저장 기제 이해용 도구이며 실용적 방법으로 의도되지 않았다"고 쓴다. Θ-경로 primitive로 세우는 것은 이 책의 재구성이다 | ch19 |
| EWC 논문에는 결과표가 없고 EWC의 정확도 수치가 본문 어디에도 없다 | ch07 |
| RAG의 학습된 검색기가 FEVER에서 BM25에 진다(논문 인정) | ch10 |
| model collapse의 헤드라인은 VAE/CelebA에서 성립하지 않는다(증가를 늦출 뿐) | ch06 |
| PLOS sleep 실험에서 강한 baseline(interleaved)이 방법과 거의 같고, 우월성 주장은 실증이 아니라 이론적이라고 논문이 인정한다 | ch08, ch23 |
| PAD(eLife)는 **episodic memory 혼합 없이 자발 활동만 써도 유의한 차이가 없다**고 저자가 보고한다 — 논문 제목이 가리키는 성분("dreaming from mixed episodes")이 통제 대비 효과를 보이지 않았다 | ch08, ch23 |
| PAD는 **sleep 단계 순서를 바꿔도 성능 차이가 없다**. 저자는 생물학의 sequential hypothesis를 인정하면서 모델의 순서 독립성을 단계별 시냅스 변화가 작기 때문으로 추정한다 | ch08, ch23 |
| PAD의 표현 품질 지표는 linear classifier readout 하나뿐이고(저자가 "명백한 단순화"라고 인정), CIFAR-10 절대 성능은 약 59%다. 성능 근거로 인용하면 오용이다 | ch08, ch23 |
| 기억 용량 논문은 "3.6 bits-per-parameter"를 여섯 가지 다른 값으로 인쇄한다 | ch09 |
| 세 경로가 서로를 인용하지 않는 구간이 존재하며, 일부는 연대가 아니라 선택이다 | ch11, ch27 |
| 상각식 (A)는 $N_q$를 안다고 가정한다. 실서빙 $N_q$ 분포를 보고한 논문이 corpus에 없다 | ch14, ch25 |
| corpus 전체에서 상태 용량을 바이트로 보고한 논문이 0편이다 | ch26 |

### 4.1 Part I에서 이송한 불리 사실 (해당 장이 반드시 흡수한다)

Part I은 개념 장이라 논문 평가에 속하는 불리 사실을 싣지 않았다. 아래는 **이송 대상**이며,
받는 장이 본문에 담지 않으면 결함이다. 근거는 각 노트의 `unfavorable_facts`에 있다.

| 이송 항목 | 출처 노트 | 받는 장 | 이송 사유 |
|---|---|---|---|
| EWC 7항(replay buffer 크기 불일치, Fisher를 stale off-policy buffer에서 추정, 20M frame gating, $o(n/\log n)$ 조건 미검증, Fig. 3C 단일 게임, A/B 라벨 오기, v2 초고 흔적) | `ewc` | ch22 | ch07 분량 상한 |
| DGR 2항(3자 결합 권고 미검증, 생물학 근거가 인용 신경과학에만 의존) | `deep-generative-replay` | ch22 | 〃 |
| GEM 10항(F2–F14: Table 2↔App. B 불일치, γ ablation 부재, 단일 시드, MNIST 스트림 25.6% 원문 저장 등) | `gem` | ch22 | 〃 |
| 기억용량 논문의 membership-inference 내부 결함(U4·U6·U7·U8·U9·U13·U14·U16·U17·U20) | `lm-memorization-capacity` | ch22 | ch09 논지 밖 |
| rate–distortion: Experiment 1에서 어느 압축법도 random-eviction 대조군을 유의하게 이기지 못함, 저자가 "방법 순서는 논점이 아니다"라고 씀. 예측 (4)의 super-linearity 미확립 | `rate-distortion-compaction` | ch22 | 〃 |
| LoRA: RAG에 LoRA를 붙이면 QuALITY 63.79→42.42로 무너짐, KMSDCD가 절반의 모델에서 single-LoRA를 이김, Table 2의 42.42 반복(복사 오류 정황), LongBench v2 30문서·∞Bench 20문서의 통계적 공백, Adapter가 QQP·SST-2에서 LoRA를 이기는 행 | `lora`, `lora-as-knowledge-memory` | ch19 | ch03 분량 상한 |
| Memory Layers: 채택된 +swilu가 자기 ablation 표에서 최고가 아님, vanilla Memory가 PEER에 짐(1.3b NQ 9.83 vs 12.33), MOE 학습/평가 불일치, 8B에서 Llama3.1 8B에 9개 중 7개 뒤짐 | `memory-layers` | ch19 | ch09는 용량 항목만 실었다 |
| RAG 8항(TriviaQA split 의존성 56.8 < DPR 57.9, MS-MARCO 비대칭 비교, Jeopardy 저자 제안 과제·452쌍 human study, Table 4 Specificity 합 93.0%, 산문-표 불일치 17% vs 11.7%, null-document 3변형 폐기, RAG-Seq/Tok 순서 비일관, 단일 corpus 증거) | `rag` | ch13 | ch10은 $E$층 형태·비용 분해만 다룬다 |

전달해야 할 단서 하나: GEM App. B.1.1의 **인쇄된 BWT와 같은 행렬에서 직접 계산한 값이 어긋난다**
(원문 자체 불일치). ch07이 이를 정정·명시했다. 같은 인쇄값을 재사용할 수 있는 ch18·ch25·ch30은
ch07의 처리를 따른다.

### 4.2 Part I 형식 정리 목록 (S5 조립 QA에서 일괄 처리)

개별 장을 다시 열지 않고 조립 직전 한 번에 처리한다.

- ch08: 표 8-1~8-6 캡션이 표 **아래**에 있다 → 위로(§5.4).
- ch02: 템플릿 고정 절에 번호가 붙어 있다(`## 02.11 (state, update, cost) 정리`) → 번호 제거(§5.2).
- ch02 §02.3: `(→ NM STYLE §1.2)`로 기준 문서를 지목한다 → 책 안의 위치로(§5.5).
- ch03 §03 머리말: $\mathcal{S}$를 "장-국소 기호"로 선언한다 → v0.2에서 전역 등록됨을 반영.
- ch06 §06.4: 채택식의 표본 기호 $s$가 SGD step 첨자와 충돌 → $z$로 교체(§1.3).
- ch09: 절 자기 참조가 `09.2`와 `(§09.3)`로 섞여 있다 → `§` 통일(§5.2).

---

## 5. 척추 논지 — A1 확정

> **"sleep-time compute"은 축이 아니라 세 개의 서로 모르는 축이며, 그 이름은 기제보다 넓다.**
> 통일 판별식을 적용하면 이 이름 아래 모인 문헌의 상당수가 어떤 층도 오프라인에 갱신하지
> 않는다. 남는 세 경로($E$·$W$·$\Theta$)는 (i) 서로를, 때로는 같은 경로의 창시 논문마저
> 인용하지 않고, (ii) 비용을 보고하지 않아 비교 불가능하며, (iii) $E$-경로에서는 품질 1위가
> 반복적으로 "기억 없음"이다. 그러므로 "유망한가"는 판별과 회계를 세운 뒤에만 답할 수 있는
> 질문이고, 그 둘을 세우면 답은 **경로마다 다르다**.

**왜 A1인가.** 후보 A2(쓰기 용량이 새 병목)·A3($B_s$ scaling law)·A4(반증 우선)는 전부 A1의
**하위 판정**으로 들어간다 — A2는 ch26·ch29, A3는 ch28, A4는 ch22·ch30. A1만이 사용자의 원
질문("정말 promising한가, 주류가 될까, 최선인가, 대안 대비 우월한가")을 **답할 수 있는 형태로
바꾼다**. 그리고 A1은 로컬 실험으로 warrant를 만들 수 있다(X1·X2 완료).

**NM D4 제약 승계.** memory-centric 논증은 근거가 실제로 있는 지점에만 쓴다. v2에서 그
지점은 X2가 정량화한 **웜 계층 delta 스테이징**이며, NM이 짚은 HBM RMW 기회와 **다른 자리**임을
명시한다. 두 책의 결론을 억지로 하나로 잇지 않는다.

---

## 6. ToC와 분량 배정

| 장 | 제목 | pp | 근거 노트 |
|---|---|---|---|
| **Part I — 배경 (≈92pp)** | | | |
| ch01 | 오리엔테이션 — 세 층·세 경로·Rosetta | 6 | S0-DESIGN |
| ch02 | training 레짐 지도 | 10 | — (NM ch02 위에서 시작) |
| ch03 | fine-tuning과 PEFT | 10 | lora |
| ch04 | distillation | 8 | lm-need-sleep, seal |
| ch05 | RL for LLM | 10 | seal, reasoningbank |
| ch06 | 데이터를 만드는 법 — synthetic과 붕괴 | 10 | model-collapse, seal |
| ch07 | 망각과 replay | 10 | ewc, deep-generative-replay |
| ch08 | CLS 이론과 생물학적 sleep | 10 | sleep-snn-plos, pad-dreaming |
| ch09 | 기억 용량 | 10 | lm-memorization-capacity, memory-layers, lora-as-knowledge-memory |
| ch10 | 외부 기억과 검색 비용 | 8 | rag |
| **Part II — 계보 (≈140pp)** | | | |
| ch11 | 분기 장 — 판별식과 전수 분류 | 10 | 전체 |
| ch12 | Memorizing Transformers | 8 | memorizing-transformers |
| ch13 | RAG에서 MemGPT로 | 10 | memgpt |
| ch14 | Letta `Sleep-time Compute` | 14 | letta-stc |
| ch15 | Mem0·Zep·ReasoningBank·SCM | 14 | mem0, zep, reasoningbank, scm, multi-timescale |
| ch16 | Generative Adapter — 경첩 | 10 | generative-adapter |
| ch17 | `Do Language Models Need Sleep?` | 12 | need-sleep-recurrence |
| ch18 | Nested Learning과 Memory Caching — 연속체는 다리인가 | 12 | nested-learning, memory-caching |
| ch19 | LoRA와 모델 편집 | 12 | lora, rome, memit |
| ch20 | SEAL | 10 | seal |
| ch21 | `Language Models Need Sleep` | 14 | lm-need-sleep |
| ch22 | 반증 축 — 가중치에 사실을 계속 넣을 수 있는가 | 10 | continual-facts-in-weights, rate-distortion-compaction |
| ch23 | 생물학이 실제로 보인 것 | 10 | sleep-snn-plos, pad-dreaming |
| **Part III — 판정 (≈72pp)** | | | |
| ch24 | 무엇이 실제로 sleep-time compute인가 | 8 | ch11 결과 |
| ch25 | 통일 비용 회계 | 10 | **X1** |
| ch26 | 상태 용량과 정보 밀도 | 10 | **X2** |
| ch27 | 유망한가 — 경로별 판정 | 12 | 전체 |
| ch28 | scaling law — $B_s$를 축으로 | 10 | X3 예정 |
| ch29 | system·infra와 memory-device 기회 | 12 | X2, X4 예정 |
| ch30 | 한계와 반증 조건 | 8 | 전체 |

front/back matter ≈ 16pp. **합계 ≈ 320pp.**

---

## 7. 실험 계획

| 실험 | 상태 | 지원 장 | 무엇을 warrant하는가 |
|---|---|---|---|
| **X1** 통일 비용 회계 | ✅ 완료·결정론 | ch14, ch25 | 상각 논증의 유효 범위, 레짐 간 순서 |
| **X2** 상태 용량 회계 | ✅ 완료·결정론 | ch26, ch29 | 경로 간 자릿수, dtype이 정하는 정보 밀도, 웜 계층 규모 |
| **X3** $B_s$ scaling 형태 | ⬜ | ch28 | 보고된 $B_s$–품질 점들을 통일 축에서 재배치. 논문 대부분이 $B_s$ 미보고임을 정량화 |
| **X4** 반복 consolidation 열화 $\rho$ | ⬜ | ch29, ch30 | 자기생성 $\mathcal{R}$로 $K$라운드 반복 시 열화. CPU 소형 재현 |

X3·X4는 Part III 집필 직전에 실행한다. 등급 규칙은 NM `experiments/REPRODUCE.md` §0을 승계한다 —
비율·crossover·순서는 본문 단정문, 절대치는 하한/방향성.

---

## 8. S0 종료 점검

- [x] 원문 vendored — `papers/` + `papers/stc/` 35편 신규 포함
- [x] deep-read 28편, 전부 유효 JSON, `unfavorable_facts` 비지 않음
- [x] 통일 프레임 확정 (S0-DESIGN)
- [x] 집필 기준 확정 (STC-STYLE-NOTATION v0.1) — §1.7·§4.4·§6.4는 이 문서로 갱신
- [x] 판별식 + 전수 분류
- [x] bridge 사슬 + 침묵 지도
- [x] 정직성 caveat 표 확정
- [x] 척추 논지 확정 (A1)
- [x] ToC + 분량 배정
- [x] `pad-dreaming-elife` 노트 3차 패스 보완 (2026-08-06, 원문 대조로 편집자 직접 작성 — 새 caveat 3항이 §4에 추가됨)

**S0 종료.** 미결 항목 없음.
