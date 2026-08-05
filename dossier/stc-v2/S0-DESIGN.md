# S0 설계 — Sleep-Time Compute v2 pre-research

**버전**: v0.1 (2026-08-06)
**지위**: v2의 모든 집필과 deep-read는 이 문서의 프레임을 따른다. 이 문서와 충돌하는 서술은
결함으로 처리한다. 변경은 이 문서를 고쳐서 하고, 본문에서 임의로 다른 프레임을 쓰지 않는다.
**선행**: Neural Memory 모노그래프(`build/BOOK.pdf`, 327pp)와 그 집필 기준
`style/STYLE-NOTATION.md`. v2는 그 방식을 그대로 적용한 두 번째 모노그래프다.

---

## 0. v1이 실패한 이유 — 반복하지 말 것

v1은 70pp Study + 12pp 학회형 + 57pp Training Background + 62pp easy + 112 slides를 냈고
증거 수집(73항목 source registry, primary-source audit 7종)도 성실했다. 실패는 분량이 아니라
**논지 전개 장치의 부재**였다. 구체적으로:

| v1 결함 | v2에서 강제하는 것 |
|---|---|
| 통일 표기 없음 — 논문마다 원문 기호를 그대로 씀 | §2 세 층 프레임으로 전 논문 환원. 표기 대응표 의무 |
| 논문별 장 0개. 계보 전체가 `05-lineage.tex` 103줄 | Part II = 논문 1편 1장, 8절 고정 템플릿 |
| bridge 사슬 없음 — 논문이 서로 독립 나열 | 각 장 §1 bridge-in은 **전작이 명시적으로 남긴 open question**에서 시작 |
| 주장/해설/평가 구분 없음 | 3층 표기 강제(논문 귀속 / [해설] / [평가]) |
| 논문에 불리한 사실 누락 | §7 의무 caveat 표. 해당 장에 없으면 결함 |
| 배경이 별도 문서로 분리 | Part I을 본문에 통합(별도 PDF는 같은 소스에서 파생 빌드) |
| 근거 없는 전략 서술 | 모든 수치 주장에 원문 위치 병기, 자체 주장은 실험 warrant 또는 [평가] 블록 |

---

## 1. 독자 모델

Neural Memory 모노그래프와 **같은 독자 한 명**을 상정한다: transformer inference와
efficient-transformer를 잘 아는 AI system infra·architecture exploration 엔지니어.
KV cache, paged attention, FlashAttention tiling, GEMM shape, MFU, roofline, prefill/decode,
batching은 안다.

v1과의 차이는 이 독자가 **NM 모노그래프를 이미 읽었다**는 것이다. 따라서:

- backward pass, optimizer state, inner/outer loop, test-time learning의 기본은 **전제**한다.
  NM의 해당 장을 참조로 넘긴다("→ NM ch02", "→ NM 식 (M)").
- 반면 **본격적인 training 실무**는 여전히 모른다: fine-tuning 레짐, LoRA/PEFT, distillation,
  RL(RLHF/RLVR), synthetic data 생성, replay buffer, catastrophic forgetting 대응, 평가·검증
  루프. Part I은 정확히 이 공백을 메운다.
- 이 독자의 1번 질문은 항상 **"그래서 그게 decode에서 뭘 바꾸는데?"** 이다. 모든 장은
  이 질문에 답하는 절을 갖는다.

---

## 2. 통일 프레임 — 세 층 기억과 세 경로

### 2.1 상태의 3분할

모델이 답을 만들 때 참조하는 상태를 세 층으로 나눈다. 이 3분할이 v2 전체의 축이다.

| 기호 | 이름 | 무엇 | 갱신 시간척도 | 저장 위치 |
|---|---|---|---|---|
| $\Theta$ | **slow weights** | pre-train된 파라미터 | 학습 라운드(episode) | 가중치 |
| $W$ | **fast weights** | test-time에 움직이는 내부 상태 | 토큰/청크 | 가중치 형태의 상태 |
| $E$ | **external store** | text·vector·graph 기억 | 이벤트/세션 | 모델 밖 |

$\Theta$와 $W$의 정의는 NM `style/STYLE-NOTATION.md` §1.2를 **그대로 승계한다**. 예외 없다.
$E$는 v2에서 신설한다.

읽기는 세 층을 모두 통과한다.

$$
\hat y \;=\; f\big(q;\ \Theta,\ W,\ \mathrm{ret}(E, q)\big)
\tag{R}
$$

### 2.2 세 갱신 규칙

$$
W_t \;=\; \alpha_t\, W_{t-1} + S_t,
\qquad
S_t \;=\; \beta_t\, S_{t-1} - \eta_t \nabla_W\, \ell\big(W_{t-1}; k_t, v_t\big)
\tag{U-W}
$$

$$
\Theta_{k+1} \;=\; \Theta_k \;-\; \eta_\Theta\, \nabla_\Theta\, \mathcal{L}\big(\mathcal{R}_k;\ \Theta_k\big),
\qquad
\mathcal{R}_k \;=\; \mathrm{gen}\big(\mathcal{H}_k;\ B_s\big)
\tag{U-$\Theta$}
$$

$$
E_{k+1} \;=\; \mathrm{wr}\big(E_k,\ \hat c_k\big),
\qquad
\hat c_k \;=\; S\big(c_k;\ B_s\big)
\tag{U-E}
$$

(U-W)는 NM의 master equation (M)과 **글자 그대로 같다**. v2는 NM을 대체하지 않고 그 위아래로
확장한다. (U-$\Theta$)의 $\mathcal{R}_k$는 replay·distillation·합성 데이터를 통칭하는 학습
집합이고 $\mathcal{H}_k$는 그 라운드까지의 경험이다. (U-E)의 $S(\cdot; B_s)$는 질의를 보기 전
문맥에 대해 수행하는 오프라인 추론이다.

### 2.3 sleep-time compute의 조작적 정의

> **정의.** *sleep-time compute*은 질의 $q$가 도착하기 **전**, 유휴 시간에 예산 $B_s$만큼
> 소비하는 계산으로서, 그 산출물이 이후 질의의 wake-time 비용 $B_t$를 줄이거나 정확도를
> 높이는 것을 말한다.

이 정의는 갱신 대상 층을 지정하지 않는다. 그래서 **세 경로**가 생긴다.

| 경로 | 갱신 층 | 대표 | 산출물의 물리적 형태 |
|---|---|---|---|
| **E-경로** | $E$ | Letta `Sleep-time Compute`, MemGPT, Mem0, Zep | 텍스트·벡터·그래프 레코드 |
| **W-경로** | $W$ | `Do Language Models Need Sleep?`, Memory Caching | fast weight 텐서 |
| **Θ-경로** | $\Theta$ | `Language Models Need Sleep`, SEAL, Nested Learning의 CMS | 가중치 delta |

**이 3경로 구분이 v2의 척추다.** 사용자가 원래 세운 축("모수적 update냐, 모델 외부
text/vector/graph냐")을 정밀화한 것이며, 세 경로는 비용 구조가 근본적으로 다르다 — 그것이
Part III의 판정 재료가 된다.

### 2.4 비용 4종 — 모든 장이 채워야 할 표

각 논문의 기여를 반드시 이 네 값으로 환산한다. 값이 논문에 없으면 "논문에 없음"이라고
쓴다(추정치를 본문 수치로 쓰지 않는다).

| 기호 | 이름 | 단위 | 왜 필요한가 |
|---|---|---|---|
| $B_s$ | sleep 계산 예산 | FLOPs 또는 tokens | 상각 가능성의 분자 |
| $L_w$ | wake 지연 기여 | ms/token, TTFT | 독자의 SLA 감각 |
| $C$ | 상태 용량 | bytes/세션 또는 bytes/사용자 | memory-device 기회의 근거 |
| $\rho$ | 망각·열화율 | 라운드당 성능 손실 | 반복 sleep의 지속가능성 |

상각식은 전 장 공통으로 아래 형태를 쓴다($N_q$ = 한 문맥을 공유하는 질의 수).

$$
C_{\text{avg}} \;=\; C_{\text{wake}} \;+\; \frac{C_{\text{sleep}}}{N_q}
\tag{A}
$$

---

## 3. corpus와 장 배정

1차 소스는 `papers/`(NM 라인 + 신규 2편), `papers/stc/`(v2 확보분),
`translations-kr/stc-core/sources/`(의역 원문 plate)에 vendored 되어 있다. v1의
source registry 73항목은 `git show` 로 참조하되, **v2의 인용은 vendored 원문 위치로만** 한다.

### 3.1 Part II 장 후보 (연대순 × 경로)

| # | 논문 | arXiv | 경로 | 역할 |
|---|---|---|---|---|
| 공통조상 | CLS (McClelland 1995) | — | 이론 | 왜 두 시스템인가 |
| 공통조상 | EWC / Deep Generative Replay | 1612.00796 / 1705.08690 | Θ | 망각 대응의 두 원형 |
| 공통조상 | Sleep prevents catastrophic forgetting (PLOS) | — | 생물 | sleep이 실제로 작동한다는 실증 |
| 공통조상 | PAD (eLife) | — | 생물 | dreaming의 표현학습 역할 |
| E-1 | MemGPT | 2310.08560 | E | 외부기억을 OS로 |
| E-2 | **Sleep-time Compute (Letta)** | **2504.13171** | E | 이 라인의 이름을 만든 논문 |
| E-3 | Mem0 / Zep | 2504.19413 / 2501.13956 | E | 프로덕션 외부기억 |
| E-4 | ReasoningBank | 2509.25140 | E | 경험을 재사용 가능한 기억으로 |
| W-1 | **Do Language Models Need Sleep?** | **2605.26099** | W | offline recurrence → SSM fast weight |
| W-2 | Memory Caching | 2602.24281 | W | 성장하는 상태 |
| Θ-1 | LoRA / 모델 편집 (ROME·MEMIT) | 2106.09685 / 2202.05262 / 2210.07229 | Θ | 가중치에 쓰는 두 방식 |
| Θ-2 | Generative Adapter / Memory Layers | 2411.05877 / 2412.09764 | Θ | 한 번의 forward로 파라미터화 |
| Θ-3 | SEAL: Self-Adapting LMs | 2506.10943 | Θ | 자기 학습데이터 생성 |
| Θ-4 | Nested Learning | 2512.24695 | Θ | 갱신 주파수 연속체 |
| Θ-5 | **Language Models Need Sleep** | **2606.03979** | Θ | Knowledge Seeding + Dreaming |
| Θ-6 | SCM: Sleep-Consolidated Memory | 2604.20943 | Θ | 알고리즘적 망각 |
| 용량 | How much do LMs memorize | 2505.24832 | 이론 | 용량의 상한 |
| 용량 | LoRA as Knowledge Memory | 2603.01097 | 이론 | 저랭크 쓰기의 용량 |
| 용량 | Rate–Distortion Memory Compaction | 2607.08032 | 이론 | 무엇을 버릴 것인가 |
| 용량 | Can a LM Learn Facts Continually in Its Weights? | 2607.11020 | 이론 | Θ-경로의 반증 후보 |

최종 12–15장으로 압축한다. 압축 기준은 §5의 bridge 사슬이 끊기지 않는 최소 집합이다.

### 3.2 Part I 모듈 (training 무경험 독자용)

NM의 prereq 커리큘럼(B0–B9)에 이어 **T-모듈**로 번호를 붙인다.

| 모듈 | 내용 | (state, update, cost) |
|---|---|---|
| T0 | training 레짐 지도 — pre-train / SFT / RL / continual | 무엇이 언제 움직이는가 |
| T1 | fine-tuning과 PEFT — full FT, adapter, LoRA·QLoRA | $\Theta$의 부분 갱신 |
| T2 | distillation — logit·feature·self-distill, upward distill | 교사→학생 전이 |
| T3 | RL for LLM — RLHF, DPO, RLVR, reward hacking | 보상으로 $\Theta$ 이동 |
| T4 | 데이터 — augmentation, synthetic·self-generated, model collapse | $\mathcal{R}$을 어떻게 만드나 |
| T5 | replay와 catastrophic forgetting — rehearsal, EWC, GEM | $\rho$의 정체 |
| T6 | CLS 이론과 생물학적 sleep — 왜 두 시스템·왜 오프라인인가 | 이 라인의 은유의 출처 |
| T7 | 기억 용량과 rate–distortion — 얼마나 쓸 수 있나 | $C$의 상한 |
| T8 | 외부 기억 — RAG, vector·graph store, retrieval 비용 | $E$의 비용 |
| T9 | 평가와 검증 — 지속학습 벤치마크, 오염, 누출 | 주장을 어떻게 반증하나 |

각 모듈은 NM Part I 템플릿을 따른다: 목표 박스 → 본문 절(절마다 systems 접점) →
worked micro-example(손으로 따라가는 수치) → 요약 → 자가 점검 → 다음 장으로.

---

## 4. deep-read 추출 규격

논문 1편 = `notes/stc-v2/<slug>.json` 1개. 필수 필드:

```
paper_id, arxiv, title, authors, venue, date, path        # 식별
axis                     # E | W | Theta | bio | theory
problem_stated           # 논문이 스스로 정의한 문제 (원문 위치 병기)
inherited_open_question  # 어느 선행 논문의 어떤 open question을 받았는가 (bridge-in 재료)
core_mechanism           # 통일 표기 §2로 환원한 수식 + 원문 수식 번호 대조
outer_vs_inner           # 무엇이 학습되고 무엇이 test-time에 움직이는가 (표)
update_layer             # (U-W)/(U-Θ)/(U-E) 중 무엇을 어떻게 바꾸는가
cost_table               # B_s, L_w, C, rho — 논문에 있는 값만. 없으면 "absent"
experiments              # 모델 크기·토큰·벤치·수치 (원문 위치 필수)
scale_ceiling            # 이 논문의 실증 상한
concept_ledger_delta     # 신설 / 확장·개명 / 폐기 개념
open_questions           # 논문이 직간접으로 남긴 문제 (bridge-out 재료)
unfavorable_facts        # 논문에 불리한 사실 — ablation 역전, 미검증 합성, 단일 설정 근거 등
systems_implications     # decode·serving·용량 관점 번역
figures                  # 인용 가치 있는 원저자 그림 (번호 + 무엇을 보여주는가)
```

`unfavorable_facts`와 `inherited_open_question`은 **비워둘 수 없다**. 비면 deep-read 미완료로
처리한다. v1의 가장 큰 결함이 이 둘의 부재였다.

---

## 5. bridge 사슬 규약

Part II의 각 장 §1은 **전작이 명시적으로 남긴 open question**을 인용으로 복원하며 시작한다.
사슬은 경로 안에서 연대순으로 잇고, 경로가 갈라지는 지점에는 **분기 장**을 둔다.

```
공통조상(CLS·replay·생물 sleep)
        │
        ├── E-경로 : MemGPT → Letta STC → Mem0/Zep → ReasoningBank
        ├── W-경로 : (NM의 TTT 라인) → Do LMs Need Sleep? → Memory Caching
        └── Θ-경로 : LoRA·편집 → GenAdapter/MemLayers → SEAL → NL → LM Need Sleep → SCM
                                                                        │
                                                        용량·이론(반증 축)
```

경로가 서로를 인용하지 않는 지점은 **그 자체가 발견**이다. 그 단절을 [평가] 블록으로 명시한다
— v1은 이 단절을 서술하지 않고 세 경로를 한 덩어리로 뭉갰다.

---

## 6. Part III 척추 논지 후보

S0 종료 시 하나를 확정한다. 현재 후보:

- **A1 — 3경로 비용 분해**: sleep-time compute은 단일 축이 아니라 비용 구조가 다른 세 축이다.
  E는 상각으로 즉시 이득이 나오고(식 (A)의 $N_q$가 크면 승), Θ는 용량을 실제로 늘리지만
  쓰기 검증 비용이 병목이며, W는 둘의 다리다. "promising한가"라는 질문은 경로를 지정하지 않으면
  답할 수 없다.
- **A2 — 쓰기 용량이 새 병목**: Θ-경로가 주류가 되면 per-user/per-session 가중치 delta가
  상태량을 지배한다. NM의 pair thesis(decode state RMW = memory-centric 기회)를 sleep-time
  쓰기 트래픽으로 확장할 수 있는지 검증한다. **NM D4 제약을 그대로 승계** — 억지 연결 금지.
- **A3 — $B_s$를 축으로 하는 scaling law**: 품질을 $(N, D, B_s, C)$의 함수로 놓았을 때
  $B_s$의 지수와 포화점. Letta가 보고한 5×/13%/18%와 Θ-경로 보고치의 형태가 같은지 다른지.
- **A4 — 반증 우선**: `Can a LM Learn Facts Continually in Its Weights?`와 용량 상한 결과를
  Θ-경로의 falsifier로 세우고, 그것을 통과하는 설계 조건만 남긴다.

선정 기준은 NM D4와 동일하다 — **로컬 실험으로 warrant를 만들 수 있는 것**을 우선한다.

---

## 7. 정직성 의무 caveat (초안)

deep-read 완료 시 확정한다. 현재까지 확인된 것:

| caveat | 의무 장 |
|---|---|
| Letta STC의 5×·13%·18%는 Stateful GSM-Symbolic / Stateful AIME라는 **저자 제작 변형 벤치**에서 나온 값 | E-2 |
| Letta STC의 "learned context"는 가중치 갱신이 아니다 — 논문이 본문에서 명시 | E-2, Part III |
| `Language Models Need Sleep`의 기제는 pre-trained Llama/Qwen 위의 graft — end-to-end meta-learn 아님 (NM ch17에서 이미 확정) | Θ-5 |
| 세 경로가 서로를 인용하지 않는 구간이 존재 | 분기 장, Part III |
| 상각식 (A)는 $N_q$를 안다고 가정 — 실제 서빙에서 $N_q$ 분포는 어느 논문도 보고하지 않음 | E-2, Part III |

---

## 8. 산출물과 게이트

| 산출물 | 경로 | 게이트 |
|---|---|---|
| deep-read notes | `notes/stc-v2/*.json` | 필수 필드 전부, `unfavorable_facts` 비지 않음 |
| 종합 dossier | `dossier/stc-v2/PRE-RESEARCH.md` | bridge 사슬 무결, ToC 페이지 배정 합계 확인 |
| 집필 기준 | `style/STC-STYLE-NOTATION.md` | 예약기호 충돌 0 |
| 본문 | `study-kr-stc/part{1,2,3}/` | veridraft claim bundle PASS |
| PDF | `build/stc/BOOK-STC.pdf` | `build/overflow_gate.py` PASS, 미해결 글리프 0 |
