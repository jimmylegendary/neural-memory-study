# 공동연구 제안 — HOPE-with-Sleep의 memory-tier co-design

> 대상: **Vahab Mirrokni** (Google Research). 그는 Titans·Nested Learning(2512.24695)·Memory Caching(2602.24281)
> 라인의 시니어 저자다 — 이 제안은 그 논문들이 **스스로 열어 둔 open question**과 **언급조차 하지 않은 이음새**를
> 정면으로 겨눈다. 뒷받침 도구: MEMOIR(saw-09) — HOPE-with-Sleep을 memory-tier 하드웨어 위에서 analytical하게
> 모델링하는 DSE 웹앱. 근거: neural-memory-study 검증 노트 + 장치 스펙 리서치 144항목(출처·신뢰등급 포함).
>
> 상태: v0.1 (draft). §1 "우리가 풀려는 문제"는 seminar Open Questions(2단) + qa-memory Q042 종합에서 조립. §2–5는 scaffold.

---

## 0. TL;DR

Titans→Nested Learning→Sleep→Memory Caching으로 이어지는 이 라인은 **"기억을 test-time에 학습시킨다"**는 한
가지 수를 여섯 번 변주했다. 그 과정에서 저자들은 세 가지를 반복해서 인정했다 — (1) catastrophic forgetting은
"안 풀렸다"(NL 결론), (2) 반복 self-improvement는 붕괴할 수 있다(Sleep), (3) 무엇을 어떻게 캐시/통합할지의
**선택은 열려 있다**(MC). 이 명시된 open point들은 서로 다른 조각처럼 보이지만 **하나의 빈칸**을 가리킨다:

> **온라인으로 계속 변하는 기억에서, 무엇을(선별) · 언제(트리거) · 어디로(계층) 옮기고 · 언제 잊을 것인가 —
> 그리고 그 비용은 하드웨어에서 얼마인가.**

앞의 네 조각은 알고리즘 문제로 논의됐지만, 다섯 번째(비용)는 **어느 논문도 모델링하지 않았다**. 우리의 주장은:
이 다섯은 분리된 문제가 아니라 **하나의 algorithm × systems co-design 문제**이며, 최근 NSTM(2607.15271)이
그 알고리즘 쪽 실마리(invariant/contextual 분해 + memory-only probe)를 실험으로 보여줬고, 우리는 그 시스템
쪽(memory-tier 배치·서빙 비용)을 정량화하는 도구를 이미 만들었다. 둘을 합치면 공동연구의 표적이 된다.

---

## 1. 우리가 풀려는 문제

### 1.1 세 논문이 수렴하는 하나의 빈칸

각 논문이 **본문·결론에서 스스로 남긴** open point를 모으면(원문 대조, 부록 A):

- **NL / HOPE (2512.24695).** 결론의 "Is Catastrophic Forgetting Solved?" 절은 스스로 아니라고 답한다 — 망각은
  유한 용량 아래 압축의 자연 귀결이며, "NL은 destination이 아니라 roadmap"이고 진전은 static 심화가 아니라
  **levels(업데이트 주파수) 축**에서 온다. 또한 discussion은 "같은 objective·같은 search space라면 parametric
  메모리는 스케일에서 softmax attention을 못 이긴다"고 못 박고, 이길 축을 **계산 깊이·self-modification**으로만
  지목한다(§8.2는 chunk-size를 "제한 없다"면서 실험은 2값만).

- **Sleep (2606.03979).** 부록 discussion은 반복 self-distillation의 실패모드를 인정한다 — privileged
  conditioning이 epistemic verbalisation을 억압해 **OOD가 최대 40%까지 붕괴**(Qwen3-8B 등)하고, naive 반복은
  정보누출·training collapse를 낳는다. Sleep의 2단 분리(consolidate→dream)는 이 위험을 **"줄일(reduce)"** 뿐
  제거를 주장하지 않는다. 망각을 "용량 부족"으로 재정의하고 파라미터 성장으로 대응하지만, **성장이 언제 멈추나
  / slot이 소진되면 어쩌나**는 언급하지 않는다.

- **Memory Caching (2602.24281).** future work는 "더 표현력 있는 pooling/routing" 한 줄이고(현재는 MeanPooling·
  단순 게이트), §3.4의 **체크포인트 vs 독립 압축기**는 "각자 장단, 과제 의존"으로 열어 둔다. recall 왕좌는
  여전히 Transformer임을 인정한다. 그리고 **캐시가 per-user 상태를 N배로 키우는 서빙 비용**은 다루지 않는다.

세 조각은 표현이 다를 뿐 같은 질문의 다른 면이다. 이를 하나의 빈칸으로 접으면 다음 다섯 하위 질문이 된다.

### 1.2 빈칸의 정확한 형태

| # | 하위 질문 | 어느 논문이 열어 뒀나 | 현재 최선의 단서 |
|---|---|---|---|
| Q-선별 | **무엇을** 옮기나 | MC: Soup 평균이 "언제" 성립하나 무답 | NSTM: 분해 안 된 fast weight의 평균은 붕괴(27.77→20.25) → invariant 성분만 |
| Q-트리거 | **언제** 옮기나 | NSTM은 고정 주기(1Hz)·Sleep은 고정 wake/sleep 경계 | surprise·불일치·memory-only probe 실패 기반 learned trigger (미탐색; NSTM도 "갱신 사이 사건 놓침" 인정) |
| Q-계층 | **어디로** 옮기나 | CMS(주파수)×Sleep expert(수명)×MC(이력) 3축 병행 무논의 | 세 축은 직교 — 조합·상호작용이 열린 설계공간 |
| Q-용량 | **언제 잊나** / 성장 한계 | Sleep: slot 소진·장기 성장 지속가능성 침묵 | 병합·증류 회수 / 더 느린 레벨 추가 / tiered eviction (추론) |
| Q-비용 | **얼마나 드나** (HW) | **어느 논문도 모델링 안 함** | MEMOIR: per-op FLOPs/bytes·roofline·memory-tier 배치·pJ/byte 정량화 |

앞 네 질문은 알고리즘 문헌 안에서 부분적으로 논의됐다. **다섯 번째(비용)만 통째로 비어 있고**, 그것이 나머지
넷을 실제로 결정한다 — 무엇을·언제·어디로 옮길지는 결국 "그 이동이 하드웨어에서 감당되는가"에 걸린다.

### 1.3 왜 지금까지 안 풀렸나 — algorithm × systems co-design 문제이기 때문

이 빈칸이 오래 열려 있는 이유는 그것이 순수 알고리즘 문제도, 순수 시스템 문제도 아니기 때문이다. 세 가지가
동시에 얽힌다.

1. **per-user 상태가 자란다.** self-modifying Titans의 fast weight(각 memory ≈ $2rD^2b$)와 MC의 N개 스냅샷은
   request마다 개별 존재한다 — "weights는 공유된다"는 서빙의 대전제가 깨진다. MC는 이 N배 상태의 회계를 다루지
   않는다.
2. **read/write가 비대칭이다.** NSTM이 실측으로 보였듯(H100에서 memorization 58.14ms / synthesis 27.01ms),
   기억은 매 토큰 **읽지만** 주기적으로만 **쓴다**. 이 비대칭은 어느 계층(cHBM/HBM/zHBM+PIM/HBF/DRAM/SSD)에
   무엇을 두느냐로 비용이 수 배 갈린다 — 순수 알고리즘 시야 밖의 자유도다.
3. **분해가 배치를 정한다.** NSTM의 교훈(consolidation은 invariant/contextual 분해 후에만 성립)은 곧 배치
   원칙이 된다 — 불변 성분(frozen expert)은 read-only라 값싼 계층(HBF)에, 가변 성분은 빠른 계층에. 즉 알고리즘의
   분해가 하드웨어 배치를 직접 지시한다.

그래서 이 문제는 알고리즘 팀과 시스템 팀이 **각자 풀 수 없다.** 알고리즘 쪽 단서(NSTM)와 시스템 쪽 정량화
도구(MEMOIR)가 한 테이블에 있어야 답이 나온다 — 그게 공동연구의 논거다.

### 1.4 문제 — 한 문장

> **self-modifying Titans(sequence) + CMS(Sleep low-rank expert) 구조의 기억을, invariant/contextual 분해를
> 기준으로 언제·무엇을·어느 memory-tier로 이동/통합/망각시킬 때, 주어진 하드웨어에서 정확도(안정성)·용량·
> 시간·에너지의 파레토를 최적화하는 co-design 규칙은 무엇인가.**

이것을 대조군(softmax attention + FFN)과 같은 축에서, 전부 1-layer 기준으로, analytical하게 정량화한다.

---

## 2. 우리가 가져오는 것 — MEMOIR + 분해 원리  *(scaffold)*

- **MEMOIR(saw-09).** HOPE-with-Sleep과 대조군을 op 단위 symbolic FLOPs/bytes로 전개(prefill/decode 분리,
  arithmetic intensity), 최신 TPU + HBM3E roofline, cHBM/zHBM+PIM/HBF/DRAM/SSD 배치 GUI(장치 스펙·fabric·연산/
  데이터 위치 전부 편집 가능, 하드코딩 없음), pJ/byte 에너지, hat-schema식 committee로 "말이 되는" 스펙만.
- **분해 원리를 서빙으로.** NSTM의 read/write 분리를 owner/version 태깅 스케줄(read-only batching + update
  queue)로, invariant 성분을 frozen-expert-on-HBF 배치로 — MEMOIR의 P2/P3 시나리오가 이를 이미 표현.
- **정량화된 tiered eviction.** Q-용량/Q-계층을 HBF→warm→cold→KG 계층 이동 비용으로 환산.

## 3. 공동연구가 답할 구체 질문 (측정가능)  *(scaffold)*

1. invariant/contextual 분해 비율이 바뀔 때 memory-tier별 시간·에너지 파레토가 어떻게 이동하는가?
2. read/write 비대칭을 PIM/cHBM 오프로드로 흡수하면 user batch를 얼마나 키울 수 있는가(TPU 병렬화 여유)?
3. MC의 N(세그먼트 수)·top-k(SSC)를 HW 예산에 맞춰 정할 때 recall–비용 곡선의 무릎은 어디인가?
4. Sleep의 slot 성장 vs eviction을 결합했을 때 장기 안정성(OOD 붕괴 방지)과 용량의 교환은?

## 4. 성공 기준  *(scaffold)*

- 절대값이 아니라 **추세·구조 비교**의 정합(HATIR 원칙) — 여러 시나리오에서 파레토 순서가 물리 인자로 설명될 것.
- 대조군 대비 HOPE-with-Sleep의 병목 축(capacity/update/frozen-expert)이 정량적으로 분리·재현될 것.

## 5. 왜 이 팀 · 왜 지금인가  *(scaffold)*

- **왜 지금:** MC(2026-02)·NSTM(2026-07)이 마지막 두 조각(캐싱의 성립 조건·read/write 분리)을 방금 채웠다.
- **왜 이 팀:** 우리는 이 라인 6편을 검증 노트·의역본·easy booklet로 재구성했고(neural-memory-study), 시스템
  비용을 계산하는 도구(MEMOIR)와 하드웨어 believability 스택(HAT/HATIR)을 이미 보유.

---

## 부록 A — 근거 대조 (claim → 원문)

| 주장 | 출처 |
|---|---|
| forgetting 미해결·"roadmap"·levels 축 | 2512.24695 Conclusion "Is Catastrophic Forgetting Solved?" |
| parametric ⊀ attention (동일 objective/space) | 2512.24695 §8 discussion (Nadaraya-Watson 관점) |
| chunk-size "제한 없음"·실험 2값 | 2512.24695 §8.2 |
| self-distillation OOD 최대 40%↓·collapse·"reduce" | 2606.03979 부록 "Limitations of OPSD / Positioning" |
| 망각=용량 재정의·파라미터 성장 | 2606.03979 Positioning (upward distillation) |
| future work=pooling/routing·recall 왕좌 Transformer | 2602.24281 §6 Conclusion, §5.3 |
| 체크포인트 vs 독립 압축기 "각자 장단" | 2602.24281 §3.4, §5.6 |
| 분해 후 평균만 성립(27.77→20.25 / 29.24→30.09) | 2607.15271 Tab.2 ablation |
| read/write 비대칭 58.14/27.01ms | 2607.15271 §4.4 |
