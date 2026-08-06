# claims/stc-v2 번들 검증 보고 — 2026-08-06

대상: `claims/stc-v2/bundle.json` (검증 전 89 claims = P1 29 + P2 60)
결과: **88 claims (P1 28 + P2 60)** — 1건 삭제, 2건 수정, 1건 보강(등급 범례).
검증자는 직전 라운드의 `gate-report.txt`와 외부 모델 검토를 **참고만** 하고 모든 판정을 다시 내렸다.

---

## 0. 한 줄 요약

번들의 **구조는 흠이 없었다**(중복 0, 포인터 318/318 해석, 전사값 154/154 바이트 일치).
문제는 전부 **등급 정합성**에서 나왔고, 그것은 veridraft 게이트도 직전 라운드의 `verify_bundle.py`도
검사하지 않는 층이다. 가장 무거운 건은 X1의 "load-bearing 결과"라고 선언된 claim이
**자기가 인용한 숫자에 의해 반증**되고 있었다는 것이다.

| | 검증 전 | 검증 후 |
|---|---|---|
| claims | 89 (P1 29 / P2 60) | **88 (P1 28 / P2 60)** |
| results | 29 | 28 |
| result 블록 JSON 포인터 | 159 | 154 |
| veridraft gate | 89/89 PASS, exit 0 | **88/88 PASS, exit 0** |
| grade_legend 항목 | 13 (9개 태그 미정의) | **22 (미정의 0)** |

---

## 1. 실행한 검사

전부 이 세션에서 직접 실행했다. 실행하지 못한 검사는 실행했다고 쓰지 않았다.

### 1.1 veridraft 게이트 — **실제로 돌렸다**

```
cd /home/jimmy/repos/neural-memory-study
export PYTHONPATH=/home/jimmy/repos/veridraft
python3 -m veridraft.cli --config veridraft.stc-study.config.json import-bundle claims/stc-v2/bundle.json
python3 -m veridraft.cli --config veridraft.stc-study.config.json gate sleep-time-compute-v2
```

- 수정 **전**: `imported bundle sleep-time-compute-v2: 89 claims, 29 results` → gate `89/89 [PASS]`, `blocked: (none)`, exit 0.
- 수정 **후**: `imported bundle sleep-time-compute-v2: 88 claims, 28 results` → gate `88/88 [PASS]`, `blocked: (none)`, exit 0.

**게이트가 무엇을 검사하는지 확인했다.** 출력은 claim마다 `admissible_evidence=1`뿐이다 —
즉 **증거의 존재와 종류만** 본다. statement를 읽지 않고, 그것을 result.json과 대조하지 않는다.
그래서 아래 §2.1의 반증된 claim이 89/89 PASS를 받았다. **게이트 통과는 등급 정합성의 증거가 아니다.**

### 1.2 구조 검증 — 직접 작성한 독립 스크립트

`verify_bundle.py`의 로직을 재사용하지 않고 처음부터 다시 짜서 돌렸다. 검증 후 기준:

| 검사 | 결과 |
|---|---|
| 모든 claim에 `claim_id`·`type`·`statement` 존재 | 88/88 ✅ |
| `claim_id` 중복 | **0** ✅ |
| P1 `result_refs` 비어 있지 않음 · 선언된 result로 해석됨 | 28/28 ✅ |
| P2 evidence 비어 있지 않음 · 전부 `source_artifact` · 전부 locator 보유 | 60/60 ✅ |
| evidence 경로가 **디스크에 실재** | 41/41 ✅ |
| evidence 경로가 pin된 commit에 **git-tracked** | 41/41 ✅ |
| `result_pointers` / `results[].pointers`가 **실제 result.json에서 해석됨** | **308/308** ✅ |
| 전사된 metric/verbatim 필드가 원본과 **바이트 일치** | 154/154 ✅ |
| orphan result / 미선언 result | 0 / 0 ✅ |

### 1.3 등급 정합성 — 수작업 대조 + 스크립트

`experiments/REPRODUCE.md §0`의 정직성 계약과 각 result.json의 `meta.caveat_tags`·`findings`를
기준으로 P1 28건을 전수 대조했다. 여기서 §2의 세 건이 나왔다.

기계 검사로 돌린 것:
- **P1 숫자 출처**: statement에 나오는 모든 숫자를 인용한 result.json에서 찾는다 (177건 검사).
- **P2 숫자 출처**: statement에 나오는 모든 숫자를 vendored `.txt` 원문에서 찾는다.
- **X1 레짐 순서 재계산**: sweep 12칸 전부에서 순서를 다시 뽑아 claim의 주장과 대조.
- **X4 절대치 이송 금지**: `NOT-AN-LLM-EXPERIMENT` 태그 claim에 K* 라운드 수·유지율 %·반감기·라운드당 pp가 있으면 실패.
- **X2 bound 방향**: "최소 4.4배" 류가 단정문에 있으면 실패.
- **X3 corpus 범위**: `CORPUS-SCOPED` claim이 "문헌의 N%"로 일반화하면 실패.

산출물: **`claims/stc-v2/verify_claims.py`** (2127 assertions, exit 0).

### 1.4 P2 원문 대조 — 직접 grep

- **부재 주장 36건 검사, 35건 일치, 1건은 내 정규식의 위양성**
  (`exp**lora**tion`이 `LoRA`에 걸린 것 — `p2_memlayers_silent_on_all_paths`는 정상).
- 검증한 부재: Zep의 10개 키워드 0회, Mem0/ReasoningBank/Memory-Caching의 `sleep` 0회,
  Letta 원문의 `FLOP` 0회, Nested Learning의 E-경로 6개 키워드 0회, LoRA의 4개 키워드 0회.
- `p2_genadapter_headline_not_in_tables`: `31.5`가 파일 전체에서 **정확히 1회**, 그것도 초록(line 26)
  — 주장("본문 어느 표에도 없다")이 그대로 확인됐다.
- 숫자를 주장하는 P2 25건 중 **24건은 인용한 원문에 그 숫자가 실재**한다.
  나머지 1건이 §2.3.

### 1.5 스크립트 자체의 음성 검사 (regression test가 실제로 발화하는가)

검사가 절대 실패하지 않으면 아무것도 증명하지 못한다. 결함을 하나씩 다시 주입해 확인했다:

| 재주입한 결함 | 잡힘? |
|---|---|
| 삭제한 GSM≫AIME 순서 claim을 그대로 복원 | ✅ B1 |
| 뒤집힌 "최소 4.4배 / 양자화는 전제" 문장 복원 | ✅ B2 |
| X4 claim에 절대 `K*=46 라운드` 이송 | ✅ B3 |
| corpus 밖으로 비율 일반화 | ✅ B4 |
| P2 표 숫자 59.4 → 59.9 위조 | ✅ C |
| `claim_id` 중복 | ✅ A |
| P1 `result_refs` 비우기 | ✅ A |
| P2 locator 지우기 | ✅ A |
| 전사 metric을 result.json과 다르게 조작 | ✅ A |
| **X3에 없는 census 수 "13편" 주장** | ❌ **놓침** |

마지막 항목은 스크립트의 한계다 — X3 result.json에 13이 다른 맥락으로 실재하기 때문에
(`n_hits: 13`, 인용된 GSM8K 예시의 "13 stickers") 전 파일 검색을 통과한다.
포인터 하위트리로 범위를 좁혀 보았으나 JSON 포인터 **경로**가 값을 나르는 탓에
(`kappa_cost=20.0`, `14000_tokens`, `dense-8B`) 정상 claim 6건에 위양성이 나서 채택하지 않았다.
**§C 검사는 필요조건이지 충분조건이 아니다**라고 스크립트에 명시했다.

---

## 2. 처리 내역 — 무엇을 왜

### 2.1 삭제 — `p1_stc_x1_regime_ordering_is_the_load_bearing_result`

**사유: 자기가 인용한 값에 의해 반증된다.**

claim은 상각 여유의 레짐 간 순서를 `GSM-Symbolic >> AIME > SWE-Features`라 단정하고
이것을 "본문 단정문으로 쓸 수 있는 load-bearing 결과"라고 선언했다.
X1 자신의 sweep에서 `L*/|c|`는 이렇다:

| 조건 | GSM | AIME | SWE | 실제 순서 |
|---|---:|---:|---:|---|
| flops, κ_par=1 | 12.00 | **600.00** | 4.17 | AIME > GSM > SWE |
| flops, κ_par=2 | 6.00 | **300.00** | 2.08 | AIME > GSM > SWE |
| flops, κ_par=5 | 2.40 | **120.00** | 0.83 | AIME > GSM > SWE |
| flops, κ_par=10 | 1.20 | **60.00** | 0.42 | AIME > GSM > SWE |
| pricing, κ_par=1 | 48.00 | **2400.00** | 16.67 | AIME > GSM > SWE |
| pricing, κ_par=2 | 24.00 | **1200.00** | 8.33 | AIME > GSM > SWE |
| pricing, κ_par=5 | 9.60 | **480.00** | 3.33 | AIME > GSM > SWE |
| pricing, κ_par=10 | 4.80 | **240.00** | 1.67 | AIME > GSM > SWE |

**null이 아닌 8칸 전부에서 AIME가 GSM보다 크다.** 한 칸도 예외가 없다.
게다가 claim의 세 numeric 포인터가 전사하는 값이 바로 `12.0 / 600.0 / 4.17` —
**자기를 반증하는 숫자를 자기 warrant로 달고 있었다.**
손익분기 N_q(Q3)로 봐도 같다: AIME 0.32 < GSM 0.84 ≪ SWE 36.75. 어느 지표로도 AIME가 GSM보다 여유가 크다.

원인: warrant인 `X1 findings.F4`가 순서를 "세 파라미터(절감률, 문맥 길이, 병렬 벌수)의 방향"으로
정당화하면서 **`gen_base`(B_t)를 빠뜨렸다.** 그런데 GSM과 AIME는 절감률이 5배로 같고 |c|도 400 대 300으로
비슷하며, 갈라놓는 것은 오직 `gen_base` 600 대 22500(37.5배)이다. 즉 순서를 만드는 파라미터가
F4의 목록에 없다. **claim도 그 warrant도 같은 이유로 결함이다.**

**왜 고치지 않고 지웠나.** 참인 순서(AIME > GSM > SWE)는 sweep 안에서 견고하지만, 그 순서는 전적으로
`gen_base`에 달려 있고 그 값은 실험 자신이 `AXIS-READ`로 표시한 것이다 — 수치표가 없어 **그림 축에서
읽은 값**이다. result.json의 어떤 finding도 그 순서를 주장하지 않는다. 검증 라운드가 warrant에 없는
새 load-bearing 발견을 만들어 넣는 것은 원래의 결함보다 나쁘다.
그리고 실제로 살아 있는 내용 — **SWE-Features가 가장 빡빡하고 1 아래로 내려가는 유일한 레짐** — 은
`p1_stc_x1_swe_regime_crosses_below_one`과 `p1_stc_x1_breakeven_Nq_spread_across_regimes`가 이미 나른다.
따라서 삭제해도 잃는 주장이 없다.

함께 제거: result `r_stc_x1_regime_ordering_is_the_load_bearing_result`,
`provenance_manifest.experiments[X1].claims`의 해당 항목.

> **상류 조치 필요 (이 라운드에서 하지 않음).**
> `experiments/stc/X1-unified-cost-accounting/result.json`의 `findings.F4`는 같은 파일의
> `Q1_Q2_critical_length_by_regime`과 모순된다. result.json은 pin된 warrant blob이라
> **수정하지 않았다** — 고치면 evidence의 pinned-blob 일치가 깨지고 게이트가 막힌다.
> 본문(ch25)이 F4의 순서를 인용하고 있다면 그 문장도 같이 손봐야 한다.

### 2.2 수정 — `p1_stc_x2_dtype_sets_density_and_survives_the_withdrawal`

**사유: bound 방향이 뒤집혀 있다.**

3.64 bits/param은 원 논문이 **하한**이라고 못 박은 값이다(`bound_direction: "LOWER BOUND"`).
claim은 그 하한에서 "표현 자체에서 **최소** 4.4배의 저장 여유가 나온다"를 끌어내고
거기서 "**delta 양자화는 최적화가 아니라 전제다**"를 결론했다. 방향이 반대다:

> 여유 = 저장 폭 / 정보량 = 16 bits ÷ (**≥** 3.64 bits) **≤** 4.4

파라미터가 실제로 담는 bits가 하한 **위로** 올라갈수록 압축 여유는 **줄어든다**.
따라서 4.4배는 여유의 **상한**이고, 보장된 최소 여유는 **없다**. 보장된 여유가 없으면
"양자화는 전제다"는 따라 나오지 않는다.

이것은 X2가 이미 한 번 철회한 것과 **같은 종류의 오류**다 —
`findings.F3_withdrawn_cross_path_density_claim`이 하한을 상한으로 읽은 비교를 철회했는데,
바로 옆의 `F3_quantisation_is_a_precondition_not_an_optimisation`이 같은 오독을 그대로 안고 있다.

- **철회**: "표현 자체에서 최소 4.4배의 저장 여유", "델타 양자화는 최적화가 아니라 전제다".
- **유지**: dtype 축의 2배씩 상승(분모만으로 결정 — result.json의 `what_survives`가 명시적으로 보증),
  세 하한값 bf16 **≥**1.82 / int8 **≥**3.64 / int4 **≥**7.28, 그리고 "corpus의 어느 논문도
  delta 양자화를 논의하지 않는다"는 독립적인 부재 사실.
- **추가**: 4.4가 상한이라는 올바른 방향 진술과, 판정하려면 delta에 알려진 엔트로피를 주입해
  회수율을 재는 별도 실험이 필요하다는 범위 선언.
- 태그에 `WITHDRAWN` 추가, `transport_rule`을 "16/3.64 = 4.4는 여유의 **상한**이며 하한으로 인용 금지"로 교체.
- result 블록의 description에도 "이 두 문장은 claim이 나르지 않는다"는 주석을 달았다.

**약화가 아니라 절제다.** 헤지를 덧붙여 통과시킨 것이 아니라, 근거 없는 두 문장을 들어내고
근거 있는 부분만 남겼다.

> **상류 조치 필요**: X2 result.json의
> `findings.F3_quantisation_is_a_precondition_not_an_optimisation`과
> `Q2_information_density.note`가 둘 다 "최소 4.4배의 저장 여유"라고 쓴다. 방향이 뒤집혀 있다.
> 역시 pin된 blob이라 수정하지 않았다.

### 2.3 수정 — `p2_rb_silent_on_letta`

**사유: 사실이 아닌 부수 숫자.**
"its **2481-line** text"라고 썼는데 `papers/stc/2509.25140.txt`는 **2511줄**이다.
locator의 범위 `txt:1-2481`도 파일 끝 30줄을 덮지 못한다.
load-bearing 내용(`sleep` 0회, `2504.13171` 0회)은 **전체 2511줄에 대해 다시 확인했고 성립한다.**
논증에 아무 역할이 없는 틀린 줄 수를 제거하고 locator 범위를 파일 전체로 고쳤다.

### 2.4 보강 — `grade_legend` 미정의 태그 9개

P1 claim들이 쓰는 caveat 태그 중 **9개가 번들의 `grade_legend`에 정의되어 있지 않았다**:
`E-PATH-TOKENS-ONLY`, `PRINTED-EXAMPLE-N1`, `COMPETITOR-MEASURED`, `EXACT-WITHIN-THE-TOY`,
`SHAPE-FIT-IS-MODEL-SELECTION`, `CAPABILITY-COUPLED-TO-RETENTION`,
`MECHANISM-EXCLUSION-NOT-CONFIRMATION`, `SURVIVAL-HALF-ONLY`, `OMEGA-IS-NOT-A-TOKEN-BUDGET`.
번들만 읽는 사람은 태그를 해독할 수 없다. 각 실험 result.json의 `meta.caveat_tags`에서
**정의를 그대로 복사**해 채웠다(13 → 22항목).

---

## 3. 검사했지만 문제가 없던 것 — 특히 지시받은 세 지점

### 3.1 X4는 LLM 실험이 아니다 → 절대치 이송이 있는가

**없다.** X4 claim 11건 전부 부호·함수형·순서·crossover만 나른다.
`D1_absolute_numbers`가 금지한 ρ의 절대 pp, 유지율 %, K*의 라운드 수, 필요 ω, 반감기가
단정문으로 이송된 사례가 하나도 없다. 오히려 각 claim이 자기 statement 안에서 그 금지를 되풀이한다.

두 개의 숫자만 이송되는데 둘 다 정당하다:
- **"라운드 200에서 replace/accumulate 위험비 약 122배"** — 검산했다.
  replace ~ *k*, accumulate ~ Σ1/i² → 200 ÷ 1.6399 = **121.96**. 배율 σ²/n이 **약분되어 사라지므로**
  이 비는 모형 상수에 의존하지 않고 [model-collapse] Eq.3/4의 **함수형만으로** 결정된다. 이송 가능.
- **"self replay에서 Spearman 1.00"** — 단조성(순서) 진술이며 `self_Kstar` = 1.0으로 정확히 일치.
  gold는 ω≥0.25에서 K=150에 censored이므로 claim이 gold를 단조성 판정에서 **제외한 것도 옳다**
  (result.json의 `monotonicity.note`와 같은 처리).

`gold > self`가 모든 ω에서 성립하는지도 확인했다 — K* 비 2.41 / 2.54 / 2.34 / 2.03 / 1.52,
전부 1 초과이고 censoring은 gold를 키우는 방향이므로 순서 진술은 안전하다.

### 3.2 X2의 3.64 bits/param을 상한처럼 쓴 claim이 있는가

**있었고 §2.2에서 처리했다.** 그 외에는:
- `p1_stc_x2_cross_path_density_comparison_withdrawn`은 철회 자체를 주장으로 올바르게 세운다. 유지.
- `p1_stc_x2_three_paths_differ_by_orders_of_magnitude`는 자릿수와 반사실 표시만 나른다.
  E 56.0 KB / Θ 16.78 MB / W 17177.6 MB에 대해 "수십 KB / 수십 MB / 수십 GB" — 자릿수 진술로 타당. 유지.

### 3.3 X1의 절대 임계치를 단정한 claim이 있는가

**핵심 위반 1건은 §2.1에서 삭제했다.** 남은 X1 claim 5건은 값을 조건부 산술로 명시한다:
- `swe_regime_crosses_below_one`: `L*/|c| = 0.42`(κ_par=10) / `4.17`(κ_par=1) — result.json과 정확히 일치.
  주장의 본체는 "1 아래로 내려가는 영역이 존재한다"는 **부호**이고, |c|=8000 가정을 명시한다.
  X3 D5가 그 가정을 **하한으로만** 지지하는데, |c|가 커지면 L*/|c|는 **더 작아지므로** 결론은 강화된다.
  방향이 안전한 쪽으로 틀려 있어 유지.
- `kappa_cost_is_load_bearing_and_unmeasured`: κ_cost 1→20에서 0.42→8.33 = **정확히 20배**.
  선형 의존이라는 함수형 주장과 정확히 맞는다. 유지.
- `breakeven_Nq_spread_across_regimes`: "GSM은 sweep한 모든 칸에서 N_q<1" — 9칸 전부 확인
  (0.31~0.84) ✅. "SWE는 N_q ≈ 37" — 실제 **36.75**의 반올림. 유지(스크립트에 반올림으로 등록).
- `amortisation_has_an_unamortised_term`, `does_not_refute_the_5x`: 숫자 없는 구조·범위 진술. 유지.

### 3.4 X3

census 수치를 전부 재확인했다 — (a)1 (b)5 (c)2 (d)2 (e)0 (f)19 / 29편, absent 19 중 정의 안 됨 9,
후보 8 중 환산 가능 1, MFU 보고 1/29, ≥2점 4편, 벤치 교집합 ∅, pairwise 0, `can_draw = False`.
**전부 일치.** Θ 0.625 > E 0.222의 **순서만** 인용하고 비율을 금지한 처리도 정확하다.

---

## 4. 고치지 않고 남긴 것 — locator 줄번호 표류

P2 locator가 `papers/…txt:NNN` 꼴로 정확한 줄을 약속하는데, **실제 줄과 어긋난다.**
숫자 앵커가 유일하게 결정되는 12건을 측정한 결과:

| claim | 인용 줄 | 실제 줄 | 표류 |
|---|---:|---:|---:|
| p2_memit_specificity_below_unedited | 827 | 833 | +6 |
| p2_memlayers_peer_beats_vanilla_memory | 694 | 701 | +7 |
| p2_lmns_ablation_reversal | 629 | 641 | +12 |
| p2_seal_genadapter_beats_seal | 1592 | 1614 | +22 |
| p2_seal_gpt41_beats_seal | 570 | 596 | +26 |
| p2_memgpt_gpt4_below_gpt35 | 540 | 569 | +29 |
| p2_zep_knowledge_update_regression | 397 | 439 | +42 |
| p2_genadapter_fullconv_wins_msc | 834 | 880 | +46 |
| p2_nl_transformer_wins_fda | 3345 | 3407 | +62 |
| p2_mem0_fullcontext_wins | 781 | 846 | +65 |
| p2_rb_bs_reported | 2140 | 2214 | +74 |
| p2_zep_trivial_baseline_beats_sota | 257 | 471 | +214 |

표류는 **전부 양수**이고 파일 뒤로 갈수록 커진다 — locator를 쓴 뒤 `.txt` 추출본이
줄바꿈이 늘어난 형태로 재생성됐다는 뜻이다.

**내용은 무사하다**: 위 claim들이 주장하는 숫자는 전부 해당 원문에 실재하고, 문맥도 맞다
(예: SEAL Table 2에서 `… 50.6 | 59.4 | 58.2 …` / `… 43.4 | 49.2 | 46.4 …` — claim이 비교하는 바로 그 쌍).

**그럼에도 줄번호를 자동으로 고쳐 쓰지 않았다.** locator는 여러 참조가 섞인 산문이고,
기계적으로 재작성하면 사람이 확인한 적 없는 정밀도를 새로 만들어 넣게 된다.
확인 가능한 사실 — 표류의 존재·부호·크기와, 숫자가 원문에 실재한다는 것 — 을 여기 기록한다.
`.txt` 추출본을 다시 만들 때 locator를 함께 재생성하는 것이 옳은 처리다.

---

## 5. `verify_bundle.py`(직전 라운드 스크립트) 판정

**구조 검사로는 타당하다.** JSON 포인터를 작업본이 아니라 **커밋된 blob**에 대고 푸는 것
(`git show <commit>:<path>`), 전사 metric을 타입까지 포함해 바이트 비교하는 것, `note` 종류를
증거에서 배제하고 `reading_provenance`로 재등록됐는지 확인하는 것 — 전부 옳은 설계다.
그대로 두었고, 이번 수정본에 대해서도 **PASS (2098 assertions, 0 failures)**로 돈다.

**두 가지 한계를 기록한다.**

1. **등급 검사가 전혀 없다.** statement를 result.json의 findings·caveat_tags와 대조하는 층이 없다.
   §2.1의 반증된 claim이 이 스크립트와 veridraft 게이트를 모두 통과한 이유가 이것이다.
2. **2b 검사가 교정을 막는다.** statement가 `p1-claims.json`/`p2-claims.json`과 **바이트 동일**할 것을
   요구한다. 병합이 몰래 claim을 약화시키는 것을 막는 좋은 장치지만, 동시에 번들을 **아직 검토받지 않은
   초안**에 못 박아 정당한 교정까지 실패시킨다. 이번 라운드는 두 half를 번들과 **함께** 갱신해
   비교 대상을 맞췄고(그래서 2b가 통과한다), 대신 "몰래 바꾸지 않았다"는 성질을 바이트 동일성이 아니라
   **이 문서의 변경 내역**으로 보장한다. 무엇이 어떻게 바뀌었는지는 §2에 전부 있다.

이 두 구멍을 메우려고 **`claims/stc-v2/verify_claims.py`**를 새로 썼다(구조 + 등급 + 숫자 출처, 2127 assertions).
스스로의 한계는 스크립트 docstring과 §1.5에 명시했다.

---

## 6. 최종 상태

```
$ python3 claims/stc-v2/verify_claims.py
claims           : 88  (P1=28, P2=60)   duplicate ids: 0
results          : 28   orphans: 0
A structure      : 308 JSON pointers resolved; every transcribed metric/text byte-compared
B grade rules    : X1 regime ordering recomputed from all 12 swept cells (true order: AIME > GSM > SWE)
                   X2 bound direction (3.64 bits/param = LOWER bound => 4.4x is a CEILING)
                   X4 absolute-transport ban; X3 corpus-scoping; tag/grade coherence
C number warrant : 177 numbers checked against the cited result.json / vendored source text
assertions       : 2127
VERIFICATION: PASS (0 failures)

$ python3 claims/stc-v2/verify_bundle.py
SELF-VERIFICATION: PASS (0 failures)        # 2098 assertions

$ python3 -m veridraft.cli --config veridraft.stc-study.config.json gate sleep-time-compute-v2
88/88 [PASS]   blocked: (none)   exit 0
```

**남은 상류 작업 2건** (result.json은 pin된 warrant blob이라 이 라운드에서 건드리지 않았다):
1. X1 `findings.F4` — 순서 진술이 같은 파일의 값과 모순. 본문 ch25가 인용 중이면 함께 수정.
2. X2 `findings.F3_quantisation_is_a_precondition_not_an_optimisation` 및
   `Q2_information_density.note` — "최소 4.4배"의 방향이 뒤집혀 있음.
