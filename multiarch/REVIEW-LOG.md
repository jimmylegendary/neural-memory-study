# 리뷰 로그 — 다중 라운드 검증 기록

성능 수치 신뢰성이 최우선이므로, twin 저작과 실험 결과를 **여러 겹의 독립·적대 검증**으로 통과시켰다.
이 문서는 각 라운드가 무엇을 잡았고 무엇을 고쳤는지의 감사 추적이다.

## 라운드 1 — twin 저작 + 이중 적대검증 (workflow, 28 agents)
칩당 파이프라인: 다중 독립 검색(리서처) → 신선 검색 반박 2명(vendor 위주 / 3rd-party 위주) → 조정.
- 산출: 7 신규 twin + 필드별 provenance/confidence(`specs/*.json`) + 등급.
- 잡은 것(예): B100 SM수 vs clock trade-off, TPU VMEM 미공개, Groq SRAM BW 정의차, Vera Rubin 전면 미공개(BRONZE).
- 결과: h100·mtia2 GOLD / b100·mi355x·tpu-v7·groq·wse3 SILVER / vr200 BRONZE. 전부 `validate_twin` PASS.

## 라운드 2 — 결과 재검증 + 감사 (workflow, 10 agents)
8개 twin headline(BW·peak·용량·backing) 독립 재확인 + 결과표 산술·물리 감사 + 보고서 과대주장 감사.

### 확정(confirm) — 신선 검색으로 재확인, 수정 불요
- **b100 / mi355x / tpu-v7 / mtia2 / groq-lpu / h100**: 4개 load-bearing headline 전부 vendor+3rd-party 재확인.
  (mi355x 2.5 PF = AMD 공식; tpu-v7 7.38 TB/s·2307 TFLOPS = Google Cloud 문서 정확; mtia2 204.8 GB/s·177 TF 확인.)
- **vr200**: "진짜 미공개"임을 재확인(BF16 dense는 어디에도 미발표) → BRONZE 정당. 단 BF16 점추정 6.25 PF는
  다소 높을 수 있음(semianalysis 신호 ~3.6 PF dense) → **directional이라 결론 불변**, 범위 5–13 PF로 표기 유지.

### 수정 반영(applied fixes)
1. **[MAJOR] wse3 compute peak 교정**: 125 PFLOPS는 Cerebras **sparse** FP16 마케팅 수치(10× sparsity).
   **dense FP16 = 12.5 PFLOPS(1.25e16)** 로 교정(`twins/wse3.json` peak_macs_per_s 6.944e10→6.944e9). 용량(44GB)·BW(21PB/s)는 GOLD 불변.
   → **파급**: wse3 ridge 5.95→**0.60**. decode AI 0.59와 거의 같아져 **Cerebras만 roofline knee에 위치**(나머지 7개는 memory-bound).
   오류 교정이 오히려 더 정확한 발견(BW-rich 극단에서 bound가 knee로 이동)을 드러냄. 보고서 ①②③·표 갱신.
2. **[MINOR] 표 B_max@10ms(wse3) 2596→32596**: 콘솔 출력 열 붙음으로 인한 전사 오류. `results/wse3.json`(32596.29)와 일치시킴.
3. **[MINOR] robustness 주장 범위 한정**: "100배 오차로도 memory-bound 불변"은 HBM/LPDDR(여유 380~1455×)에만 참.
   Groq ~4×, Cerebras ~1%(knee)로 명시. §4②·§5 갱신.
4. **[NIT] §4③ HBM-GPU decode 범위**: 하단 0.29 ms/tok는 vr200(BRONZE) → GOLD/SILVER 범위 0.8~1.9로 교체, vr200 제외 명시.
5. **[NIT] 표기**: 컬럼 헤더 "dense BF16"→"BF16/FP16"(groq/wse3는 FP16 스펙), tpu-v7 tier에 ⚠️(reuse-tier BRONZE) 복원.

### 감사 결론
두 감사관 모두 `physical_ok=true`, `verdict=needs_minor_fixes`(위 반영으로 해소). 남은 인정 한계:
- **절대 ms/tok·µs·mJ**은 전부 ideal roofline 하한 — 보고서가 이미 "단정 안 함"으로 표기(과대주장 아님).
- **AI 정의**(분자=QKV+update MAC, 분모=state RMW 바이트)는 E3 검증본의 관례 그대로. Cerebras knee는 이 관례 하에서 ~1% 여유이므로 "경계"로만 진술(단정적 compute-bound 아님).
- **groq 절대치**는 vendor aggregate BW 기반 → effective(~1/3)면 ~3배, 그래서 절대치 유보.
