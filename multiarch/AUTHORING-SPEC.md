# Multi-architecture HW-twin authoring spec (for the research+author+verify pipeline)

목표: 동일한 Part III 실험 8개를 **8개 아키텍처**에서 동일 형태로 돌리기 위해, 각 칩의 HW-twin을
**여러 번의 독립 검색 + 여러 번의 적대 검증**으로 만들고 **필드별 provenance + confidence**를 붙인다.
성능 수치 신뢰성이 이 프로젝트의 최우선 가치다 — **추측을 사실로 포장하지 말 것.** 모르면 BRONZE로 표기.

## 대상 8종 (정확한 칩 식별)

| key | 칩 | 비고 |
|---|---|---|
| `h100` | NVIDIA H100 SXM5 (GH100) | GOLD 완료 (기준 twin, `twins/h100.json`) |
| `b100` | NVIDIA B100 (Blackwell, GB100 die, 700W bin) | die는 B200과 공유; B100은 낮은 전력 bin |
| `vr200` | NVIDIA Vera Rubin (Rubin GPU) | ⚠️ 내부 미공개 — BRONZE 예상, 추정 명시 필수 |
| `tpu-v7` | Google TPU v7 "Ironwood" | 최신 TPU (2025 발표); v6e Trillium을 교차참조 |
| `mtia2` | Meta MTIA v2 (2nd-gen, 2024/25) | PE grid·SRAM 일부 공개, ALU 미상 가능성 |
| `mi355x` | AMD Instinct MI355X (CDNA4) | 데이터 얇으면 MI300X(CDNA3, 완비)로 보강/대체 |
| `groq-lpu` | Groq LPU / TSP | ⚠️ **HBM 없음** — SRAM 기반(아래 §SRAM 주의) |
| `wse3` | Cerebras WSE-3 | ⚠️ **HBM 없음** — wafer-scale SRAM(아래 §SRAM 주의) |

**단일 가속기 단위(minimal-device convention).** 각 twin은 **칩 1개** 수준으로 저작한다(H100 1장, B100 1장,
MI355X 1장 …). 실험은 backing memory tier(HBM 또는 on-chip SRAM) → reuse tier → compute leaf만 쓴다.
fabric/multi-GPU 레벨은 넣지 않는다(단일 decode-state 질문을 잘못 bound함).

## 필수 필드 (이게 없으면 실험이 깨진다)

RB-05 스키마. 최소 3레벨: backing memory → reuse(on-chip SRAM/L2) → compute leaf.

```json
{ "level_id":"hbm", "role":"hbm", "capacity_bytes":<B>, "bandwidth_bps":<bps>,
  "energy_pj_per_byte":<rd>, "energy_pj_per_byte_wr":<wr>, "line_bytes":<burst>,
  "provenance":{"value":"vendor_whitepaper|press|third_party|estimate","confidence":"GOLD|SILVER|BRONZE","source":"<url/문서>"},
  "children":[
   { "level_id":"sram", "role":"sram", "capacity_bytes":<B>, "bandwidth_bps":<bps>,
     "energy_pj_per_byte":<pj>, "line_bytes":<B>,
     "children":[
      { "level_id":"mxu", "role":"compute", "instances":<N>,
        "peak_macs_per_s":<per-instance MAC/s>, "matrix_unit":[m,n,k], "dtype_bytes":2,
        "acc_dtype_bytes":4, "clock_hz":<hz>, "energy_pj_per_mac":<pj> } ] } ] }
```

- `bandwidth_bps` (memory): tier 대역폭. **필수.** device peak BW.
- `capacity_bytes` (memory): tier 크기. reuse tier 용량이 GEMM C-block·fusion spill을 정한다. **필수.**
- `peak_macs_per_s` × `instances` (compute): **per-instance** peak × instances = device peak. **둘 다 필수.**
  device peak FLOP/s = peak_macs_per_s × instances × 2 (MAC=2FLOP). dense BF16 기준으로 datasheet와 대조.
- `matrix_unit [m,n,k]`: systolic/tensor array 형태. `dtype_bytes`: 원소 바이트(bf16=2).
- `energy_pj_per_byte` / `_wr`: read/write 에너지(비대칭 — write-heavy state DSE에 중요).
- `line_bytes`: ISA 이동 granularity(burst). `clock_hz`, `energy_pj_per_mac`: 있으면 채움.
- 각 수치에 `provenance` 객체(value/confidence/source) — **필드별로**.

## 신뢰도 등급 (confidence) — 필드별 + twin 전체

| 등급 | 기준 |
|---|---|
| **GOLD** | 2개 이상 독립 출처(vendor whitepaper/datasheet + 3rd-party 확인)로 die/PE 내부까지 일치. 발표에서 단정 가능. |
| **SILVER** | vendor 발표/공식 수치는 있으나 PE 내부 일부는 합리적 도출(예: FLOPS+clock에서 array 크기 역산). 발표에서 조건부 인용. |
| **BRONZE** | 발표/추정에 의존, 내부 미공개(예: Vera Rubin). **directional only — 발표 단정 금지, "예비"로만.** |

twin 전체 등급 = **가장 낮은 load-bearing 필드**의 등급(BW·peak·capacity 중 하나라도 BRONZE면 twin=BRONZE).

## provenance 타입
`vendor_whitepaper`(공식 whitepaper/datasheet) · `press`(발표/기사) · `third_party`(SemiAnalysis/Chips&Cheese/Hot Chips/ISCA 등 분석) · `estimate`(도출/추정 — 근거 식 명시). **`fitted` 금지.**

## SRAM 아키텍처 주의 (Groq LPU, Cerebras WSE-3)
이 둘은 **HBM이 없다.** backing tier = on-chip SRAM(Groq ~230MB급, Cerebras ~44GB on-wafer).
- `role:"hbm"` 대신 backing을 `role:"sram"`으로 두되 실험이 읽는 필드명은 동일하게(bandwidth_bps/capacity_bytes).
- 이 경우 "HBM spill cliff"는 물리적으로 없음 → 실험 해석이 근본적으로 다름(state가 on-chip에 상주).
  이건 결함이 아니라 **핵심 대비 결과**다. spec에 `arch_note`로 "no HBM; on-chip resident" 명시.
- Cerebras는 wafer-scale라 "1 chip"=전체 wafer. per-PE 수치와 wafer 총계 둘 다 기록.

## 검증 프로토콜 (반드시)
1. **다중 검색**: 각 load-bearing 수치를 최소 2개의 독립 경로로 확인(vendor + 3rd-party). 한 번 검색으로 끝내지 말 것.
2. **적대 재검증**: 저작된 각 수치를 별도 에이전트가 신선한 검색으로 반박 시도 → confirm/dispute/correct.
3. **자릿수 sanity**: device peak FLOP/s·HBM BW·SRAM 크기가 H100 기준(아래)과 같은 자릿수 범위인지 확인.
4. **disagreement → 등급 강등**: 출처가 엇갈리면 confidence를 한 단계 내리고 range로 기록.

## H100 GOLD 기준값 (자릿수 sanity anchor)
- HBM3: BW 3.35e12 bps (3.35 TB/s), capacity 80 GB, 7 pJ/B, line 32–64 B
- on-chip: L2 50 MB @ 1e13 bps; SMEM 228KB/SM @ 3.34e13
- compute: per-SM 3.748e12 MAC/s × 132 SM = 4.947e14 MAC/s → **~0.99 PFLOP/s dense BF16**; matrix_unit [16,8,16]; clock 1.98e9 (sustained 1.83e9)
- TDP 700W, N4 공정
