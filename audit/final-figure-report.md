# 최종 그림 QA 리포트 (final-figure-report)

**라운드**: 통합-수준 마무리 QA — 그림 QA 담당
**날짜**: 2026-07-12
**대상**: `study-kr/{part1,part2,part3}` 전체의 `![...](.../figures/exp-*.png)` 임베드 전수
**기준**: STYLE-NOTATION v1.1 §5, figures/SPEC.md, 정직성 계약

---

## 1. 임베드 인벤토리 (15건 — 전부 part3)

| # | 파일:행 | 번호 | exp 이미지 | 소속 장 일치 |
|---|---|---|---|---|
| 1 | ch18:40 | 그림 18-1 | exp-a-kv-ttt-crossover | ✓ |
| 2 | ch19:87 | 그림 19-1 | exp-a-kv-ttt-crossover | ✓ |
| 3 | ch19:130 | 그림 19-2 | exp-f-scaling-fits | ✓ |
| 4 | ch19:186 | 그림 19-3 | exp-c-chunk-roofline | ✓ |
| 5 | ch20:140 | 그림 20-1 | exp-c-chunk-roofline | ✓ |
| 6 | ch20:170 | 그림 20-2 | exp-b-state-placement | ✓ |
| 7 | ch20:176 | 그림 20-3 | exp-e-rmw-cliff | ✓ |
| 8 | ch21:41 | 그림 21-1 | exp-c-chunk-roofline | ✓ |
| 9 | ch22:18 | 그림 22-1 | exp-c-chunk-roofline | ✓ |
| 10 | ch22:32 | 그림 22-2 | exp-a-kv-ttt-crossover | ✓ |
| 11 | ch23:101 | 그림 23-1 | exp-d-frequency-tiers | ✓ |
| 12 | ch24:66 | 그림 24-1 | exp-b-state-placement | ✓ |
| 13 | ch24:72 | 그림 24-2 | exp-e-rmw-cliff | ✓ |
| 14 | ch24:90 | 그림 24-3 | exp-d-frequency-tiers | ✓ |
| 15 | ch25:42 | 그림 25-1 | exp-f-scaling-fits | ✓ |

- part1/part2/front/back 에는 exp-* 임베드 없음(확인).
- **장 번호 정합**: 15/15 모두 소속 장 번호와 일치(chNN → 그림 NN-M). 충돌·오배치 0건.
- **task 브리핑은 14건으로 명시했으나 실제 15건**(exp-c가 4장, exp-a가 3장에 임베드). count-of-record 정정 필요.

## 2. 경로 resolve

- 소스는 전부 상대경로 `../../figures/exp-*.png`. `study-kr/part3/chNN.md` 기준 `../../figures/` → `/home/jimmy/repos/neural-memory-study/figures/` 로 정확히 해석됨.
- 빌드 `build/BOOK.md`는 이 상대경로를 **절대경로**로 재작성해 임베드(15/15 확인). `build/BOOK.pdf`(3.29MB) 정상 빌드, `build/build_err.txt` 0행 → 전 경로 resolve.
- 6개 PNG 전부 유효(PIL 로드 OK). exp-f만 1481×1069(2×2 grid), 나머지는 가로형 1~2 panel.

## 3. 캡션↔내용 대조 (실제 PNG Read 검증)

- **exp-a** (2 panel: 좌=1.3B anchor crossover S*, 우=S*∝m·d² scaling): 18-1/19-1/22-2 캡션 = "crossover + scaling" → ✓
- **exp-b** (2 panel: 좌=energy/latency 4-device tradeoff, 우=residency crossover layers-that-fit): 20-2/24-1 → ✓
- **exp-c** (2 panel: AI(C) roofline / chunk C, memory→compute): 19-3/20-1/21-1/22-1 → ✓
- **exp-d** (single: cadence→memory-tier 배정 표, ★=recommended): 23-1/24-3 → ✓
- **exp-e** (single: on/off-die RMW 대역폭 cliff, 3.9× L3→DRAM): 20-3/24-2 → ✓
- **exp-f** (**2×2 4 panel**: (a)ppl-vs-params, (b)ppl-vs-state@1.3B, (c)retention-vs-capacity, (d)TNT train/serve U-curve): **19-2 캡션 불일치 → 수정함(§5).** 25-1 캡션은 panel 수 미주장(제너릭 "digitize scaling fit") → 허용.

## 4. ch21/ch22 임베드 확인 (이전 감사 지적)

- ch21:41 = exp-c 실제 이미지 임베드(그림 21-1) ✓
- ch22:18 = exp-c(그림 22-1), ch22:32 = exp-a(그림 22-2) 실제 이미지 임베드 ✓
- 이전 라운드의 "텍스트만 언급, 임베드 누락" 우려는 **해소됨**.

## 5. 직접 수정 (Edit 완료)

**그림 19-2 (ch19-scaling-laws.md) 캡션 4-panel 불일치 — 수정.**
- 원 캡션(L130)과 재서술(L132)은 "(좌)/(중)/(우)" 3-panel로 기술했으나, exp-f는 2×2 **4-panel**이고 우하 panel = **TNT train/serve chunk mismatch U-curve**가 캡션에서 누락.
- 본문 L127은 이 U-curve를 panel (D)로 명시 서술하며 "scaling law가 아니라 resolution mismatch"라 경고 → 캡션과 본문이 어긋난 상태였음.
- **수정**: L130 캡션을 "(좌상)/(우상)/(좌하)/(우하)" 4-panel로 재기술, U-curve 항목 추가. L132 재서술도 4 quadrant로 갱신(정직성 프레이밍 보존, load-bearing 문구 유지).

## 6. 판단 필요 (미수정, human 결정)

1. **재사용 정책 이탈**: SPEC.md §재사용("draw once, reuse everywhere — 재사용 장은 텍스트 참조만, 이미지 재삽입 금지")과 task의 기대 패턴("최초 임베드 + 이후 (그림 X-Y 참조)")에 반해, exp-* 이미지가 2~4회 **재임베드**됨(exp-c 4×, exp-a 3×). 번호 충돌은 없음(장별 고유 번호). 단 책은 이미 ch18-1을 ch20:8·ch23:151에서 **텍스트 참조**로 처리 → 두 패턴이 혼재(비일관). 결정 필요: (a) 309pp part-분할의 독자 편의를 위해 part별 재임베드 유지, vs (b) 후속 사용을 텍스트 참조로 전환해 SPEC 준수. 정직성/충돌 리스크는 없으므로 편집 정책 결정 사항.
2. **exp-f 재현 경로 부재**: `figures/render_experiments.py`는 docstring·코드 모두 exp-a..e만 생성; exp-f-scaling-fits.png의 render 경로가 어느 스크립트에도 없음(grep 0). 정직성 계약(재현성) 하에서 그림 1건의 provenance 미포착 = 갭. exp-f 생성 코드를 SoT 스크립트에 편입하거나 별도 render 스크립트를 명시할 것.
3. **임베드 count-of-record**: 브리핑 14 → 실제 15. 후속 문서에서 정정.

## 7. 판정

- 장-번호 정합·경로 resolve·캡션 정합: 15/15 통과(19-2 캡션 1건 수정 반영 후).
- 하드 결함(번호 충돌/경로 미해석/그림-내용 오배치): **0건**.
- 열린 항목: 재사용 정책 일관성(§6.1), exp-f 재현 경로(§6.2) — 둘 다 정직성 계약 관련 판단 사항, 빌드/정본 번호 체계에는 영향 없음.
