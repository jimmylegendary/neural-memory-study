# Sleep-Time Compute v2 Seminar — 125 slides

2026-08-06 source freeze의 `study-kr-stc`(30장)와 `experiments/stc`(X1–X4)를 2시간 발표 + 30분 Q&A로 옮긴 125-slide 한국어 deck이다. v1 deck이 쓰던 28-slide 승인 design을 exact template로 그대로 가져오고, 15개 source frame을 재사용했다. 새 geometry primitive를 만들지 않고 inherited text와 speaker notes만 rewrite한다.

## 산출물

- [편집용 PPTX](../../build/publications/SLEEP-TIME-COMPUTE-V2-SEMINAR-KR.pptx)
- [발표·검토용 PDF](../../build/publications/SLEEP-TIME-COMPUTE-V2-SEMINAR-KR.pdf) — 125 pages
- [Deck manifest](reports/deck-manifest.json), [text edit log](reports/text-edit-log.json)
- [Template audit](reports/template-audit.md), [deviation log](reports/template-deviation-log.md)

125/125 slide에 speaker note가 있고, 각 note는 `[발표 대본]`(그 슬라이드에서 말할 내용)과 `[Sources]`(study 절 번호·논문 위치·실험 result.json)를 함께 담는다. 수치를 쓰는 슬라이드는 `status` 라벨로 증거 등급을 표시한다 — 확립 / 논문 수치 / 회계(측정 아님) / 감사 / 최소모형(LLM 아님) / 부재 확인 / 본서 판정.

## 내용 구성

| 절 | 슬라이드 | 무엇 |
|---|---|---|
| ORIENTATION | 1–9 | 왜 이 주제인가, 답할 질문 7+1개, 등급 계약 |
| BACKGROUND | 10–30 | 세 층·세 갱신식·비용 4종·상각식, training 어휘 9종, Rosetta |
| DISCRIMINANT | 31–46 | 판별식 네 조건, 다섯 단계 절차, corpus 29편 전수 분류, 침묵 지도 |
| E/W/THETA-PATH | 47–77 | 세 경로 순회 — 대표 논문의 기제와 그 대가 |
| EXPERIMENTS | 78–92 | X1–X4의 설계·등급·결과 |
| VERDICT | 93–107 | 경로별 판정, scaling law 판정, device 좌표 |
| SYSTEMS·LIMITS | 108–116 | 웜 계층 요구, 자기 정정, 반증 조건 |
| Q AND A | 117–125 | 예상 질문 3매, 실험 제안, 근거 지도, 한 장 요약 |

내용 원천: `study-kr-stc/front`(서사·순서), `part2/ch11`(판별식·전수 분류), `part3/ch24–ch30`(판정 7장), `experiments/stc/X1–X4/result.json`(실험 4건), `dossier/stc-v2/PRE-RESEARCH.md §5`(척추 논지).

## 재현

Deck authoring은 presentations skill이 제공하는 `@oai/artifact-tool` runtime을 사용한다. `python-pptx`, PptxGenJS, direct OOXML은 authoring에 쓰지 않는다. LibreOffice는 PDF export에만 쓴다.

Repository root에서:

```bash
# 0) source template이 없으면 history에서 복원한다 (v1 폐기 커밋에서 삭제되었다)
git show 9079bad^:presentation/sleep-time-compute/build/sleep-time-compute-research.pptx \
  > presentation/sleep-time-compute/build/sleep-time-compute-research.pptx

# 1) template inspect 산출물(layouts + ndjson)을 만든다 — build/deck-work는 gitignore 대상이다
node <presentations-skill>/template_following_scripts/inspect_template_deck.mjs \
  --workspace build/deck-work \
  --pptx presentation/sleep-time-compute/build/sleep-time-compute-research.pptx

# 2) frame map → starter → deck
node presentation/sleep-time-compute-deep/src/generate-frame-map.mjs
node presentation/sleep-time-compute-deep/src/prepare-starter.mjs
node presentation/sleep-time-compute-deep/src/build-deck.mjs

# 3) PDF
soffice --headless --convert-to pdf --outdir build/publications \
  build/publications/SLEEP-TIME-COMPUTE-V2-SEMINAR-KR.pptx
```

`content.mjs`가 절별 모듈 8개를 모아 slideNumber를 부여하고, import 시점에 내용 계약을 검사한다(id 중복, pattern 범위, refs·script 존재, script 최소 길이). 계약이 깨지면 build 전에 즉시 실패한다.

## v1에서 고친 것 (harness)

렌더러·레이아웃·visual contract는 그대로 두었고, 다음 세 가지만 손봤다.

1. **runtime 경로** — `presentations/26.802.11031` → `26.805.11740`. 이전 버전이 disk에서 사라져 harness가 아예 뜨지 않았다.
2. **multi-paragraph rewrite** — inherited shape 중 문단이 둘 이상인 것은 `shape.text.replace(original, …)`가 문단 경계를 넘지 못해 **v1 텍스트가 그대로 남았다**(첫 build에서 177개). 이제 문단 수를 유지한 채 시각 폭 기준으로 분배해 다시 쓴다. 재검증: 남은 미치환 shape 2개는 planner에 하드코딩된 cover 문자열뿐이다.
3. **speaker note에 발표 대본 추가** — `slide.script`가 있으면 note 앞부분에 `[발표 대본]` 블록으로 넣는다(없으면 종전과 동일).

## Pattern 사용 시 주의 (내용 작성자용)

`build-deck.mjs`의 `textPlanner`가 shape 이름으로 슬롯을 배정하므로, pattern마다 요구 슬롯 수와 허용 길이가 다르다. 이번 판에서 확인한 것:

- **pattern 12**는 원형 6칩 도식이다. body가 한 줄을 넘으면 원과 겹친다 — 한글 기준 16자 이하의 키워드구로 쓴다. 숫자 칸은 한 자리이므로 `bigNumbers`에 `1..6`을 준다(`content.mjs`가 기본값을 채운다).
- **pattern 27·28**은 번호 목록 14칸이다. points가 14개 미만이면 남는 칸이 `refs`로 채워져 저장소 경로가 목록처럼 보인다. 반드시 14개를 채운다.
- **pattern 3·5·6·8·9·10·11·14·15·19·22·23·25**는 같은 point가 두세 슬롯에 중복 렌더되거나 작은 상자에 긴 문자열이 들어간다. 이번 판은 쓰지 않았다.
- **pattern 16**의 equation 상자는 좁다(14자 이하). body 슬롯 두 개(`nfr-result` 13자, `nfr-carriers-note` 52자)를 비워 두면 v1 텍스트가 남으므로 points[0]·points[1]의 body를 채운다.
- **pattern 18**의 하단 상자는 짧다. takeaway를 30자 이내로 쓴다.

## 검증 상태

- slide XML 125 · notes XML 125 · `[발표 대본]` 125/125 · `[Sources]` 125/125 · PDF 125 pages.
- 미치환(v1 잔존) shape: 2 (cover 하드코딩 문자열).
- **overflow 자동 검사는 돌리지 못했다.** presentations skill의 `slides_test.py`가 `pdf2image`를 요구하는데 이 host의 python에 없다. 대신 (i) starter layout 대비 시각 폭 비교 스크립트와 (ii) 렌더 PNG 육안 확인으로 패턴별 상한을 잡았다. 위 "Pattern 사용 시 주의"가 그 결과다.

## 권리·공개 경계

Deck은 저자 작성 synthesis지만 인용 source와 일부 source-derived visual을 포함한다. 외부 공개 전 organizational legal/publication review가 필요하며, `internal-only` translation PDF나 source plate를 deck package와 함께 공개하지 않는다.
