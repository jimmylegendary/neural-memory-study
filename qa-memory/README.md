# QA 연상메모리 — 공부하며 던진 질문의 축적 DB

Jimmy가 이 계열(TTT/neural-memory)을 공부하며 던지는 **모든 질문**을, taskops의 "안다 4축"으로 분류·추출해 **연상메모리(associative memory)** 로 저장하는 곳. 나중에 이 DB + Jimmy가 짠 story line을 결합해 **세미나 자료를 다시** 만든다.

> 이 저장소 자체가 이 스터디의 주제(associative memory = key→value로 저장하고 내용으로 회상)를 그대로 실천한다: 각 질문을 **개념 key**로 색인해, 나중에 연상으로 묶어 꺼낸다.

## "안다 4축" (taskops `uncertaintyState`, Fable unknowns 2×2)

| 축 | 뜻 | 해소 메커니즘 |
|---|---|---|
| `known` | 안다는 걸 안다 (known-known) | 이미 앎·검증됨 |
| `known_unknown` | 모른다는 걸 안다 | **decomposition** (구조화) |
| `unknown_known` | 아는데 안 드러남 (암묵지·taste·recognize-when-seen) | **prototype→react** (보고 알아챔) |
| `unknown_unknown` | 모른다는 것도 모른다 | **exploration** (탐색) |

보조 축 — `comprehension`: 답을 받았다고 끝이 아니라 **이해했는지**(shallow vs deep)를 별도로 표시.

## 저장 규약 (매 질문마다)

각 질문 1건 = `qa.jsonl` 한 줄(append-only DB) + `LOG.md`에 사람이 읽을 형태. 스키마:

```json
{
  "id": "Q001",
  "date": "2026-07-14",
  "question": "질문 원문",
  "answer_summary": "답의 핵심(3~5문장)",
  "topic": "Titans | Nested Learning | optimizer | serving | ...",
  "concepts": ["momentum", "surprise", ...],        // 연상 key (색인·클러스터용)
  "axis": {
    "before": "known_unknown",                      // 질문이 나온 축
    "after": "known",                               // 답 이후 상태
    "surfaced": ["unknown_unknown: ...", "unknown_known: ..."],  // 답이 새로 드러낸 것
    "comprehension": "shallow | working | deep"
  },
  "think_about": ["아직 열린 실·후속으로 파볼 것"],
  "storyline_seed": "나중 story line에서 이 조각이 놓일 자리 한 줄",
  "links": ["Q000"]                                 // 연상으로 이어지는 다른 항목
}
```

## 사용

```bash
# 항목 1건 추가(JSON을 인자로) → qa.jsonl append + LOG.md, INDEX.md 재생성
python3 qa-memory/append_qa.py '<entry json>'
# 색인만 재생성
python3 qa-memory/append_qa.py --reindex
```

`INDEX.md` = 연상 회상용 색인: (1) 안다-4축별 클러스터, (2) 개념 key별 클러스터, (3) 열린 "think_about" 모음, (4) storyline seed 모음. story line 짤 때 여기서 꺼낸다.

## 상태
세미나는 이 DB가 충분히 쌓이고 Jimmy가 story line을 준 뒤 **처음부터 다시** 만든다(기존 145슬라이드 판은 참고용). 지금은 **질문 축적 단계**.
