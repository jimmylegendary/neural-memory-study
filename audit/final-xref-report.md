# Final cross-reference integrity report

**감사 유형**: 통합 수준 cross-reference 무결성 (per-chapter 리뷰가 못 잡는 결함 전용)
**대상**: study-kr 25개 장(ch01–25) + front/back matter
**일자**: 2026-07-12
**방법**: 전 장 grep으로 내부 참조 전량 수집 → 정의 대상(절 헤더·그림 임베드·표 캡션·식 tag) 집합과 대조 → dangling·mismatch·중복 탐지. 스크립트: `scratchpad/xref{,2,3}.py`.

## 결론

**genuine dangling = 0, mismatch = 0, 중복 번호 = 0, Edit 수정 = 0.**
309pp / 25장 전체에서 깨진 내부 참조를 하나도 발견하지 못했다. 자동 grep이 1차로 뱉은 후보는 전수 분류 결과 모두 (a) 원 논문 절 인용, (b) STYLE-NOTATION 문서 참조, (c) 표 헤더 단어, (d) master 수식(STYLE 정의) — 즉 정본 체계상 정상 참조였다.

## 참조 그래프 요약

| 유형 | 정의된 대상 | 수집된 참조 | 해소 | dangling | 중복 |
|---|---|---|---|---|---|
| 장 (N장) | ch01–25 (25개, 전부 존재) | 전량 in-range | 100% | 0 | — |
| 절 (§X.Y / X.Y절) | 266개 헤더 (## X.Y, ### X.Y.Z) | book-internal 448 | 100% | 0 | 0 |
| 그림 (그림 X-Y) | 15개 (18-1 … 25-1, 전부 Part III) | 전량 | 100% | 0 | 0 |
| 표 (표 X-Y) | 77개 캡션 (ch01–24) | 전량 | 100% | 0 | 0 |
| 식 (식 (X-Y)/(MX)/(X-Y)) | 101 numeric tag + 6 master(M,M1–M5) | 전량 | 100% | 0 | 0 |

### 장 참조
- 모든 `N장`·`→N장`·`(→ N장)` 참조가 1–25 범위. 범위 밖(0장, 26장+) 없음.
- **concept → chapter 포인터를 concept ledger(STYLE §2.4) 소유권과 대조**: retention gate→13장, delta rule→5장, associative memory→5장, FTRL→3장, update frequency→16장, wake/sleep→17장, catastrophic forgetting→11장, FWP→6장, TTT/dual form→8장, Omega rule/surprise-window→14장, periodic reset→15장, W_init→12·15장, SSD capacity→5장, chunkwise/semantic-hyperparam→9장, Titans→12장 — **전부 올바른 소유장을 가리킴**. 오배치 포인터 없음.

### 절 참조
- book-internal §X.Y 448건 전부 실제 절 헤더로 해소.
- 자동 후보로 걸린 §X.Y 중 **305건은 원 논문 절 인용**(`[TNT §4.1.1]`, `[Mamba §3.3.2]`, `[Titans §5.9]`, `[Gu & Dao 2023 …, §3.5.1]`, `원문은 §5.9` 등) — STYLE §2.5의 `[Paper §X.Y]` 규약대로이며 book 절 참조가 아님.
- `§1.7.1`–`§1.7.6`은 **STYLE-NOTATION 표기 대응표**를 가리키는 관례 참조(ch12–17의 표 캡션 "…§1.7.x 사본/기준", ch09:137 "전체 대응표는 12장 §1.7.1"). 6개 장의 매핑(§1.7.1↔Titans/ch12 … §1.7.6↔Sleep/ch17) 모두 정합. 정본 house-style이며 결함 아님.

### 그림 참조
- 그림은 Part III(ch18–25)에만 정의·존재. Part I/II에서 그림을 참조하는 곳 없음(정합).
- cross-chapter: `그림 18-1`을 ch20·ch23이 참조(정의는 ch18) — KV/TTT crossover 그림, 내용 맞물림.
- 정의된 15개 그림 모두 최소 1회 참조됨(orphan 그림 없음). 중복 번호 없음.
- 임베드↔exp-*.png 매핑 확인: 18-1=exp-a, 19-2/25-1=exp-f, 20-1/21-1/22-1/19-3=exp-c, 20-2/24-1=exp-b, 20-3/24-2=exp-e, 23-1/24-3=exp-d, 22-2/19-1=exp-a. (여러 장이 같은 실험 그림을 각자 임베드하는 것은 의도된 재사용.)

### 표 참조
- 77개 표 캡션 모두 고유 번호(중복 0). 모든 `표 X-Y` 참조가 캡션으로 해소.
- cross-chapter 후방 참조: ch8→표6-2, ch9→표7-3·표8-1, ch10→표1-1·표9-2, ch12→표10-2, ch24→표18-1 — 전부 이전 장 대상(정합 방향).

### 식 참조
- numeric `\tag{X-Y}` 101개 고유. master 수식 (M)(M1–M5)은 STYLE-NOTATION §0/§1.4가 정의(장 본문 아님); ch01=\tag{M}, ch09=\tag{M4} 재정의 존재.
- `식 (X-Y)`, `(MX)`, 괄호형 `(X-Y)` 참조 전량 해소. dangling 0, 중복 tag 0.
- 자동 후보 `식 (요지)`(ch02:106)는 표의 열 헤더 단어이지 식 참조 아님.

## 판단 필요 항목 (danglers)

없음. 단, 아래 1건은 **결함은 아니나 사람이 관례로 재확인**해 두면 좋은 사항이다:

- **`§1.7.x` = STYLE-NOTATION 표기 대응표 참조 관례**: ch09:137 "전체 대응표는 12장 §1.7.1." 및 ch14/ch15 표 캡션의 "(§1.7.3/§1.7.4 기준)"은 book 장의 절 번호가 아니라 STYLE 문서의 절을 가리킨다. 엄격한 독자가 ch12에서 "§1.7.1" 헤더를 찾으면 없다(대신 표 12-1). ch12–17 전반에서 일관된 house-style이므로 그대로 두는 것이 옳다고 판단했으나, 최종본에서 이 관례를 유지할지(예: "STYLE §1.7.1"로 명시 병기)만 확인 권장.

## 수정 내역

없음(genuine 오류 0건이므로 어떤 장 파일도 Edit하지 않음).
