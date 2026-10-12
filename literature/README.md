# Neural Memory / Sleep-time Compute 논문 DB

초기 조사부터 2026-10-12까지 회수한 논문 기록을, 기존 저장소의 선행 연구와 연결한 관계형 문헌 대장이다. 보고서 참고문헌만 세거나 주간 요약을 이어 붙인 목록이 아니다. 같은 논문의 버전은 하나의 논문에 연결하고, 미검증·오류 기록도 원래 값으로 보존한다.

## 바로 보기

| 목적 | 파일 |
|---|---|
| 발행일순 목록 | [CHRONOLOGICAL.md](CHRONOLOGICAL.md) |
| 검색·필터 화면 | [index.html](index.html) — 내려받아 브라우저에서 열기 |
| 실제 관계형 DB | [data/papers.sqlite](data/papers.sqlite) |
| 다른 도구로 가져오기 | [CSV](data/papers.csv) · [JSONL](data/papers.jsonl) |
| 입력→논문 전수 대조 | [coverage.json](coverage.json) |
| 미검증·충돌 | [UNRESOLVED.md](UNRESOLVED.md) |
| 원래 기록 | [observations.json](data/observations.json) · [sources.json](data/sources.json) |
| 정정 내역 | [식별자 정정](data/corrections.json) · [날짜 불일치](data/field_disagreements.json) |
| 기존 원문 파일 보관 현황 | [assets.json](data/assets.json) |
| 데이터 구조 | [schema.sql](data/schema.sql) |

현재 건수는 위 `coverage.json`과 생성된 발행일순 대장이 정본이다. README에 수를 고정해 새 자료 추가 후 오래된 숫자가 남는 것을 피한다.

## 수집 범위

초기 입력은 다음 출처를 회수해 구성했다.

- 2026-08-11 최초 회수 PDF의 참고문헌 6건과 2026-08-16 통합 보고서의 공개 서지 43건.
- 2026-08-17, 08-24, 08-31, 09-07, 09-14, 09-21, 09-28, 10-05, 10-12의 주간 요약에서 개별 논문으로 다룬 105건.
- 2026-10-06 PDF 편집 원본의 참고문헌 55건. 논문과 공식 문서·커뮤니티를 구분한다.
- 회수된 08-07 HOPE/HBF 선행 참조 1건 및 이전 Markdown의 잘못된 식별자도 포함한 대조 기록 7건.
- 저장소 `research/ttt-efficient-corpus.json`의 기존 TTT 대장 전체.
- 저장소의 명시적인 bibliography/BibTeX, 작성 문서에서 직접 인용한 arXiv 식별자, `papers/`의 기존 원문 PDF·텍스트.

초기 task 입력은 **217개 관측 기록**이며, 이것은 고유 논문 217편이라는 뜻이 아니다. 중복을 합친 고유 논문 수는 빌드 결과에 따로 표시한다. 각 원본 관측이 canonical 논문에 연결됐는지 검사하여 빠진 행이 있으면 빌드를 실패시킨다.

08-11은 **회수된 가장 이른 task 문서 날짜**다. 08-07은 관련 선행 대화의 참조 날짜다. 이것만으로 자동화 task의 실제 생성일이나 모든 과거 대화에 접근했다고 주장하지 않는다. 원문 논문의 bibliography에 인용됐다는 이유만으로 다시 수천 편을 재귀적으로 수집하지 않는다. 개인 프로젝트 평가 본문은 공개 DB에 재배포하지 않는다.

## 날짜와 검증 규칙

`publication_date`는 확인된 arXiv v1 또는 발행처 공개일을 우선 사용한다. 최초 공개일과 수정일 `updated_at`을 분리한다. arXiv v1보다 이른 OpenReview/학회 공개가 확인되지 않으면 arXiv 날짜를 **최초의 모든 공개일**이라고 단정하지 않고 `date_basis=arxiv_v1`로 표시한다.

원문 날짜가 확인되지 않은 항목은 과거 보고서의 날짜를 `(기록값)` 및 `historical_report_unverified`로 표시한다. 연도·월까지만 아는 경우 그대로 남긴다. arXiv ID의 숫자에서 발행일을 만들어내거나 임의의 1일을 실제 날짜로 기록하지 않는다. 날짜가 없으면 목록 끝에 둔다.

| 검증 상태 | 뜻 |
|---|---|
| `primary_metadata_verified` | arXiv의 제목·식별자·서지 메타데이터를 확인함 |
| `publisher_metadata_verified` | 원 발행처 서지 확인 |
| `metadata_pending` | 식별자는 기록됐지만 1차 서지 재확인이 아직 부족함 |
| `identifier_pending` | 정확한 arXiv/DOI가 미확정 |
| `identity_conflict` | 기록 제목과 조회된 식별자의 제목이 맞지 않음 |
| `non_paper` | 블로그·문서·커뮤니티 등 별도 자료 |

**메타데이터 확인은 논문 전체를 재독하거나 성능을 재현했다는 뜻이 아니다.** 과거 요약의 수치·평가는 `observations.note`에 출처 주장으로 남는다. 미검증 논문은 누락하거나 검증된 연구처럼 승격하지 않는다. 네트워크 오류는 논문 부재의 증명이 아니다.

## 원본과 정정

`publications`는 현재 선택한 서지이며 `observations`는 회수한 원래 제목·별칭·날짜·식별자·출처를 보존한다. 제목이나 ID를 바로잡아도 관측값을 덮어쓰지 않는다. 같은 약어를 사용한 다른 논문을 합치지 않도록 alias 정정은 task 관측에 한정한다. arXiv 버전은 별도 논문으로 중복 집계하지 않는다.

원문 PDF는 기존 저장소에 있는 것만 경로·크기·SHA-256으로 연결한다. 이번 DB 작업이 새 PDF 전부를 다운로드·재배포했다는 뜻은 아니다. 파일 이름과 arXiv ID 연결은 원문 내용의 일치 검증과 별개로 기록한다. 기존 저작권·배포 제한을 변경하지 않는다.

## SQLite 예시

```sql
-- 전체 논문 발행일순
SELECT publication_date, title, arxiv_id, verification_status
FROM chronological;

-- 이 task에서 실제 회수한 항목만
SELECT publication_date, title, updated_at, date_basis
FROM task_papers;

-- HBF 키워드 검색 (FTS5)
SELECT p.publication_date, p.title, p.url
FROM paper_fts f JOIN publications p ON p.id=f.publication_id
WHERE paper_fts MATCH 'HBF'
ORDER BY p.publication_date;

-- 원래 잘못 적힌 ID와 현재 정정값
SELECT o.reported_title, c.reported_arxiv, c.resolved_arxiv, o.source_id
FROM corrections c JOIN observations o ON o.id=c.observation_id;

-- 최신 회차 17개가 모두 연결됐는지
SELECT COUNT(*) FROM observations WHERE source_id='weekly-2026-10-12';
```

## 재빌드와 갱신

Python 3.11 이상, 표준 라이브러리만 사용한다. 원본 입력 수정 후 다음을 실행한다.

```bash
python3 -m unittest discover -s literature/tests -v
python3 literature/tools/build_catalog.py --online --deadline 780 --as-of 2026-10-12
python3 literature/tools/check_catalog.py
```

`--online`을 빼면 저장된 메타데이터만 사용한다. 온라인 조회는 제한된 시간·요청 간격으로 수행하며 실패 로그를 `coverage.json`에 남긴다. 출력은 SQLite, CSV, JSONL, 발행일순 Markdown, HTML, 누락 대조와 체크섬이다. DB는 임시 파일 검증 뒤 교체한다.

GitHub Actions는 `literature/BUILD_REQUEST` 변경 또는 수동 실행 시 같은 검사를 수행하고 결과를 해당 브랜치에 commit/push한다. 자동 반복 스케줄은 만들지 않았다. 새 회차는 출처 ID와 원래 서지값을 입력에 추가하고, `source_manifest.json`의 예상 개수를 맞춘 뒤 같은 빌드로 갱신한다.

`seed_task.py`와 `seed_metadata.py`는 초기 복구용이다. 이후 편집한 TSV·수동 정정을 덮어쓰지 않도록 초기화 단계를 일상 갱신에 무조건 다시 실행하지 않는다. 기존 모노그래프와 PDF 보고서는 이 DB로 대체하지 않는다.
