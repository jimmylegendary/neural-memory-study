# Neural Memory / Sleep-Time Compute 연구 패키지

이 저장소는 두 연구 프로그램을 담는다.

| 프로그램 | 상태 | 시작점 |
|---|---|---|
| **Neural Memory / TTT** | ✅ 완료 (327pp 모노그래프 + 파생 산출물 31개) | [`deliverables/neural-memory/`](deliverables/neural-memory/README.md) |
| **Sleep-Time Compute** | 🔄 **v2 재구축 중** (v1 산출물은 2026-08-06 폐기) | 아래 §Sleep-Time Compute v2 |

## Neural Memory — 빠른 경로

| 목적 | 파일 |
|---|---|
| 통합 학습서 (327pp) | [`BOOK.pdf`](deliverables/neural-memory/study/BOOK.pdf) |
| 쉬운 설명 12편 | [`easy/`](deliverables/neural-memory/easy/) |
| 핵심 논문 의역 10편 | [`translations/`](deliverables/neural-memory/translations/) |
| Seminar (145 slides) | [`seminar/`](deliverables/neural-memory/seminar/) |
| Attn vs HOPE·효율·multiarch 분석 | [`analysis/`](deliverables/neural-memory/analysis/) |

집필 기준의 단일 SoT는 [`style/STYLE-NOTATION.md`](style/STYLE-NOTATION.md), 빌드·QA 방법은
[`build/METHOD.md`](build/METHOD.md), 결정 이력은 [`DECISIONS.md`](DECISIONS.md)다.

## Sleep-Time Compute v2

v1 산출물(상세 Study·학회형 논문·Training Background·easy companion·112-slide deck·release
manifest)은 **2026-08-06에 전량 폐기**했다. 폐기 사유는 분량이 아니라 구조다 — v1은 전략 분석
리포트였고, 논문별 심층 해부·통일 표기·bridge 사슬·3층 구분(주장/해설/평가)·정직성 caveat
같은 Neural Memory 모노그래프의 논지 전개 장치를 갖추지 않았다. v2는 Neural Memory의 방식을
그대로 적용해 처음부터 다시 쓴다. v1은 git history에서만 참조한다.

폐기 대상에서 제외해 유지한 것:

- `papers/` — 1차 소스 원문 PDF·텍스트 (v2에서 그대로 warrant로 사용)
- `translations-kr/stc-core/` — 의역 15편 + 원문 plate (v2에서 유지·보강)
- `presentation/sleep-time-compute-deep/src/` — deck 렌더 하네스 (디자인은 유지, 내용만 신규)
- `build/` — pandoc·lualatex 템플릿과 `overflow_gate.py` 픽셀 게이트

진행 계획은 [`PLAN.md`](PLAN.md) §Sleep-Time Compute v2를 따른다.

## 핵심 해석 경계

- 의역본 15편 가운데 PLOS와 eLife 2편만 `public`이다. 나머지 13편과 원문 보존 plate는
  `internal-only`이며 외부 배포 대상이 아니다.
- 저자 작성 문서도 인용 및 권리 제한 source figure를 포함하므로, 조직 외부 공개 전 별도의
  법무·출판 검토가 필요하다.

## 배포 패키지

```bash
python3 scripts/package_deliverables.py --root . --output deliverables          # 재생성
python3 scripts/package_deliverables.py --root . --output deliverables --check  # 검증
```
