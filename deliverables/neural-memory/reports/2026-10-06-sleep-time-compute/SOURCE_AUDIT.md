# Source audit — 2026-10-06

이 문서는 이전 Markdown 인덱스의 정정과 PDF의 근거 사용 범위를 기록합니다.

## 식별자 정정

| 자료 | 이전 인덱스 | 확인 후 PDF에서 사용 |
|---|---|---|
| Hindsight Memory-PRM | 2608.31254 | https://arxiv.org/abs/2608.29605 |
| KV-streams | 2609.35893 | https://arxiv.org/abs/2609.35750 |
| HBFSim | 2609.09902 | https://arxiv.org/abs/2609.09800 |
| Mem++ | 2609.39246 | https://arxiv.org/abs/2610.02002 |
| Schema | 2609.39943 | https://arxiv.org/abs/2609.39140 |
| Beyond Memory / PoS | 2609.40334 | https://arxiv.org/abs/2610.01415 |
| ActiveSaddler | 2609.40515 | https://arxiv.org/abs/2610.00906 |
| EfficientAgent | 2609.34901 | https://arxiv.org/abs/2609.33762 |

## 주요 해석 정정

- **CMS update schedule와 read schedule은 다릅니다.** Nested Learning의 CMS를 모든 block이 forward에 참여하는 경우와 구분하지 않고 slow CMS=HBF라고 일반화하지 않습니다.
- canonical state는 진실 또는 불변 데이터의 동의어가 아닙니다.
- HBF 수명 비용은 commit 횟수가 아니라 physical program bytes, refresh, GC, mapping, 온도에 의존합니다.
- HBFSim의 20.8×는 simulation 실행 속도 개선입니다. LLM inference speedup으로 해석하지 않습니다.
- MemCodex의 10.1%는 상대 향상이며 10.1%p로 쓰지 않습니다. 해당 논문은 work in progress로 분류합니다.
- adapter generation rollback과 파라미터에서 특정 사실만 제거하는 unlearning은 다릅니다.
- RPMem inference의 recurrent state update는 session별 gradient 학습과 구별됩니다.
- 논문별 model·reader·harness가 다른 성능을 하나의 leaderboard로 합치지 않습니다.
- PlanFence는 v1을 고정하여 인용합니다. 최신 버전이 다른 제목·내용으로 갱신될 수 있습니다.

## 검토 깊이

sources/references.json의 review_scope 필드가 열람 범위를 정의합니다. 초록/서지 검토를 full-paper review로 부풀리지 않았습니다. 일부 핵심 논문은 본문, 식, figure를 함께 확인했습니다. 어떤 실험도 이 보고서에서 독립 재현하지 않았습니다.

이전 기록에서 재검증이 부족한 추가 자료는 PDF 38–39쪽의 별도 추적 목록에 보존되어 있습니다. 55개 정식 출처 대장에 자동 편입되지 않습니다.

## 그림과 데이터

본문 31개 figure는 원본 SVG로 다시 설계한 도식 또는 데이터 재시각화입니다. 원문 그림을 복사하여 제공하지 않습니다. 각 그림의 설명·출처·개념 재구성 여부는 sources/media_manifest.json에 있습니다. 저작권과 원문의 권리는 각 원저자에게 있습니다.

정정 범위는 이 PDF와 source package입니다. GitHub의 이전 Markdown 파일이 자동으로 교체된 것은 아닙니다.
