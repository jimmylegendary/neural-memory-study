# Neural Memory / Sleep-Time Compute 연구 패키지

이 저장소의 현재 release는 **2026-08-05 source freeze**를 기준으로 Sleep-Time Compute를 pre-training, wake/test-time learning, sleep-time consolidation, 외부 장기기억, scaling law, system·memory-device 기회까지 연결해 검토한 연구 패키지다. 시작점은 독자의 목적에 따라 다르다.

## 프로젝트별 배포 디렉터리

기존 생성·편집 경로는 재현 빌드를 위해 유지하고, 사람이 열람·전달할 최종 파일은 [`deliverables/`](deliverables/README.md) 아래에 같은 계층으로 분리했다.

| 프로젝트 | 배포 디렉터리 | 구성 |
|---|---|---|
| Neural Memory | [`deliverables/neural-memory/`](deliverables/neural-memory/README.md) | Study 1, 쉬운 설명 12, 의역 10, seminar 5, 분석 3 — 총 31개 |
| Sleep-Time Compute | [`deliverables/sleep-time-compute/`](deliverables/sleep-time-compute/README.md) | Study 10, 쉬운 설명 13, 의역 15, seminar 2, manifest 6 — 총 46개 |

각 디렉터리의 `manifest.json`은 package file과 기존 source path의 대응, byte size, SHA-256, 권리 metadata를 기록한다.

### Neural Memory 빠른 경로

| 목적 | 파일 |
|---|---|
| 통합 학습서 | [`BOOK.pdf`](deliverables/neural-memory/study/BOOK.pdf) |
| 쉬운 설명 12편 | [`easy/`](deliverables/neural-memory/easy/) |
| 핵심 논문 의역 10편 | [`translations/`](deliverables/neural-memory/translations/) |
| Seminar | [`seminar/`](deliverables/neural-memory/seminar/) |
| Attn vs HOPE·효율·multiarch 분석 | [`analysis/`](deliverables/neural-memory/analysis/) |

## 독자별 시작 경로

| 독자 | 먼저 읽을 것 | 다음 경로 |
|---|---|---|
| 학술 검토자 | [학회형 논문 PDF](deliverables/sleep-time-compute/study/SLEEP-TIME-COMPUTE-CONFERENCE-KR.pdf) | [상세 Study PDF](deliverables/sleep-time-compute/study/SLEEP-TIME-COMPUTE-STRATEGIC-STUDY-KR.pdf) → [Appendix](deliverables/sleep-time-compute/study/SLEEP-TIME-COMPUTE-CONFERENCE-APPENDIX-KR.pdf) |
| Training 입문자 | [Training Background PDF](deliverables/sleep-time-compute/study/TRAINING-BACKGROUND-FOR-SLEEP-TIME-COMPUTE-KR.pdf) | [쉬운 설명 통합본](deliverables/sleep-time-compute/easy/SLEEP-TIME-COMPUTE-EASY-COMPANION-KR.pdf) → 상세 Study |
| Seminar 참가자 | [112-slide PPTX](deliverables/sleep-time-compute/seminar/SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pptx) | [발표용 PDF](deliverables/sleep-time-compute/seminar/SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pdf) → [deck 재현·검수](presentation/sleep-time-compute-deep/README.md) |
| 핵심 논문 번역 독자 | [15편 의역 PDF](deliverables/sleep-time-compute/translations/) | [권리 포함 package manifest](deliverables/sleep-time-compute/manifest.json) |
| System·device 연구자 | [System/infra blueprint](research/sleep-time-compute/SYSTEM-INFRA-BLUEPRINT.md) | [Scaling-law agenda](research/sleep-time-compute/SCALING-LAWS-THEORY-AGENDA.md) → [Benchmark blueprint](research/sleep-time-compute/BENCHMARK-EXPERIMENT-BLUEPRINT.md) |
| 재현·QA 검토자 | [전체 산출물 지도](research/sleep-time-compute/DELIVERABLES.md) | [Package manifest](deliverables/sleep-time-compute/manifest.json) → [Final QA](deliverables/sleep-time-compute/manifests/publications/FINAL-QA.json) → [SHA-256](deliverables/sleep-time-compute/SHA256SUMS) |

## 핵심 해석 경계

- Study의 문헌 종합, 비교, 가설, scaling-law 후보는 주장 강도를 구분한다. 사전 등록된 GPU benchmark와 capacity 실험은 **설계**이며 완료된 실험 결과가 아니다.
- 15편 의역본 가운데 PLOS와 eLife의 2편만 `public`으로 분류한다. 나머지 13편과 원문 보존 plate는 `internal-only`이며 외부 배포 대상이 아니다.
- 저자 작성 논문·쉬운 설명·deck도 인용 및 일부 권리 제한 source figure를 포함할 수 있으므로, 조직 외부 공개 전 별도의 법무·출판 검토가 필요하다.
- 공개/내부 경계는 파일명 추정이 아니라 [release manifest generator](research/sleep-time-compute/program/build_release_manifest.py)가 각 translation의 `rights` 값을 읽어 강제한다.

## Release 상태 확인

```bash
python3 research/sleep-time-compute/program/build_release_manifest.py --root . --require-ready
(cd build/publications && sha256sum -c SHA256SUMS)
python3 scripts/package_deliverables.py --root . --output deliverables --check
```

배포 패키지 재생성은 `python3 scripts/package_deliverables.py --root . --output deliverables`로 수행한다. 전체 재현 명령과 예상 출력은 [Sleep-Time Compute deliverables index](research/sleep-time-compute/DELIVERABLES.md)에 고정한다.
