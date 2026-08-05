# Neural Memory / Sleep-Time Compute 연구 패키지

이 저장소의 현재 release는 **2026-08-05 source freeze**를 기준으로 Sleep-Time Compute를 pre-training, wake/test-time learning, sleep-time consolidation, 외부 장기기억, scaling law, system·memory-device 기회까지 연결해 검토한 연구 패키지다. 시작점은 독자의 목적에 따라 다르다.

## 독자별 시작 경로

| 독자 | 먼저 읽을 것 | 다음 경로 |
|---|---|---|
| 학술 검토자 | [학회형 논문 PDF](build/publications/SLEEP-TIME-COMPUTE-CONFERENCE-KR.pdf) | [상세 Study PDF](build/publications/SLEEP-TIME-COMPUTE-STRATEGIC-STUDY-KR.pdf) → [Appendix](build/publications/SLEEP-TIME-COMPUTE-CONFERENCE-APPENDIX-KR.pdf) |
| Training 입문자 | [Training Background PDF](build/publications/TRAINING-BACKGROUND-FOR-SLEEP-TIME-COMPUTE-KR.pdf) | [쉬운 설명 통합본](build/publications/SLEEP-TIME-COMPUTE-EASY-COMPANION-KR.pdf) → 상세 Study |
| Seminar 참가자 | [112-slide PPTX](build/publications/SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pptx) | [발표용 PDF](build/publications/SLEEP-TIME-COMPUTE-DEEP-SEMINAR-112SLIDES-KR.pdf) → [deck 재현·검수](presentation/sleep-time-compute-deep/README.md) |
| 핵심 논문 번역 독자 | [15편 의역 코퍼스 안내](translations-kr/README.md) | [번역 build manifest](build/publications/translations-kr/build-manifest.json) |
| System·device 연구자 | [System/infra blueprint](research/sleep-time-compute/SYSTEM-INFRA-BLUEPRINT.md) | [Scaling-law agenda](research/sleep-time-compute/SCALING-LAWS-THEORY-AGENDA.md) → [Benchmark blueprint](research/sleep-time-compute/BENCHMARK-EXPERIMENT-BLUEPRINT.md) |
| 재현·QA 검토자 | [전체 산출물 지도](research/sleep-time-compute/DELIVERABLES.md) | [Release manifest](build/publications/STC-STUDY-RELEASE-MANIFEST.json) → [Final QA](build/publications/FINAL-QA.json) → [SHA-256](build/publications/SHA256SUMS) |

## 핵심 해석 경계

- Study의 문헌 종합, 비교, 가설, scaling-law 후보는 주장 강도를 구분한다. 사전 등록된 GPU benchmark와 capacity 실험은 **설계**이며 완료된 실험 결과가 아니다.
- 15편 의역본 가운데 PLOS와 eLife의 2편만 `public`으로 분류한다. 나머지 13편과 원문 보존 plate는 `internal-only`이며 외부 배포 대상이 아니다.
- 저자 작성 논문·쉬운 설명·deck도 인용 및 일부 권리 제한 source figure를 포함할 수 있으므로, 조직 외부 공개 전 별도의 법무·출판 검토가 필요하다.
- 공개/내부 경계는 파일명 추정이 아니라 [release manifest generator](research/sleep-time-compute/program/build_release_manifest.py)가 각 translation의 `rights` 값을 읽어 강제한다.

## Release 상태 확인

```bash
python3 research/sleep-time-compute/program/build_release_manifest.py --root . --require-ready
sha256sum -c build/publications/SHA256SUMS
```

전체 재현 명령과 예상 출력은 [Sleep-Time Compute deliverables index](research/sleep-time-compute/DELIVERABLES.md)에 고정한다.
