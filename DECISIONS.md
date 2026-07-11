# DECISIONS — Neural-Memory/TTT Study Paper

인터뷰 확정일: 2026-07-11 (Jimmy)

## D1. 산출물 전략 = 단일 모노그래프 우선
150–200p 모노그래프 완결에 전력. 학회용 추출 논문(예: serving 실측 → MLSys/ISCA 계열, scaling → NeurIPS/ICLR 계열)은 **완성 후 별도 판단**. arXiv 공개 여부/시점도 완성 후 결정.

## D2. 언어 = 한국어 스터디판(1차) + 영어 제출판(2차)
한국어 스터디판을 먼저 완성(학습 목적 최우선). 영어 제출판은 스터디판에서 파생. 기술 용어는 원어(영어) 유지, 수식은 LaTeX.
- 스터디판: `study-kr/` Markdown (장별 파일, pandoc으로 통합 빌드 가능)
- 제출판: 추후 `paper-en/` LaTeX

## D3. 실험 = 3-트랙
1. **사내 A100 실측 (runbook 산출)**: HOPE model + Dreaming(Sleep 논문) 기능 실측이 목표. 이 호스트에서는 불가 → 사내 클러스터(fast-path: A100 80G ×4 ×2 node = 8장, 최대 ~32장 multi-node)에서 수행할 수 있도록 **runbook 스타일 상세 가이드**(단계별 명령, 예상 예외상황·트러블슈팅 포함)를 산출물로 만든다. 구현 가용성(공식/비공식 코드) recon이 선행 조건.
2. **로컬 host 실험**: 이 호스트 환경에서 가능한 실험을 설계·직접 수행.
3. **HAT 툴 modeling/DSE**: 다른 세션의 hatir / hat-schema / hat-vllm-mock 시뮬레이션·비용모델 툴로 TTT decode 워크로드 modeling 실험 및 design-space exploration.

## D4. Part III 척추 = 워크로드 분할 페어 논지
"decode/serving state 관리 = memory-centric 기회 (per-token state read-modify-write, unshared·write-heavy), training/prefill = 여전히 accelerator 영역 (fused chunk kernel, grouped-GEMM)" — 이 **쌍(pair)** 을 Part III의 척추로 세우고 scaling law·player 분석을 그 아래 배치. memory-centric 논증은 정직 판정에서 진짜로 판명된 두 지점(decode state RMW 트래픽, update-frequency↔메모리 계층 배치)에만 사용.

## 상수
- 대상 독자: transformer inference / efficient-transformer를 아는 AI system infra·architecture exploration 엔지니어, training 무경험.
- 품질 우선(토큰/시간 비용 무시), veridraft v0.1.0 게이트 사용.
- memory-centric 억지 연결 금지 (D4의 두 지점 외 사용 금지).
- 모든 변경은 commit + push (비공개 repo `jimmylegendary/neural-memory-study`).
