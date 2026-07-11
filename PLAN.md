# PLAN — 실행 계획

목표: 150–200p (작업 목표 ≈190p = Part I 62 / Part II 67 / Part III 52 + 서론·결론) 스터디 페이퍼 + TTT scaling·대규모 학습/서빙 청사진.

## Phase 현황
- **P0 Pre-research** ✅ 2026-07-11 — 6편 deep-read notes, prereq 커리큘럼(B0–B9), veridraft 통합 플랜, 종합 dossier (`dossier/PRE-RESEARCH.md`)
- **P1 1차 집필 (W1 workflow)** ◀ 진행 중 — style/notation foundation → Part I 11개 장 + Part II 6개 장 한국어 초안 병렬 집필 + recon 2종(구현 가용성, 로컬 실험 환경) → 3-way audit (notation / continuity / coverage)
- **P2 보정 라운드** — audit 결과 기반 fix 워크플로 반복, 서론·결론 골격, 장간 cross-ref 확정
- **P3 Part III** — (a) HOPE+Dreaming 사내 A100 runbook (예외 시나리오 포함), (b) 로컬 host 실험 실행, (c) HAT IR 기반 decode 워크로드 modeling/DSE, (d) scaling 분석·법칙·projection, (e) pair-thesis 본문 + player-strategy 분석 집필
- **P4 Veridraft 게이팅** — claims/bundle.json 원장 구축: 논문 요약 claim = P2(vendored PDF@commit warrant), 원저 정량 claim = P1(experiments/results.json@commit + result_refs), readiness 리포트
- **P5 영어 제출판** — `paper-en/` LaTeX 파생 + 추출 논문 후보 재평가 (D1)

## 디렉토리
```
papers/     원문 PDF+txt (vendored, veridraft warrant 대상)
notes/      deep-read JSON ×6, prereq-curriculum, veridraft-tooling, recon 산출물
dossier/    PRE-RESEARCH.md
style/      STYLE-NOTATION.md (통일 표기·용어·장 템플릿) ← W1 foundation
study-kr/   한국어 스터디판 (part1/ part2/ part3/)
audit/      감사 리포트
experiments/ 로컬·HAT 실험 (Part III)
runbook/    사내 A100 실측 가이드 (Part III)
claims/     veridraft bundle.json (P4)
```

## 열린 항목
- HOPE/Sleep 공식 구현 존재 여부 → W1 recon(impl-availability)이 판정, runbook의 구현 전략(공식/비공식/자체 구현) 결정
- 사내 클러스터 소프트웨어 스택(스케줄러, 컨테이너, 네트워크) 정보 필요 → runbook 작성 시 Jimmy 인터뷰
- Titans 후속 "larger models" 논문 출현 감시 (라인이 아직 움직이는 중)
