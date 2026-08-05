# PLAN — 실행 계획

목표: 150–200p (작업 목표 ≈190p = Part I 62 / Part II 67 / Part III 52 + 서론·결론) 스터디 페이퍼 + TTT scaling·대규모 학습/서빙 청사진.

## Phase 현황
- **P0 Pre-research** ✅ 2026-07-11 — 6편 deep-read notes, prereq 커리큘럼(B0–B9), veridraft 통합 플랜, 종합 dossier (`dossier/PRE-RESEARCH.md`)
- **P1 1차 집필** ✅ — style/notation foundation → Part I 11장 + Part II 6장 한국어 초안 + recon 2종 → 3-way audit
- **P2 보정** ✅ — 3축 감사 fix + **codex(gpt-5.6-sol) cross-model 검증** (ch10 AI 누락항·ch14 Thm1 등 포착), 잔여 closure
- **P3 Part III** ✅ — 실험 E1–E4(HAT DSE + CPU microbench + scaling digitize-fit) 실행 + ch18–25 집필 + 심화 + HOPE A100 runbook(`runbook/`); 그림 exp-a~f 렌더
- **P4 Veridraft 게이팅 + 조립** ✅ — `claims/bundle.json` 30 claim gate PASS(P2 18 + P1 12); front/back matter; **309→310pp PDF 빌드** (`build/BOOK.pdf`, `build/template.tex`)
- **P5 한국어판 마무리 QA** ✅ 2026-07-12 — cross-ref dangling 0, 인용 미해결 0, 그림 라벨 겹침 재생성(exp-b/exp-f), 시각 결함 5건 수정; 최종 빌드 310pp/미해결글리프 0/overfull 0. **사용자 결정: 309(310)pp depth 유지, 한국어판 우선.**
- **P6 영어 제출판** ⬜ (미착수, D1) — `paper-en/` LaTeX 파생 + 추출 논문 후보 재평가
- **P7 Sleep-Time Compute v1** ❌ 2026-08-06 폐기 — 전략 분석 리포트로 완성(70pp Study + 학회형
  12pp + Training Background 57pp + easy 62pp + 112 slides)했으나, 논문별 심층 해부·통일 표기·
  bridge 사슬·3층 구분·정직성 caveat이 없어 이 저장소의 모노그래프 기준에 미달. 전량 폐기.
  git history에서만 참조한다.

---

## Sleep-Time Compute v2 — Neural Memory 방식 전면 적용

**결정 2026-08-06 (Jimmy)**: 집필 주체 = Claude 전량 / v1 = git rm 완전 삭제 /
규모 = NM 동급 풀세트(본문 300pp+, Part II 논문장 12–15편, easy 10+, 의역 15편 유지·보강,
seminar 100+ slides) / 로컬 실험 트랙 포함.

| Phase | 내용 | 상태 |
|---|---|---|
| S0 Pre-research | 소스 재확보(2605.26099·2504.13171 신규 vendored), 논문별 deep-read notes, 계보·bridge 사슬 확정, training 사전지식 커리큘럼, `dossier/STC-PRE-RESEARCH.md`, 척추 논지 후보 랭킹 | 🟡 |
| S1 STYLE-NOTATION | `style/STC-STYLE-NOTATION.md` — 통일 표기·master equation·Rosetta 사전·장 템플릿·분량 가이드·정직성 caveat 표 | ⬜ |
| S2 Part I | training 무경험 독자용 배경 (pre-training/fine-tuning/LoRA/distillation/RL/synthetic data/replay/forgetting/CLS/capacity/외부기억) — 본문 통합, 장마다 worked micro-example + systems bridge | ⬜ |
| S3 Part II | 논문 12–15편 연대순 장, 8절 고정 템플릿, bridge 사슬 | ⬜ |
| S4 실험 + Part III | 로컬 실행 실험 → `results.json` warrant + 척추 논지·promising 판정·대안 비교·scaling law·system/device 기회 | ⬜ |
| S5 게이트·빌드 | veridraft claim bundle PASS + pandoc/lualatex + `build/overflow_gate.py` PASS + source figure 임베드 | ⬜ |
| S6 파생물 | easy booklet 10+ / 의역 보강 + 신규 / seminar deck (렌더 하네스 재사용, 내용 신규) | ⬜ |

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
