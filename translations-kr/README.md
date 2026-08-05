# Sleep-Time Compute 핵심 논문 한국어 의역 코퍼스

Sleep-Time Compute, continual learning, 외부 agent memory, memory capacity를 연결하는 핵심 논문 15편의 **구조 보존형 한국어 의역본**이다. 번역자의 분석·평가는 넣지 않고 원문의 주장 강도, 수치, section 순서와 한계를 유지한다.

각 PDF는 두 부분으로 구성된다.

1. 원문 section 순서에 맞춘 한국어 의역 본문
2. equation, table, figure, caption을 직접 역대조할 수 있는 **고정 원문 전체 보존 부록**

따라서 한국어 문장의 해석이 애매할 때 같은 PDF 안에서 원문을 즉시 확인할 수 있다. PNAS 논문(STC-T05)은 공식 NCBI BioC full text와 원 figure 4개를 사용한다.

## 15편 코퍼스

| ID | 논문 | 고정 버전 | 배포 | 출력 PDF |
|---|---|---|---|---|
| STC-T01 | Sleep-time Compute | arXiv:2504.13171v1 | internal-only | `STC-T01-SLEEP-TIME-COMPUTE-KR.pdf` |
| STC-T02 | Language Models Need Sleep | arXiv:2606.03979v2 | internal-only | `STC-T02-LANGUAGE-MODELS-NEED-SLEEP-KR.pdf` |
| STC-T03 | Can a Language Model Learn Facts Continually in Its Weights? | arXiv:2607.11020v2 | internal-only | `STC-T03-CONTINUAL-FACT-WRITES-KR.pdf` |
| STC-T04 | Sleep prevents catastrophic forgetting in SNNs | PLOS Version of Record | public, CC BY 4.0 | `STC-T04-PLOS-SLEEP-SNN-KR.pdf` |
| STC-T05 | Autonomous hippocampus–neocortex interactions | PNAS / PMC9636926 | internal-only, CC BY-NC-ND | `STC-T05-PNAS-HIPPOCAMPUS-NEOCORTEX-SLEEP-KR.pdf` |
| STC-T06 | Dense Associative Memory | arXiv:1606.01164v2 | internal-only | `STC-T06-DENSE-ASSOCIATIVE-MEMORY-KR.pdf` |
| STC-T07 | Deep Generative Replay | arXiv:1705.08690v3 | internal-only | `STC-T07-DEEP-GENERATIVE-REPLAY-KR.pdf` |
| STC-T08 | MemGPT | arXiv:2310.08560v2 | internal-only | `STC-T08-MEMGPT-KR.pdf` |
| STC-T09 | Zep Temporal Knowledge Graph | arXiv:2501.13956v1 | internal-only | `STC-T09-ZEP-TEMPORAL-KG-KR.pdf` |
| STC-T10 | Nested Learning / HOPE | arXiv:2512.24695v1 | internal-only | `STC-T10-NESTED-LEARNING-HOPE-KR.pdf` |
| STC-T11 | Memory Caching | arXiv:2602.24281v1 | internal-only | `STC-T11-MEMORY-CACHING-KR.pdf` |
| STC-T12 | Rate–Distortion Memory Compaction | arXiv:2607.08032v1 | internal-only | `STC-T12-RATE-DISTORTION-MEMORY-COMPACTION-KR.pdf` |
| STC-T13 | Perturbed and Adversarial Dreaming | eLife Version of Record v2 | public, CC BY 4.0 | `STC-T13-PERTURBED-ADVERSARIAL-DREAMING-KR.pdf` |
| STC-T14 | How much do language models memorize? | arXiv:2505.24832v3 | internal-only | `STC-T14-LANGUAGE-MODEL-MEMORIZATION-CAPACITY-KR.pdf` |
| STC-T15 | LoRA as Knowledge Memory | arXiv:2603.01097v5 | internal-only | `STC-T15-LORA-KNOWLEDGE-MEMORY-KR.pdf` |

PDF 위치: [`../build/publications/translations-kr/`](../build/publications/translations-kr/), machine-readable 목록과 hash는 [`build-manifest.json`](../build/publications/translations-kr/build-manifest.json)에 있다.

## 배포 경계

- `public`: STC-T04(PLOS, CC BY 4.0), STC-T13(eLife, CC BY 4.0) 2편.
- `internal-only`: 나머지 13편. 원문 전체 보존 부록을 포함하므로 조직 외부에 배포하지 않는다.
- 개별 파일의 판정 근거는 [`source-rights.json`](stc-core/reports/source-rights.json), release에서 실제 분리된 목록은 [`STC-STUDY-RELEASE-MANIFEST.json`](../build/publications/STC-STUDY-RELEASE-MANIFEST.json)을 따른다.
- 2026-08-05 source freeze 이후 논문 revision이나 license 변경은 이 corpus에 자동 반영되지 않는다.

## 재현과 검수

```bash
python3 translations-kr/stc-core/build_translations.py --all
python3 translations-kr/stc-core/qa_translations.py --all --strict --render-all-pages
python3 translations-kr/stc-core/audit_translations.py
(cd build/publications/translations-kr && sha256sum -c SHA256SUMS)
```

최종 검수 결과:

- 15/15 source hash와 version 고정
- 15/15 구조 보존 PASS
- 75/75 load-bearing passage 역대조 PASS
- 한국어 본문 전 페이지 overflow 0
- 579쪽, 두 번의 전체 build에서 SHA-256 byte 일치
- 공개 가능 2편, internal-only 13편

세부 보고서: [`structural-parity.json`](stc-core/reports/structural-parity.json), [`source-rights.json`](stc-core/reports/source-rights.json), [`reverse-check.md`](stc-core/reports/reverse-check.md), [`corpus-audit.json`](stc-core/reports/corpus-audit.json).

## 기존 Neural Memory 의역본

기존 Titans, MIRAS, ATLAS, TNT, NSTM과 초기 의역본은 provenance로 이 directory 최상위에 그대로 보존한다. 새 15편 corpus는 `stc-core/`의 고정 manifest와 QA pipeline을 사용한다.
