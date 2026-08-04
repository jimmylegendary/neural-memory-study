# Faithful Korean Translation Candidate Set

- target: 12–15편
- purpose: Study Paper의 load-bearing mechanism과 반대 근거를 독자가 원문 구조 그대로 검증할 수 있게 한다.
- translation rule: section/equation/table/figure/caption/reference numbering을 보존하고 번역자의 해설을 본문에 삽입하지 않는다.
- selection is provisional until source version, full-text access, and figure redistribution rights are frozen.

## 1. Selection principles

최종 corpus는 한 lineage의 유사 논문만 모으지 않는다. 다음 stratum을 모두 포함한다.

| Stratum | Minimum | Why required |
|---|---:|---|
| direct STC / explicit sleep lifecycle | 2 | 연구 대상의 직접 정의와 구현 |
| biological/computational precursor | 2 | replay·NREM/REM analogy의 실제 근거와 한계 |
| continual-learning strongest baseline | 2 | STC가 해결한다고 주장하는 forgetting의 경쟁법 |
| external/agent memory | 2 | parametric sleep의 가장 강한 대안 |
| TTT/neural/latent memory bridge | 2 | wake learning과 sleep consolidation의 경계 |
| capacity/negative result | 2 | promisingness를 낮출 수 있는 근거 |
| systems/governance bridge | 1 | deployable lifecycle와 device requirement |

한 논문이 여러 stratum을 충족할 수 있으나, final 12편 안에서 최소 6개 independent research lineage를 확보한다.

## 2. Existing translation assets

| Source | Existing Korean asset | Structural status | Migration decision |
|---|---|---|---|
| Titans, arXiv 2501.00663 | `translations-kr/2501.00663-titans.md/.pdf` | substantial full translation | migrate and run equation/table/figure parity QA |
| MIRAS, arXiv 2504.13173 | `translations-kr/2504.13173-miras.md/.pdf` | substantial full translation | migrate; verify current ICLR version |
| ATLAS, arXiv 2505.23735 | `translations-kr/2505.23735-atlas.md/.pdf` | substantial full translation | migrate; verify ICML final changes |
| TNT, arXiv 2511.07343 | `translations-kr/2511.07343-tnt.md/.pdf` | translation available | migrate after status/version audit |
| Nested Learning / HOPE, arXiv 2512.24695 | `translations-kr/2512.24695-nested-learning-FULL.md/.pdf` | full variant available | migrate full variant and preserve appendix |
| Memory Caching, arXiv 2602.24281 | `translations-kr/2602.24281-memory-caching.md/.pdf` | translation available | migrate after latest-version comparison |
| Language Models Need Sleep, arXiv 2606.03979 | `translations-kr/2606.03979-sleep-FULL.md/.pdf` | full v1-oriented asset | compare v2, update only source-faithful deltas |
| Online Neural Space-Time Memory, arXiv 2607.15271 | `translations-kr/2607.15271-nstm.md/.pdf` | translation available | migrate after metadata and figure QA |

기존 8편은 자동 선정되지 않는다. Google neural-memory lineage가 과대표집되므로 final set에서는 일부를 Study reference translation으로 유지하되 core 12–15 corpus의 lineage balance를 별도로 맞춘다.

## 3. Candidate ranking

| Rank | Paper | Role | Strength | Risk / rights issue | Provisional decision |
|---:|---|---|---|---|---|
| 1 | Sleep-Time Compute: Beyond Inference Scaling at Test-Time, 2504.13171 | direct STC + agent lifecycle | defining paper, directly answers scope | preprint; figure license must be checked | required |
| 2 | Language Models Need Sleep, 2606.03979 v2 | explicit parametric sleep | replay/dreaming and Google lineage | recent status; version delta | required |
| 3 | Can a Language Model Learn Facts Continually in Its Weights?, 2607.11020 v2 | central negative result | sequential LoRA/full-FT and composition | single-author preprint, limited task family | required |
| 4 | Sleep Prevents Catastrophic Forgetting in SNNs, PLOS 2022 | blog claim + peer-reviewed precursor | exact PLOS target, two-task mechanism | narrow SNN scope | required |
| 5 | PNAS hippocampus–neocortex autonomous consolidation, 2022 | computational NREM/REM/CLS | autonomous internal replay dynamics | synthetic categories, biology analogy boundary | high |
| 6 | Perturbed and Adversarial Dreaming, eLife 2022 | wake/NREM/REM factorization | strong ablation and open code | not continual task sequence | high |
| 7 | Deep Generative Dual Memory Network, 1606.01164 | explicit wake/sleep precursor | short/long memory and sleep trigger | older generative model, venue status | high |
| 8 | Continual Learning with Deep Generative Replay | replay baseline | foundational generated replay | not LLM/agent; figure rights | high |
| 9 | Progress & Compress | periodic knowledge transfer baseline | strong direct alternative to sleep consolidation | older task-boundary setting | medium-high |
| 10 | MemGPT: Towards LLMs as Operating Systems | external-memory system | canonical context/memory hierarchy | preprint; no background weight learning | high |
| 11 | Zep temporal knowledge graph paper | graph external memory | provenance/temporality alternative | product and paper boundaries | medium-high |
| 12 | Mem0 production-ready memory paper | vector+graph update lifecycle | widely deployed alternative | vendor-run benchmark, version drift | medium-high |
| 13 | Titans | wake neural-memory bridge | fixed-state learned update mechanism | not sleep; Google lineage already dense | medium-high, existing asset |
| 14 | Nested Learning / HOPE | multi-frequency optimization-as-memory | bridges update cadence and CMS | not deployed deferred consolidation | high, existing asset |
| 15 | Memory Caching | state reuse/serving problem | directly connects recurrent state and prefix reuse | recent status and source version | high, existing asset |
| 16 | ATLAS | long-stream neural memory | strong long-context result and pruning | foreground state, not cross-session | medium, existing asset |
| 17 | MIRAS | retention/objective framework | theory of fixed-state forgetting | same author lineage as Titans/ATLAS | medium, existing asset |
| 18 | Online Neural Space-Time Memory | memory caching/serving adjacency | system-like state abstraction | newest source, independent verification low | medium, existing asset |
| 19 | What to Keep, What to Forget, 2607.08032 | rate–distortion compaction | capacity budget formalization | proposal/small reference experiment | high |
| 20 | MemDefrag, 2607.05969 | latent capacity/defragmentation | direct bounded latent-memory mechanism | preprint; identity and reproducibility audit | high |
| 21 | Episodic-to-Semantic Consolidation Without Identity Drift, 2607.01988 | external semantic consolidation | direct recent lifecycle | venue identity and scale boundary | high |
| 22 | How Much Do Language Models Memorize?, 2505.24832 | parametric capacity | quantitative capacity boundary | static memorization protocol, not sleep | high |
| 23 | Understanding LoRA as Knowledge Memory, 2603.01097 v4 | adapter capacity | relevant to selective promotion | publication version and figure rights | high |
| 24 | SEAL: Self-Adapting Language Models, 2506.10943 | self-generated update data | bridge to wake/self-adaptation | foreground/episode boundary | medium-high |
| 25 | ROME or MEMIT | model editing alternative | direct fact-update comparator | sequential accumulation not central in original | medium |
| 26 | DANN, SSRN 5402490 | blog-named claim | useful as evidence-hygiene example | abstract-only; inaccessible methods | exclude unless verifiable full text emerges |

## 4. Provisional balanced sets

### Set A — 12-paper minimum

1. Sleep-Time Compute
2. Language Models Need Sleep
3. Can a Language Model Learn Facts Continually in Its Weights?
4. Golden et al. PLOS 2022
5. Singh–Norman–Schapiro PNAS 2022
6. Deep Generative Dual Memory Network
7. Deep Generative Replay
8. MemGPT
9. Zep temporal KG
10. Nested Learning / HOPE
11. Memory Caching
12. What to Keep, What to Forget

### Set B — 15-paper preferred

Set A에 다음을 추가한다.

13. Perturbed and Adversarial Dreaming
14. How Much Do Language Models Memorize?
15. Understanding LoRA as Knowledge Memory

Mem0는 Zep 또는 MemGPT source/rights/access 문제가 생길 때 external-memory reserve로 둔다. Titans/ATLAS/MIRAS/NSTM/TNT 기존 번역은 core set 밖에서도 appendix reference library로 유지한다.

## 5. Source-fidelity contract

각 selected paper directory는 다음을 보유한다.

```text
source-manifest.json
main.tex
body.tex
figures/
translation-qa.json
```

`source-manifest.json`은 source URL, exact version, PDF SHA-256, license evidence, section/equation/table/figure counts를 기록한다. `translation-qa.json`은 다음 parity를 fail-closed로 검증한다.

- original and translated section sequence
- equation labels/count
- table labels/count and numeric cell preservation
- figure labels/count, source and caption
- citation keys/count
- untranslated accidental English fragments와 번역자 해설 삽입 여부

## 6. Figure handling in translations

1. source license가 redistribution을 명시하면 exact figure를 보존한다.
2. redistribution이 불명확하면 public release에서는 original figure를 포함하지 않고, license-preserving source link와 numbered placeholder treatment를 legal review한다.
3. faithful translation은 figure meaning을 새로 해석한 redraw로 대체하지 않는다. redraw는 Study Paper에만 사용한다.
4. source PDF page crop은 편의상 가능하다는 이유만으로 commit하지 않는다.

## 7. Final selection gate

각 후보는 0–2점으로 평가한다.

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| load-bearing relevance | peripheral | one section | central verdict |
| lineage independence | duplicated | partial | independent |
| full-text/version | unavailable | mutable/unclear | fixed and auditable |
| structural translation feasibility | severe obstruction | manual repair | reproducible |
| figure rights | blocked | redraw/link only | explicit reuse |
| evidence balance | duplicates consensus | neutral | fills negative/alternative gap |

총점 9 이상을 우선 선정하되, stratum balance와 12–15 count가 우선한다.
