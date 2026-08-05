# STC Research

Sleep-Time Compute의 evidence, claim, artifact, publication gate를 함께 관리하는 연구 package다. 문헌 근거와 저자 제안, 사전 등록된 실험 설계와 실제 측정 결과를 구분한다.

독자용 최신 release는 [`DELIVERABLES.md`](DELIVERABLES.md)에서 시작한다. 여기서 Study, Training Background, 15편 의역, section별 쉬운 설명, 112-slide seminar, evidence ledger, theory·benchmark·system 설계와 QA report로 이동할 수 있다. Release 전체의 hash와 공개 경계는 [`STC-STUDY-RELEASE-MANIFEST.json`](../../build/publications/STC-STUDY-RELEASE-MANIFEST.json)이 고정한다.

**Source freeze:** 2026-08-05. 이후 공개된 연구는 이 판본의 근거 집합에 자동 포함되지 않는다.

**Rights boundary:** 의역본 15편 중 `public` 2편과 `internal-only` 13편을 분리한다. 특히 internal-only PDF와 source plate는 외부 배포 경로에 포함하면 안 된다. 저자 작성 산출물 역시 외부 공개 전 조직의 legal/publication review가 필요하다.

Run the package-local checks with:

```bash
uv run pytest
```

## Platform boundary

This package is Linux/POSIX-only. Registry durability requires
`fcntl.flock`, atomic same-filesystem `os.replace`, file `fsync`, and parent
directory `fsync`. Windows is unsupported.

## Registry locking and durability

`append_unique` and `write_jsonl_atomic` acquire the same persistent sibling
lock, `<registry>.jsonl.lock`, before reading or replacing a registry. Lock
files are ignored zero-byte runtime coordination artifacts.

Data destination basenames ending exactly in `.lock` or starting with
`.stc-jsonl-` are reserved sidecar namespaces. The same rule applies to
auto-created parent components. Both public writers reject reserved paths
before creating parent directories or materializing records, so rejection does
not create a destination, lock, temporary file, or directory.

An existing coordination lock must be a zero-byte regular file. Locks are
opened with Linux `O_NOFOLLOW` and `O_CLOEXEC`, then checked again through the
opened descriptor before `flock`. A hostile external process that renames or
replaces a validated lock directory entry is outside the cooperative
advisory-lock threat model.

Each destination has a deterministic temporary namespace derived from the
SHA-256 digest of its normalized lexical absolute path. The namespace prevents
writers for names such as `a` and `a.b` from deleting one another's active
temporary files. A lock holder removes only orphaned files in its own namespace,
and the registry `.gitignore` excludes that namespace to prevent staged data
leakage.

New registries use mode `0644`. Replacements preserve only the existing
owner/group/other permission bits; setuid, setgid, and sticky bits are stripped.
The selected mode is applied to the temporary file before its file `fsync`.
Registry destinations must be regular files: symbolic links and non-regular
files are rejected, and destination and referent bytes remain unchanged. A
zero-byte coordination lock may be created before rejection.

If replacement succeeds but parent-directory `fsync` fails, the writer raises
`CommitOutcomeUnknownError`. Its `retry_token` permits an exact idempotent
retry. An append retry recognizes an already committed identical row; a bulk
write retry refuses to overwrite newer state.

Bulk replacement is compare-and-swap. Creating an absent registry and writing
an identical canonical payload are allowed without a precondition. Changing an
existing registry requires `expected_current_digest`, the lowercase SHA-256
digest returned by `canonical_jsonl_digest` for the current canonical records.
If that digest is missing or stale, `write_jsonl_atomic` raises
`StaleWriteError` and preserves the current registry.

## Git object ID format

`GateRecord.evaluation_commit_sha` and
`GateRecord.scientific_candidate_sha` are Git SHA-1 object IDs encoded as
exactly 40 lowercase hexadecimal characters. Other Git object formats,
abbreviated IDs, uppercase hexadecimal, and the all-zero null object ID are
unsupported. Gate tree and artifact digest fields remain lowercase SHA-256
digests.

## Primary-source audit corpus

The pre-registry research audit is split by evidence family:

- [`PRIMARY-SOURCE-AUDIT-BIOLOGICAL-COMPUTATIONAL-SLEEP.md`](PRIMARY-SOURCE-AUDIT-BIOLOGICAL-COMPUTATIONAL-SLEEP.md)
  covers neuroscience boundaries, replay, PLOS 2022, PAD, SRC, and DANN.
- [`PRIMARY-SOURCE-AUDIT-EARLY-PRECURSORS.md`](PRIMARY-SOURCE-AUDIT-EARLY-PRECURSORS.md)
  covers direct wake--sleep algorithmic precursors from DGDMN through learned
  sleep control.
- [`PRIMARY-SOURCE-AUDIT-AGENT-MEMORY-SYSTEMS.md`](PRIMARY-SOURCE-AUDIT-AGENT-MEMORY-SYSTEMS.md)
  audits MemGPT/Letta, Mem0, Zep/Graphiti, LangMem/LangGraph, Memento 2,
  Memento-Skills/DreamDaemon, Evolve, Mela, MIRROR, and HeLa-Mem at the
  paper--code--data and implementation boundaries.
- [`PRIMARY-SOURCE-AUDIT-CAPACITY-CONSOLIDATION-THEORY.md`](PRIMARY-SOURCE-AUDIT-CAPACITY-CONSOLIDATION-THEORY.md)
  traces the bounded-state, complementary-systems, replay, regularisation,
  isolation, expansion, and plasticity-loss theories that constrain lifetime
  sleep learning, plus conditional LLM bits/parameter measurements and
  pretraining-time parametric/external allocation.
- [`PRIMARY-SOURCE-AUDIT-JULY-2026.md`](PRIMARY-SOURCE-AUDIT-JULY-2026.md)
  covers the July 2026 capacity, compaction, and facts-in-weights papers that
  constrain the central thesis.
- [`PRIMARY-SOURCE-AUDIT-LAST-MILE-2026.md`](PRIMARY-SOURCE-AUDIT-LAST-MILE-2026.md)
  closes the 2026 discovery pass with ImprintBench, LoRA as knowledge memory,
  MemoryBench, the agent-native systems comparison, TiMem, MEMORA, RecMem, and
  GAM; it adds internalization, module-routing, multi-clock service, embodied
  memory, and bounded-live-versus-total-state tests.
- [`PRIMARY-SOURCE-AUDIT-COMPANY-LINEAGES.md`](PRIMARY-SOURCE-AUDIT-COMPANY-LINEAGES.md)
  audits the Google/DeepMind, Meta/FAIR, Microsoft, and deployed agent-memory
  lineages under one strict wake--sleep lifecycle test, including chronology,
  numerical anchors, capacity boundaries, and governance nonclaims.
- [`SYSTEMS-INFRA-PRIMARY-SOURCE-AUDIT.md`](SYSTEMS-INFRA-PRIMARY-SOURCE-AUDIT.md)
  separates known database, compaction, dynamic-ANN, adapter-serving,
  scheduling, and disaggregation primitives from STC-specific integration and
  unresolved novelty.
- [`TRAINING-DATA-METHODS-ATLAS.md`](TRAINING-DATA-METHODS-ATLAS.md)
  compares replay, distillation, dream-data construction, RL control,
  parametric isolation, external consolidation, and their matched benchmark
  requirements.
- [`OPEN-QUESTIONS-ANSWERABILITY-MATRIX.md`](OPEN-QUESTIONS-ANSWERABILITY-MATRIX.md)
  maps RQ1–RQ12 to what existing evidence can answer, the experiment needed
  next, preregistered H-STC hypotheses, and separate exploratory extensions.
- [`SYSTEM-INFRA-BLUEPRINT.md`](SYSTEM-INFRA-BLUEPRINT.md)
  specifies the wake, snapshot, sleep, promotion, and memory planes; physical
  deployment options; data-movement accounting; capacity control; rollback;
  and deletion contracts.
- [`SCALING-LAWS-THEORY-AGENDA.md`](SCALING-LAWS-THEORY-AGENDA.md)
  separates accounting identities and necessary bounds from derived or
  empirical candidate laws, with units, countermodels, holdouts, and
  claim-down rules.
- [`BENCHMARK-EXPERIMENT-BLUEPRINT.md`](BENCHMARK-EXPERIMENT-BLUEPRINT.md)
  freezes the controlled lifetime stream, information boundary, matched
  methods/resources, 297-cell confirmatory design, H-STC-007 extension,
  metrics, statistics, failure injection, and null-paper rules.
- [`SCIENTIFIC-RED-TEAM-AUDIT-2026-07-25.md`](SCIENTIFIC-RED-TEAM-AUDIT-2026-07-25.md)
  records 31 adversarial theory/estimand/system/publication findings across the initial
  audit and second-pass re-audit, their incorporated resolutions, and the
  independent re-audits required before G2 and G4.

These are review artifacts, not yet canonical registry records. Stable source,
evidence, and claim IDs are assigned only after validation and rights review.
