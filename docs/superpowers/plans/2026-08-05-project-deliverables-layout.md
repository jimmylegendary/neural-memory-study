# Neural Memory / Sleep-Time Compute Deliverables Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 Neural Memory 산출물과 새 Sleep-Time Compute 산출물을 동일한 배포 루트 아래의 두 프로젝트 디렉터리로 명확히 분리한다.

**Architecture:** 기존 PDF/PPTX/XLSX 생성 경로는 재현성과 이전 링크 호환을 위해 유지한다. 새 `deliverables/` 계층은 tracked source artifacts를 결정론적으로 패키징하며, 각 프로젝트별 category 디렉터리·README·manifest·SHA-256 목록을 생성한다. 패키지 파일은 source와 byte-identical이어야 하고 Sleep-Time Compute 번역물의 `rights` metadata를 보존한다.

**Tech Stack:** Python 3 standard library, pytest, Git, SHA-256

## Global Constraints

- 승인 범위는 배포 구조, 패키징 스크립트, layout test, 관련 README·manifest·checksum으로 제한한다.
- 기존 Neural Memory 및 Sleep-Time Compute source/build artifact는 삭제하거나 경로를 변경하지 않는다.
- `deliverables/neural-memory/`와 `deliverables/sleep-time-compute/`는 동일 계층에 둔다.
- 패키징 결과는 입력 source와 byte-identical이어야 하며 결정론적으로 재생성 가능해야 한다.
- Sleep-Time Compute 번역물의 `public`/`internal-only` 권리 경계를 manifest에 그대로 보존한다.
- main의 기존 user-owned untracked 파일은 건드리지 않는다.
- 호스트 부하 때문에 build와 test는 한 번에 하나씩 실행한다.

---

### Task 1: 배포 layout 계약을 테스트로 고정

**Files:**
- Create: `tests/test_package_deliverables.py`
- Create: `scripts/package_deliverables.py` (Task 2에서 구현)

**Interfaces:**
- Consumes: repository에 tracked된 Neural Memory 및 Sleep-Time Compute binary artifacts
- Produces: `scripts/package_deliverables.py --root <repo> --output <dir>`의 검증 가능한 CLI 계약

- [x] **Step 1: Write the failing test**

  실제 repository artifacts를 임시 output에 패키징하도록 CLI를 호출하고, 두 프로젝트 디렉터리·category·정확한 artifact 수·manifest source mapping·SHA-256·translation rights를 검증한다.

- [x] **Step 2: Run test to verify it fails**

  Run: `python3 -m pytest tests/test_package_deliverables.py -q`

  Expected: FAIL because `scripts/package_deliverables.py` does not exist.

### Task 2: 결정론적 패키저와 두 프로젝트 패키지 생성

**Files:**
- Create: `scripts/package_deliverables.py`
- Create: `deliverables/README.md`
- Create: `deliverables/neural-memory/**`
- Create: `deliverables/sleep-time-compute/**`

**Interfaces:**
- Consumes: static artifact selection rules and STC translation `build-manifest.json`
- Produces: project package trees, per-project `manifest.json`, `SHA256SUMS`, and reader README files

- [x] **Step 1: Write minimal implementation**

  Neural Memory는 `study/easy/translations/seminar/analysis`, Sleep-Time Compute는 `study/easy/translations/seminar/manifests`로 분류한다. Output replacement는 지정된 project directories 안에서만 수행하고, hard-link 우선·copy fallback으로 source bytes를 보존한다.

- [x] **Step 2: Run test to verify it passes**

  Run: `python3 -m pytest tests/test_package_deliverables.py -q`

  Expected: PASS.

- [x] **Step 3: Generate the repository package**

  Run: `python3 scripts/package_deliverables.py --root . --output deliverables`

  Expected: two project package trees and deterministic manifests/checksums.

### Task 3: 독자 경로와 전체 검증 갱신

**Files:**
- Modify: `README.md`
- Modify: `research/sleep-time-compute/DELIVERABLES.md`
- Modify: `docs/superpowers/plans/2026-08-05-project-deliverables-layout.md`

**Interfaces:**
- Consumes: Task 2 package paths
- Produces: canonical reader navigation and reproducibility commands

- [x] **Step 1: Document the two-package entrypoint**

  Root README에서 `deliverables/neural-memory/`와 `deliverables/sleep-time-compute/`를 첫 진입점으로 제시하고 STC deliverables index에 패키징·검증 명령을 추가한다.

- [x] **Step 2: Verify package integrity**

  Run: `python3 scripts/package_deliverables.py --root . --output deliverables --check`

  Expected: PASS with every destination matching its source and checksum.

- [x] **Step 3: Run focused and existing release tests**

  Run: `python3 -m pytest tests/test_package_deliverables.py research/sleep-time-compute/tests presentation/sleep-time-compute-deep/tests/test_verify_deck_release.py -q`

  Expected: all tests PASS.

- [x] **Step 4: Mark this plan complete**

  Change every checkbox to `[x]` only after its evidence exists.

### Task 4: Commit, integrate, and publish

**Files:**
- Modify: Git index and feature/main refs only

**Interfaces:**
- Consumes: verified Task 1–3 changes
- Produces: pushed feature branch and `origin/main` at the same final commit

- [x] **Step 1: Review exact diff and artifact counts**
- [x] **Step 2: Commit the scoped changes**
- [x] **Step 3: Merge into main without disturbing user-owned untracked files**
- [x] **Step 4: Push feature branch and main; verify remote SHAs and clean tracked state**
