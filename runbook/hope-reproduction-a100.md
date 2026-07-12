# HOPE 재현 사내 A100 runbook

> **지위**: Part III(원저 기여) 24장 proposals의 알고리즘 실측을 사내 클러스터에서 실행하기 위한 실전 가이드이자 독립 산출물. `DECISIONS.md` D3(3-트랙 실험) 트랙①, D5(사내 A100 클러스터 스펙)의 실행 문서다.
> **대상 독자**: transformer inference / efficient-transformer를 아는 AI system infra 엔지니어. **training·multi-node·분산 경험은 전무**하다고 가정한다. 그래서 이 문서는 "왜 이 명령인가"의 근거를 명령마다 붙인다.
> **선행 정독 문서**: `notes/impl-availability.md`(구현 가용성 recon — HOPE 공식 구현 없음), `notes/2512.24695.json`([NL]/HOPE 메커니즘), `DECISIONS.md` D5(클러스터 스펙), `style/STYLE-NOTATION.md`(표기).
> **정직성 계약**: 이 라인은 6편 모두 H100 wall-clock decode를 공개하지 않았고 HOPE 공식 구현도 없다. 따라서 이 runbook이 **HOPE 결과 재현의 사실상 첫 시도 축**에 든다(`impl-availability.md` §2.5). 확실치 않은 지점(논문 미명시 하이퍼파라미터 등)은 본문에서 그때그때 **[미명시]** 로 표시하고 대체 전략을 준다. 과장하지 않는다.

표기는 `style/STYLE-NOTATION.md` v1.1을 따른다. 핵심만 재확인: $W$ = fast weights(inner-loop 상태), $\Theta$ = slow weights(outer-loop parameter). inner learning rate $\eta_t$, momentum decay $\beta_t$, retention gate $\alpha_t\in[0,1]$(남기는 비율). inner loss $\ell$, outer(task) loss $\mathcal{L}$. chunk 크기 $C$. HOPE의 self-modifying memory는 $\mathcal{M}_\square(\cdot;W_{\square,t})$, $\square\in\{k,v,q,\eta,\alpha,\mathrm{mem}\}$.

---

## 0. HOPE가 실제로 무엇을 학습·서빙하는가 (구현자용 30초 요약)

runbook의 모든 트러블슈팅은 아래 구조에서 파생하므로, 명령을 치기 전에 한 번 고정한다. HOPE block = **self-modifying Titans**(작은 state, 표현력 높은 DGD 규칙) → **CMS chain**(큰 capacity, 단순 규칙). [NL] Eqs. 94–97.

**self-modifying Titans (inner loop, per-token).** 6개의 memory가 각각 2-layer residual MLP $\mathcal{M}_\square(z;W)=z+W_1\sigma(W_2 z)$이고, 매 token마다 DGD-with-weight-decay로 자기 weights를 갱신한다:

$$
W_{\square,t} \;=\; W_{\square,t-1}\big(\alpha_t I - \eta_t k_t k_t^\top\big) \;-\; \eta_t\,\nabla_W\,\ell\big(W_{\square,t-1};\,k_t,\hat v_{\square,t}\big),\qquad \square\in\{k,v,q,\eta,\alpha,\mathrm{mem}\}
\tag{HOPE-1}
$$

여기서 (i) 학습률 $\eta_t=\mathcal{M}_\eta(x_t)$와 retention gate $\alpha_t=\mathcal{M}_\alpha(x_t)$가 **그 자체로 memory의 출력**이고, (ii) 각 memory의 regression target $\hat v_{\square,t}=\mathcal{M}_\square(v_t;W_{\square,t-1})$가 **자기 자신이 생성**한다(self-referential; Schmidhuber SRWM 계보). $q$ projection은 최종 설계에서 static($q_t=W_Qx_t$)으로 남긴다 — ablation(Table 6)에서 inner-q를 freeze하면 ppl이 오히려 약간 낫다(12.19 vs 12.24). **이 사실이 X10/X12 트러블슈팅의 근거다.** 모든 초기 상태 $W_{\square,0}=W_{\square,\mathrm{init}}$은 outer loop(pre-training)에서 meta-learn된다.

**CMS (inner loop, per-chunk).** MLP block의 chain $\mathrm{MLP}^{(f_1)}\!\to\cdots\to\mathrm{MLP}^{(f_k)}$. level $\ell$은 $C^{(\ell)}$ token마다 한 번, task loss($\mathcal{L}$, next-token prediction)의 error를 누적해 갱신(master eq. M5):

$$
\theta^{(\ell)}_{i+1} = \theta^{(\ell)}_{i} - e_{i,\ell},\qquad e_{i,\ell}=\begin{cases}\sum_{t=i-C^{(\ell)}+1}^{i}\eta^{(\ell)}_t\,\nabla_\theta\mathcal{L}(\theta^{(\ell)}_t;x_t) & i\equiv 0\ (\mathrm{mod}\ C^{(\ell)})\\ 0 & \text{else}\end{cases}
\tag{HOPE-2}
$$

표준 Transformer MLP은 $k{=}1$, frequency-0(동결)인 퇴화 case다.

**outer loop (pre-training, 우리가 A100에서 돌리는 것).** AdamW가 $\Theta$(모든 projection, gate-producer, $W_{\square,\mathrm{init}}$, CMS 각 level의 초기 파라미터, embedding/norm/conv)를 학습한다. 핵심: **outer gradient가 inner loop의 unrolled recurrence를 관통해 흐른다** — pre-training autograd 그래프 안에 fast memory의 per-token/per-chunk weight-update 전개가 통째로 들어 있다. 이것이 HOPE를 **훈련 시** 비싸게 만드는 원인이자(서빙 시엔 안 보임) 이 runbook 전체가 관리하는 대상이다.

**chunkwise parallelization (훈련·prefill을 가능케 하는 것).** 길이 $L$을 $\lceil L/C\rceil$ chunk로 쪼개고, 각 memory를 직전 chunk 끝 상태 $W_{\xi(t,C)}$로 **동결(stale snapshot)** 한 뒤 chunk 전체의 key/value/self-target/$\eta$/$\alpha$/gate를 한 번에 batched 생성하고 inner gradient를 병렬로 계산한다(master eq. M4). **실무상 chunk 크기 두 개**: 하나는 $\mathcal{M}_{\mathrm{mem}}$용, 하나는 나머지 auxiliary memory 공유. $C$는 스케줄이 아니라 **계산되는 함수 자체를 바꾸는 semantic hyperparameter**(FlashAttention tiling과의 결정적 차이) — X11/X15의 근거.

---

## 1. 목표·범위·성공 기준

### 1.1 이 runbook이 산출하는 것

1. **HOPE 아키텍처(self-modifying Titans + CMS)를 사내 A100 8장에서 from-scratch 학습**하고, language modeling으로 재현 신호를 확인한다.
2. FLA 기반 baseline(Transformer++, DeltaNet)과 **같은 data·token budget·tokenizer** 위에서 대조한다. 절대 수치보다 **상대 순위**가 성공 기준이다(정직성 계약).
3. HOPE가 학습되면, Part III 로컬 실험(`experiments/results.json`)이 A100으로 **이월(carry-forward)** 한 6개 실측 항목을 닫는다(§6.4). 로컬 실험은 analytic cost model이라 절대 wall-clock을 유보했고, 그 유보분이 이 runbook의 2차 목표다.

### 1.2 재현 대상 규모 — 8장에서 현실적인 것

[NL] 헤드라인은 두 규모다: **760M / 30B tokens** 와 **1.3B / 100B tokens** (FineWeb-Edu + long-context mix, vocab 32K, AdamW). A100 80GB×4×2node=8장 fast-path에서:

| 규모 | token budget | 8×A100-80G에서 판정 | 비고 |
|---|---|---|---|
| **HOPE-760M** | 30B | **1차 목표 (권고)** | inner-loop autograd 오버헤드까지 감안해도 GBS·activation checkpointing으로 8장 수용 가능. 예상 벽시계는 §6.2 |
| HOPE-1.3B | 100B | 2차(여유·최대 32장 확보 시) | 100B token은 8장에서 벽시계가 커진다. token budget을 줄인 축소 재현(예: 30B)으로 먼저 신호만 본 뒤 확대 |

**결정: 1차 재현 = HOPE-760M / 30B.** 이유: (i) 논문의 최소 헤드라인 규모여서 baseline 표와 직접 대조된다, (ii) 8장에서 activation checkpointing + FSDP로 벽시계가 관리 가능, (iii) inner-loop 그래프 폭발(X10) 같은 신규 리스크를 작은 규모에서 먼저 소진한다. 1.3B는 760M 신호가 확인된 뒤 판단.

> **[미명시] 경고**: [NL]은 HOPE의 **memory MLP 폭(hidden dim), CMS level 수·각 frequency, 두 chunk 크기, inner/outer lr, warmup**을 논문 본문에서 완전히 고정해 주지 않는다(`2512.24695.json` `assumptions_and_scope`: "Chunk sizes, number of CMS levels, and frequencies are hyperparameters chosen empirically"). 논문이 준 앵커: **CMS 4-level, lowest-frequency $C^{(\ell)}=2\mathrm{K}$가 효율/품질 sweet spot(512가 최상이나 2K가 near-parity·훨씬 저렴)**, 표준 memory = 2-layer residual MLP(expansion 4, GELU), $q,k$는 L2-normalize, window-4 local conv. 그 외는 §3.6의 sweep로 우리가 정한다. 이 불확실성을 성공 기준에서 감안한다.

### 1.3 검증 지표 (성공 기준, 정직성 등급 명시)

| 지표 | 측정 | 성공 기준 | 등급 |
|---|---|---|---|
| **G-1 LM ppl** | WikiText-103 / LAMBADA validation ppl | HOPE가 **동일 조건 DeltaNet·Transformer++ baseline보다 낮은 ppl** (순위 재현). 논문 절대치(760M Wiki 18.68, LAMBADA 20.07)는 참고선일 뿐, tokenizer/data 차이로 절대 일치는 기대하지 않음 | **load-bearing = 순위**. 절대 ppl은 directional |
| **G-2 common-sense acc** | 논문 avg-acc 세트 부분집합(lm-eval-harness) | HOPE avg-acc ≥ baseline (논문: 760M Hope 52.28 > Titans 51.68 > RWKV-7 50.55) | 순위 재현 |
| **G-3 long-context** | BABILong 부분(booydar/babilong) + RULER S-NIAH 일부 | 길이 증가에도 열화가 attention-free baseline보다 완만 (논문: BABILong 10M까지 유지). 8장 예산상 **부분·짧은 길이만**, full 10M은 후속 | directional, 부분 재현 |
| **G-4 학습 안정성** | outer loss curve, grad-norm, inner-state 통계 | NaN/발산 없이 target token budget 도달, inner memory가 degenerate(§X12)하지 않음 | pass/fail |

> **정직성**: G-1/G-2의 load-bearing 주장은 **HOPE vs baseline 순위**다. 절대 ppl 재현은 목표가 아니다 — 우리 data mix·tokenizer·token budget이 논문과 정확히 같지 않고(재현 불가한 부분), 자체 구현이라 hyperparameter가 다르다. "HOPE가 같은 예산에서 강한 recurrent baseline을 이긴다"를 보이면 아키텍처 재현은 성공이다.

### 1.4 범위 밖 (명시적 제외)

- **Dreaming/Sleep**(wake-sleep consolidation, self-improvement loop, periodic parameter (de)activation): 별도 후속 runbook. 공개 구현 0(`impl-availability.md` §2.6) → 완전 자체 구현이고 이 문서의 HOPE 재현이 그 전제. 포인터는 §6.5.
- **TNT 재현**: fast-path 우선순위 낮음(`impl-availability.md` §2.4).
- **M3 optimizer**(Multi-scale Momentum Muon): HOPE 아키텍처와 독립 축이므로 별도. HOPE 학습에는 표준 AdamW 사용(논문도 outer는 AdamW).

---

## 2. 환경 구성

### 2.1 컨테이너 이미지 (버전 고정)

사내 노드는 Docker·Singularity 둘 다 쓰고 **컴퓨트 노드에서 인터넷 접근이 된다**(D5) → 이미지 빌드 시 온라인 pull, 데이터/checkpoint HF 온라인 다운로드 경로로 작성한다. 재현성을 위해 **모든 버전을 핀**한다.

`Dockerfile` (NFS 상 `/nfs/hope/env/Dockerfile`):

```dockerfile
# CUDA 12.4 + cuDNN, PyTorch 2.5.x — A100(sm_80) 검증 조합
FROM nvcr.io/nvidia/pytorch:24.10-py3
# 위 베이스가 이미 torch 2.5 + CUDA 12.4 + NCCL 2.22 동봉. NCCL을 별도 빌드하지 않는다(IB 호환 위해 base 것 사용).

ENV PIP_NO_CACHE_DIR=1 HF_HUB_ENABLE_HF_TRANSFER=1
# --- 핀된 의존성 (해시는 빌드 후 pip freeze로 lockfile 고정) ---
RUN pip install \
      "triton==3.1.0" \
      "flash-linear-attention==0.3.2" \
      "einops==0.8.0" \
      "transformers==4.46.2" \
      "datasets==3.1.0" \
      "hf_transfer==0.1.8" \
      "lm-eval==0.4.5" \
      "wandb==0.18.5"
# flame(torchtitan 기반 FLA 학습 프레임워크)은 소스로 고정 커밋 체크아웃
RUN git clone https://github.com/fla-org/flame /opt/flame && \
    cd /opt/flame && git checkout <PINNED_COMMIT> && pip install -e .
# 우리 자체 HOPE 구현은 학습 시 -v 바인드마운트로 주입(이미지 재빌드 없이 반복)
WORKDIR /workspace
```

빌드·검증(빌드 노드에서 1회):

```bash
cd /nfs/hope/env
docker build -t hope:cu124-torch25 .
# 즉시 sanity: torch가 GPU/NCCL/IB를 보는지
docker run --rm --gpus all hope:cu124-torch25 python -c \
 "import torch,fla; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count()); \
  print('NCCL', torch.cuda.nccl.version()); print('FLA', fla.__version__)"
```

**근거**: NGC PyTorch 이미지를 base로 쓰는 이유는 NCCL이 InfiniBand용으로 이미 빌드·검증돼 있어(§4·X4의 IB hang 리스크를 줄인다) NCCL을 직접 컴파일하지 않아도 되기 때문이다. Triton 3.1은 FLA 0.3.x의 `fla/ops/ttt` 커널이 요구하는 버전에 맞춘다(`impl-availability.md` §3의 ttt Triton 커널을 쓸 것이므로 mismatch 시 X11).

Singularity 변환(스토리지가 Docker daemon 없이 Singularity만 되는 노드용):

```bash
# NFS에 .sif 하나 만들어 두면 전 노드가 공유(이미지 pull 트래픽 0)
singularity build /nfs/hope/env/hope.sif docker-daemon://hope:cu124-torch25
# GPU 가시성은 --nv 플래그 (Docker의 --gpus all 대응) — 빼먹으면 X1
singularity exec --nv /nfs/hope/env/hope.sif python -c "import torch;print(torch.cuda.device_count())"
```

### 2.2 의존성·HF 자산 사전 확보 (인터넷 경유)

컴퓨트 노드가 인터넷을 보므로 학습 중 스트리밍도 가능하지만, **간헐 실패(X6)와 rate-limit(X7)** 를 피하려고 **데이터·tokenizer·평가셋을 NFS에 미리 물질화**한다. HF 캐시를 NFS 한 곳으로 고정:

```bash
export HF_HOME=/nfs/hope/hf_cache          # 전 노드 공유 캐시(재다운로드 0)
export HF_HUB_ENABLE_HF_TRANSFER=1         # 대용량 병렬 다운로드
# (선택) gated면 토큰
export HF_TOKEN=<...>

# tokenizer + 데이터 사전 다운로드 (로그인 노드에서 1회, 이후 오프라인로도 학습 가능)
python - <<'PY'
from huggingface_hub import snapshot_download
# 32K vocab tokenizer: 논문 미명시 → FLA baseline과 반드시 동일 것 사용(대조 공정성)
snapshot_download("fla-hub/gla-1.3B-100B", allow_patterns=["*tokenizer*","*.json"])
# 데이터: FineWeb-Edu 샘플 shard (30B token 분량만) + long-context mix
snapshot_download("HuggingFaceFW/fineweb-edu", repo_type="dataset",
                  allow_patterns=["sample/10BT/*"])   # 필요 shard만, 전체 X
PY

# 평가셋
python -c "from datasets import load_dataset as L; L('wikitext','wikitext-103-raw-v1'); L('lambada')"
git clone https://github.com/booydar/babilong /nfs/hope/eval/babilong   # BABILong harness
```

> **[미명시] tokenizer/data 결정**: 논문은 "32K vocab, FineWeb-Edu + long-context mix"만 말하고 정확한 tokenizer·mix 비율·shard를 공개하지 않는다. **대체 전략**: 절대 재현이 불가하므로 **baseline과 100% 동일한 tokenizer·data pipeline**을 쓴다. 그러면 "같은 data 위 HOPE vs baseline"이라는 성공 기준(§1.3)이 성립한다. FLA/flame이 제공하는 tokenizer+FineWeb-Edu loader를 SoT로 삼는다.

### 2.3 NFS 레이아웃

```
/nfs/hope/
├── env/            Dockerfile, hope.sif, requirements.lock   # 이미지·핀
├── hf_cache/       HF_HOME (tokenizer, FineWeb-Edu, 평가셋)   # 전 노드 공유 read
├── code/           우리 HOPE 구현 (git; 바인드마운트로 주입)
├── data/           전처리·토크나이즈된 .bin shard (mmap용)
├── ckpt/           checkpoint (§X5: atomic write 필수)
│   ├── hope-760m/  step별 dist-checkpoint
│   └── baseline/
├── logs/           per-rank stdout/err, NCCL_DEBUG 로그
└── eval/           babilong, lm-eval 산출물
```

> **NFS 주의(§X5 예고)**: checkpoint I/O는 NFS의 최대 병목이자 부분쓰기·stale handle의 원천이다. 규칙: (1) **rank0만 rendezvous 메타를 쓰고**, sharded checkpoint는 각 rank가 **자기 파일**에 쓴다(동시 쓰기 충돌 회피), (2) 항상 `tmp`에 쓰고 `os.replace()`로 **atomic rename**(부분쓰기 방지), (3) 전처리 `.bin` shard는 `data/`에 미리 만들어 학습 중엔 **read-only mmap**만(학습 hot path에서 NFS write 금지).

---

## 3. 구현 전략 — FLA+flame 위 HOPE 자체 구현 (C1–C5)

**하이브리드 4층**(`impl-availability.md` §6)을 그대로 따른다: ① baseline·backbone = FLA+flame(공식급 신뢰), ② Titans neural memory = lucidrains 참조를 **논문 모드로 감사**한 뒤 사용 + `fla/ops/titans` naive를 수치 oracle, ③ HOPE = 자체 구현(obekt 84★/erikl2 76★는 **reference-only, fork 금지**), ④ Sleep = 범위 밖.

**철칙(정직성)**: obekt·erikl2 둘 다 **surrogate loss를 섞어** 있어(`impl-availability.md` §2.5·§5 리스크 (ii)) 그대로 fork하면 "무엇을 측정했는지"가 모호해진다. erikl2가 명시하는 **"paper-exact mode vs stable default(surrogate)"** 구분을 우리 **감사 체크리스트**로 승격해, 우리 구현은 항상 paper-exact를 기본으로 하고 surrogate는 flag로만 켠다.

### 3.1 단계별 unit 검증 원칙

각 컴포넌트는 **수식 단위 → 소형 학습 → 스케일** 순으로 올린다. 앞 단계 gate를 통과 못 하면 다음으로 못 간다.

- **수식 단위**: full-sequence 계산 == token-by-token 계산의 **수치 동치(equivalence test)**. obekt가 바로 이 테스트를 갖고 있고(`impl-availability.md` §2.5), 우리 chunkwise 구현이 naive 순차와 일치하는지 확인하는 것이 X11의 1차 방어선이다. erikl2의 finite-difference gradient check도 재사용.
- **소형 학습**: $\leq$130M, single-GPU, `torch.compile`, 짧은 seq — semantics·수렴 확인(로컬 트랙과 공유).
- **스케일**: 8×A100, FSDP, 760M.

### 3.2 C1 — deep neural memory 코어

**목표**: master update (M2)의 표준 deep memory. momentum $\beta_t$, retention $\alpha_t$, adaptive lr $\eta_t$, chunkwise inner loop.

- **재사용**: lucidrains `titans-pytorch`의 `NeuralMemory`를 **논문 모드 flag 감사** 후 참조. 논문 밖 확장(`spectral_norm_surprises`=Newton-Schulz, `momentum_order>1`, attention pooling, per-parameter lr modulation)을 **전수 off**로 고정해야 한다(`impl-availability.md` §2.1). 기본값이 논문 세팅이라는 보장이 없다.
- **oracle**: `fla/ops/titans`의 `chunk_titans_linear`(naive PyTorch, log-space)를 **수치 정답**으로 병용. Triton 없음이라 느리지만 정답 대조용으로 충분. Titans가 FLA에서 naive에 머문 사실 자체가 "chunk-level momentum은 chunkwise 커널화가 어렵다"는 증거(→ X11이 왜 어려운지의 근거).
- **gate**: 우리 chunkwise 구현 == naive 순차, rtol 1e-4(bf16이면 완화). 공수 3–5일.

### 3.3 C2 — self-modifying Titans (**최난도**)

**목표**: (HOPE-1). k/v/q/η/α/mem 6개 memory가 각각 자기 target $\hat v_{\square,t}$를 생성하는 self-referential update. DGD-with-weight-decay inner rule.

- **재사용**: obekt/erikl2 **참조** + [NL] Part 7 수식 직접. erikl2의 paper-exact/surrogate 구분을 감사 체크리스트로.
- **결정적 세부**: (i) $q$는 static($W_Q$)으로 — ablation 근거(§0), inner-q를 memory로 만들면 손해. (ii) $\eta_t,\alpha_t$가 memory 출력이므로 gate가 다른 memory에 **재귀 의존** → 구현 순서상 gate memory를 먼저 forward. (iii) self-target $\hat v_{\square,t}=\mathcal{M}_\square(v_t;W_{\square,t-1})$는 **직전 상태**로 평가(현재 상태 아님 — 이게 chunkwise 병렬화를 가능케 하는 stale-snapshot의 원천).
- **gate**: 소형에서 6-memory 각각 finite-difference gradient check 통과 + self-target이 붕괴 안 함(§X12). 공수 ~1주.

### 3.4 C3 — CMS

**목표**: (HOPE-2). multi-frequency MLP chain, level별 update period + gradient 누적.

- **재사용**: obekt 구조 참조(본질적으로 단순). 표준 Transformer MLP을 $C^{(\ell)}$마다만 갱신하도록 감싼 것.
- **[미명시] 결정**: 4-level, lowest-frequency $C^{(\ell)}=2\mathrm{K}$(논문 sweet spot). 나머지 frequency는 geometric spacing(예: $\{1, 64, 512, 2048\}$)을 §3.6 sweep으로 확정. Sequential variant(모든 초기 상태를 최저 frequency level에서 backprop) 채택 — Nested보다 구현 단순.
- **gate**: level별 update가 정확히 $C^{(\ell)}$ 주기로만 일어나는지 단위 테스트. 공수 1–2일.

### 3.5 C4 — HOPE block 조립 + LM 골격

self-modifying Titans → CMS chain을 한 block으로. flame(torchtitan) 골격에 custom block 주입, tokenizer/data/config 배선. 세부: $q,k$ L2-normalize, window-4 local conv, 두 chunk 크기(mem용/aux 공유). 공수 3일.

### 3.6 C5 — 병렬 학습 경로 (naive → chunkwise)

- **1차 목표**: naive 순차 inner loop + `torch.compile`. 느리지만 정답이고 X10/X11 리스크가 낮다. 760M/30B의 1차 재현은 여기서 시작.
- **가속(옵션)**: `fla/ops/ttt`의 진짜 Triton 커널(`chunk_ttt_linear`, `fused_chunk_ttt_linear`; group_norm 융합·varlen `cu_seqlens`·initial/final state 지원) 패턴을 HOPE memory에 이식. Titans momentum 항 때문에 chunkwise closed form이 깨지므로(FLA가 Titans를 naive로만 둔 이유) **완전 커널화는 +2주 리스크**. 1차 재현엔 불필요.
- **[미명시] chunk 크기**: 두 chunk 크기(mem/aux)를 sweep. 논문은 "2K lowest-frequency"만 앵커. 시작값 $C_{\mathrm{mem}}=64$, $C_{\mathrm{aux}}=64$에서 품질/처리율 tradeoff 관찰(§X15).

### 3.7 감시 항목 (P2 재확인, `impl-availability.md` §6)

구현 착수 전 재확인: ① `ABehrouz/Titans` 빈 repo에 공식 코드 등장 여부, ② FLA의 Titans Triton화(#107 후속), ③ TNT/Sleep 공식 코드 출현, ④ lucidrains의 HOPE repo. 공식 코드가 나오면 자체 구현 대신 감사 대상으로 승격.

---

## 4. 멀티노드 실행 — 수동 SSH 2노드 8-GPU

**전제(D5)**: 스케줄러 없음. Slurm/K8s 미사용. 노드별 SSH로 `torchrun`을 **직접** 띄우고 `MASTER_ADDR`·`node_rank`를 명시한다. NFS 공유 + InfiniBand.

### 4.1 노드 역할·주소 고정

```bash
# 두 노드에서 공통으로 export (예시 — 사내 IB 서브넷 주소로 교체)
export MASTER_ADDR=10.0.0.11        # node0의 IB 인터페이스 IP (rendezvous 서버)
export MASTER_PORT=29500            # 미사용 포트. 방화벽 열려 있어야(X2)
export NNODES=2
export NPROC_PER_NODE=4             # 노드당 A100 4장
export HF_HOME=/nfs/hope/hf_cache
```

**근거**: 스케줄러가 rank/주소를 안 잡아 주므로 사람이 못 박는다. `MASTER_ADDR`는 **node0의 IB IP**여야 한다(Ethernet mgmt IP를 쓰면 rendezvous는 되도 collective가 IB를 안 타 X4). node0가 rendezvous 서버 겸 rank0.

### 4.2 NCCL/IB 환경변수 (X4 예방)

```bash
export NCCL_DEBUG=WARN              # 문제 시 INFO로 승격
export NCCL_IB_DISABLE=0            # IB 사용(0). 절대 1로 두지 말 것(TCP fallback → 느림)
export NCCL_SOCKET_IFNAME=ib0       # bootstrap 소켓이 IB(또는 mgmt) 인터페이스를 타게
export NCCL_IB_HCA=mlx5             # 사용할 HCA 접두(사내 `ibstat`로 확인해 교체)
export NCCL_IB_GID_INDEX=3          # RoCE/IB GID (사내 fabric에 맞게; 틀리면 hang → X4)
export NCCL_NET_GDR_LEVEL=PHB       # GPUDirect RDMA 레벨(토폴로지에 맞게)
```

> 이 값들은 **사내 fabric 실측으로 교체**한다(§X4에 진단 절차). `NCCL_IB_HCA`/`NCCL_IB_GID_INDEX`가 틀리면 학습이 첫 all-reduce에서 조용히 hang한다 — 가장 흔한 첫날 사고.

### 4.3 런칭 — 두 노드에서 각각

```bash
# ---- node0 (MASTER, node_rank=0) ----
docker run --rm --gpus all --network host --ipc host \
  --device=/dev/infiniband --ulimit memlock=-1:-1 \
  -v /nfs/hope:/nfs/hope -v /nfs/hope/code:/workspace \
  -e MASTER_ADDR -e MASTER_PORT -e HF_HOME \
  -e NCCL_DEBUG -e NCCL_SOCKET_IFNAME -e NCCL_IB_HCA -e NCCL_IB_GID_INDEX \
  hope:cu124-torch25 \
  torchrun --nnodes=$NNODES --node_rank=0 --nproc_per_node=$NPROC_PER_NODE \
           --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
           /workspace/train_hope.py --config /workspace/configs/hope-760m.yaml \
           2>&1 | tee /nfs/hope/logs/node0.log

# ---- node1 (node_rank=1) — MASTER_ADDR는 node0 주소 그대로 ----
docker run --rm --gpus all --network host --ipc host \
  --device=/dev/infiniband --ulimit memlock=-1:-1 \
  -v /nfs/hope:/nfs/hope -v /nfs/hope/code:/workspace \
  -e MASTER_ADDR -e MASTER_PORT -e HF_HOME \
  -e NCCL_DEBUG -e NCCL_SOCKET_IFNAME -e NCCL_IB_HCA -e NCCL_IB_GID_INDEX \
  hope:cu124-torch25 \
  torchrun --nnodes=$NNODES --node_rank=1 --nproc_per_node=$NPROC_PER_NODE \
           --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
           /workspace/train_hope.py --config /workspace/configs/hope-760m.yaml \
           2>&1 | tee /nfs/hope/logs/node1.log
```

**컨테이너 플래그 근거**: `--gpus all`(X1: 빼면 GPU 0장), `--network host`(torchrun rendezvous가 host 네트워크·IB를 직접 보게; bridge면 MASTER_PORT 안 뚫림 → X2), `--device=/dev/infiniband` + `--ipc host` + `--ulimit memlock=-1`(NCCL이 IB verbs·shared memory·pinned memory를 쓰게; memlock 제한 걸리면 IB 등록 실패 → X4), NFS·code 바인드마운트. Singularity면 `singularity exec --nv --bind /nfs/hope /nfs/hope/env/hope.sif torchrun ...`.

### 4.4 분산 전략 (FSDP)

- **데이터 병렬 + FSDP(ZeRO-3 급 샤딩)** 를 기본으로. 760M param 자체는 작지만, **HOPE의 진짜 메모리 압박은 param이 아니라 inner-loop autograd 그래프**(unrolled per-token/per-chunk update)와 optimizer state다(§X8). FSDP로 param·grad·optimizer state를 8-way 샤딩.
- **[중요] custom fast-weight 텐서 wrapping(X9)**: FSDP의 auto-wrap policy가 HOPE block을 통째로 감싸되, **inner-loop에서 매 token 갱신되는 $W_{\square,t}$(fast weights)는 slow weights $\Theta$와 다른 클래스**임을 인지해야 한다. $W_{\square,\mathrm{init}}$(meta-learn되는 slow parameter)만 FSDP flat-param에 들어가고, per-token 갱신되는 $W_{\square,t}$는 activation-급 텐서로 그래프에 존재해야 한다. 이 분리를 틀리면 FSDP가 fast-weight 갱신을 param update로 오인해 all-gather/reduce-scatter를 잘못 건다(X9).
- **context 병렬(옵션, TNT식)**: 8장에서 760M은 데이터 병렬로 충분. 긴 seq로 activation이 터지면(X8) TNT식 periodic state reset이 sequential chain을 끊어 sequence 축 샤딩을 여는데, 이는 1.3B/긴 context 확대 시에만 검토.
- **GBS**: `global_batch = NPROC(8) × micro_batch × grad_accum`. token budget 30B / (GBS × seq_len) = step 수. activation checkpointing 켠 상태에서 micro_batch를 OOM 직전까지(§X8 절차).

---

## 5. 예외 시나리오 · 트러블슈팅

각 시나리오: **증상 → 진단 → 조치**. 번호(X#)는 본문 다른 절에서 참조된다. HOPE 특유(inner-loop 그래프, self-reference)와 인프라 공통(IB/NFS/torchrun)을 함께 다룬다.

### X1 — 컨테이너 GPU 가시성 (0장 GPU / device 못 봄)
- **증상**: `torch.cuda.device_count()==0`, 또는 "no CUDA-capable device".
- **진단**: 컨테이너 안에서 `nvidia-smi`. Docker면 `--gpus all` 누락, Singularity면 `--nv` 누락이 대부분.
- **조치**: Docker `--gpus all`(특정 장만이면 `--gpus '"device=0,1,2,3"'`), Singularity `--nv`. host에 `nvidia-container-toolkit` 설치·`nvidia-ctk runtime configure` 확인. IB까지 쓰려면 `--device=/dev/infiniband`도 필수(없으면 X4로 전이).

### X2 — torchrun rendezvous 실패 (노드가 서로 못 붙음)
- **증상**: node1이 `The client socket has failed to connect to [MASTER_ADDR]:29500` / 무한 대기.
- **진단**: node1에서 `nc -vz $MASTER_ADDR $MASTER_PORT`. 실패면 (i) MASTER_ADDR가 node0의 **도달 가능한** IP인지(IB IP 권장), (ii) 방화벽이 MASTER_PORT 여는지, (iii) 컨테이너가 `--network host`인지(bridge면 포트 격리).
- **조치**: `--network host` 사용, MASTER_PORT를 미사용 포트로, 두 노드 `MASTER_ADDR` 값이 **동일한 node0 주소**인지 재확인(node1이 자기 IP를 넣는 실수가 흔함). node_rank가 0/1로 유일한지(둘 다 0이면 rank 충돌 → X-rank).

### X3 — 노드 간 시계 skew / rendezvous timeout
- **증상**: rendezvous는 붙는데 곧 `Timed out initializing process group` 또는 rank들이 서로 다른 step에서 대기.
- **진단**: 두 노드 `date -u` 비교, `chronyc tracking`/`ntpq -p`. 큰 clock skew는 timeout 계산과 로그 상관을 망친다.
- **조치**: 두 노드 NTP 동기(사내 NTP 서버). rendezvous timeout 여유 확대(`--rdzv-conf timeout=1800` 또는 `TORCH_DISTRIBUTED_INIT_TIMEOUT` 상향). 첫 all-gather가 collective 초기화 비용으로 오래 걸릴 수 있으니 timeout을 넉넉히.

### X4 — NCCL InfiniBand hang (첫 collective에서 멈춤)
- **증상**: 프로세스가 다 뜨고 rendezvous도 됐는데 **첫 all-reduce에서 조용히 hang**(GPU util 0, 진행 로그 없음). 가장 흔한 첫날 사고.
- **진단**: `NCCL_DEBUG=INFO`로 재기동해 어떤 transport를 고르는지 본다("NET/IB" vs "NET/Socket"). host에서 `ibstat`(포트 Active/LinkUp?), `ibv_devinfo`(HCA 이름 → `NCCL_IB_HCA`), `show_gids`(올바른 GID index → `NCCL_IB_GID_INDEX`). IB 링크가 죽었으면 fabric 문제.
- **조치**: `NCCL_IB_HCA`를 실제 HCA 접두(`mlx5`)로, `NCCL_IB_GID_INDEX`를 fabric 정답(RoCEv2는 보통 3)으로 교체. `NCCL_SOCKET_IFNAME`을 bootstrap용 인터페이스로. **임시 우회**(원인 격리용): `NCCL_IB_DISABLE=1`로 TCP fallback → 느리지만 뜨면 IB 설정 문제 확정. `--ulimit memlock=-1`·`--device=/dev/infiniband` 재확인(pinned/verbs 등록 실패도 hang). 여전하면 `NCCL_P2P_LEVEL`/`NCCL_NET_GDR_LEVEL` 토폴로지 조정.

### X5 — NFS checkpoint I/O 병목 · 부분쓰기 · stale handle
- **증상**: (a) checkpoint step에서 전 rank가 수십 초~분 stall(NFS write 직렬화), (b) 재시작 시 checkpoint가 truncated/corrupt(부분쓰기), (c) `Stale file handle`(ESTALE) 에러.
- **진단**: checkpoint 시 `iostat`/NFS 서버 부하 확인. 부분쓰기는 파일 크기·해시 검증. stale handle은 여러 노드가 같은 경로를 동시에 rename/삭제할 때.
- **조치**: (1) **sharded checkpoint**: 각 rank가 자기 shard만 자기 파일에 쓴다(`torch.distributed.checkpoint`), 단일 파일 직렬화 금지. (2) **atomic write**: `tmp`에 쓰고 flush+fsync 후 `os.replace()`로 rename — 부분쓰기 근절. (3) checkpoint 주기를 늘리고(예: 매 2000 step) 비동기 저장. (4) stale handle 방지: 동일 경로 동시 조작 금지, rank0만 메타/latest 심볼릭 갱신. (5) 학습 hot path에선 NFS write 금지(§2.3) — `.bin` shard는 read-only mmap.

### X6 — 인터넷 간헐 실패 (HF/데이터 다운로드 끊김)
- **증상**: 학습·평가 시작 시 HF에서 데이터/tokenizer 받다가 timeout/connection reset.
- **진단**: `curl -I https://huggingface.co` 로 노드 인터넷 확인. 학습 중 스트리밍이면 특히 취약.
- **조치**: §2.2대로 **사전 물질화**가 1차 방어(학습 중 네트워크 의존 제거). 그래도 받아야 하면 `HF_HUB_ENABLE_HF_TRANSFER=1`(재개 지원), `snapshot_download(..., max_workers, resume)` 재시도 래퍼(exponential backoff). 완전 오프라인 강제: `HF_HUB_OFFLINE=1`(캐시만 사용, 누락 시 즉시 실패해 조용한 stall 방지).

### X7 — HF 인증 / rate-limit
- **증상**: `401/403 gated repo` 또는 `429 Too Many Requests`.
- **조치**: gated 자산은 `HF_TOKEN` export + 웹에서 라이선스 accept. 429는 backoff·`max_workers` 축소. 전 노드가 **공유 NFS 캐시**(§2.2)를 보므로 다운로드는 로그인 노드에서 1회만 → rate-limit 노출 최소화.

### X8 — OOM (80GB에서 HOPE state + optimizer)
- **증상**: `CUDA out of memory`. HOPE는 param이 작아도(760M) **inner-loop unrolled 그래프**가 activation을 크게 부풀린다 — outer gradient가 per-token/per-chunk update 전개를 관통하므로(§0).
- **진단**: `torch.cuda.max_memory_allocated()`, OOM이 forward(그래프 축적)인지 optimizer step(state)인지 구분.
- **조치(순서대로)**: (1) **activation checkpointing**을 HOPE block·CMS chain에 — inner-loop 그래프를 재계산으로 교환(가장 효과 큼). (2) **chunk 크기 $C$ 조정**: 큰 $C$는 chunk당 그래프를 키운다; 작게 하면 메모리↓(단 semantic이 바뀜 → X15와의 tradeoff). (3) **FSDP full-shard**로 optimizer state 8-way 분산, 필요 시 **CPU offload**(optimizer state를 host로; NVMe offload는 최후). (4) micro_batch↓ + grad_accum↑로 GBS 유지. (5) bf16 유지(fp32 master는 optimizer만). (6) self-target·gate memory 그래프에서 **불필요한 retain 제거**(§X10과 연동).

### X9 — FSDP가 custom HOPE fast-weight를 잘못 샤딩
- **증상**: 학습은 도는데 loss가 baseline과 근본적으로 다르거나, fast-weight 갱신 후 rank 간 상태 불일치, 또는 예상치 못한 all-gather 트래픽 폭증.
- **진단**: fast weights $W_{\square,t}$(per-token 갱신)가 FSDP flat-param에 들어갔는지 확인. 들어갔다면 FSDP가 이를 param으로 오인.
- **조치**: auto-wrap policy를 **HOPE block 경계**로 잡고, meta-learn되는 slow parameter($W_{\square,\mathrm{init}}$, projection, CMS 초기값)만 `nn.Parameter`로 등록·샤딩한다. per-token 갱신되는 $W_{\square,t}$는 **buffer/activation-급 텐서**(그래프에 존재하되 FSDP 관리 밖)로 둔다. inner-loop update가 autograd로 outer에 연결되되 FSDP의 param-update 경로를 타지 않게 분리. `use_orig_params=True`로 커스텀 텐서 접근을 단순화.

### X10 — self-modifying Titans의 test-time 갱신이 학습 그래프에서 폭발
- **증상**: seq_len·chunk가 커질수록 backward에서 OOM/극단적 느려짐. inner-loop가 길수록 autograd 그래프가 선형~초선형으로 커짐.
- **원인**: outer gradient가 (HOPE-1)의 unrolled per-token update를 **전부** 관통(§0). self-referential target $\hat v_{\square,t}=\mathcal{M}_\square(v_t)$까지 그래프에 얹히면 6-memory × chunk 길이만큼 곱연산.
- **조치**: (1) **chunkwise stale-snapshot을 그래프 경계로 활용**: chunk 시작 상태 $W_{\xi(t,C)}$를 **detach**해 chunk 내 inner gradient는 스냅샷 기준으로만 흐르게(=논문의 병렬화 근사 그 자체). 이게 seriality를 끊는 동시에 그래프 길이를 $C$로 상한. (2) **chunk 단위 activation checkpointing**: 각 chunk를 recompute 단위로. (3) self-target 생성이 안정되면 $\hat v$ 경로를 부분 detach하는 surrogate를 flag로(단 erikl2식 "paper-exact vs surrogate" 구분을 로그에 명시 — 무엇을 측정했는지 모호해지지 않게). (4) inner-loop를 `torch.utils.checkpoint`로 감싸 backward 시 재구성.

### X11 — chunkwise 커널 수치 불안정
- **증상**: chunkwise 경로 loss가 naive 순차와 갈라짐, 또는 log-space에서 inf/NaN. Titans momentum 항이 chunkwise closed form을 깨서(FLA가 Titans를 naive로만 둔 이유; `impl-availability.md` §2.1) 근사가 취약.
- **진단**: C1의 equivalence test(full-seq vs token-by-token)를 chunk 크기별로. 갈라지는 chunk 크기·정밀도 특정.
- **조치**: (1) inner gradient 누적·momentum·decay 곱은 **fp32 accumulation**(bf16 저장이라도). (2) `q,k` L2-normalize(논문 명시) 유지 — 정규화가 DGD Sherman-Morrison 안정성의 전제. (3) log-space impl(`fla/ops/titans` `log_impl`) 참조해 decay 곱을 log-합으로. (4) 1차 재현은 **naive+`torch.compile`로 시작**(X10 조치와 함께), Triton 커널화는 equivalence gate 통과 후에만. (5) gate($\eta_t,\alpha_t$) clamp: $\alpha_t\in[0,1]$, $\eta_t$ 상한.

### X12 — self-generated target 붕괴 / memory degenerate
- **증상**: inner memory가 상수/0으로 수렴(모든 token에 같은 출력), self-target $\hat v_{\square,t}$가 collapse, 아키텍처가 baseline만도 못함.
- **원인**: self-referential 구조가 trivial fixed point(항상 0 출력)로 빠질 수 있음. weight decay 과도, init 부적절.
- **진단**: 학습 중 inner-state 통계 로깅 — $\|W_{\square,t}\|$, $\hat v$ 분산, memory 출력 엔트로피. 붕괴는 분산→0로 보임.
- **조치**: (1) $W_{\square,\mathrm{init}}$ meta-learn을 **정상 warmup**으로 시작(§X13). (2) retention $\alpha_t$가 1에 과도하게 붙거나(갱신 안 함) 0으로 쏠리지(즉시 망각) 않게 gate producer 초기 bias. (3) weight decay($\lambda$) 축소. (4) inner-q를 static으로 두는 논문 설계 준수(§0; freeze-q가 ppl 더 낫다는 ablation) — 불안정 memory 하나 제거. (5) obekt/erikl2 oracle(§X16)로 정상 학습 시 통계 범위 대조.

### X13 — meta-learned initial state 초기 불안정
- **증상**: 학습 초반 loss가 튀거나 NaN, 특히 아직 $W_{\square,\mathrm{init}}$이 무의미할 때 inner-loop가 발산.
- **조치**: outer lr **warmup**(수백~수천 step 선형 상승), $W_{\square,\mathrm{init}}$ 작은 scale init, 초반엔 inner $\eta_t$ 상한을 낮게(gate producer bias). 논문도 meta-learned init이 "fast adaptation·stability·noise robustness에 필수"라고 명시 — 초기 단계 보호가 핵심.

### X14 — 학습 발산 / loss NaN
- **증상**: outer loss가 발산하거나 NaN.
- **진단**: grad-norm 로깅으로 spike 지점, bf16 overflow인지 inner-loop 불안정(X11/X12)인지 구분.
- **조치**: (1) **grad clipping**(global norm, 예: 1.0). (2) outer lr warmup + cosine decay. (3) gate clamp($\alpha_t,\eta_t$; X11과 공유). (4) loss/logits fp32 계산. (5) NaN이 특정 chunk 크기에서만이면 X11로. (6) 재현 대조: 같은 config에서 baseline(DeltaNet)이 안정적으로 학습되는지 먼저 확인해 인프라 vs HOPE-특유 문제 분리.

### X15 — chunk-size mismatch (train↔serve)
- **증상**: 학습 chunk 크기와 서빙(decode) chunk 크기가 다르면 품질이 열화. $C$는 **함수 자체를 바꾸는 semantic hyperparameter**(master eq. M4; FlashAttention tiling과 달리 bit-exact 아님)이므로 train/serve 불일치가 정확도에 영향(TNT Challenge 3, `impl-availability.md` §2.4).
- **조치**: 학습·평가·서빙에서 **동일 chunk 크기 $(C_{\mathrm{mem}}, C_{\mathrm{aux}})$** 사용을 기본. decode는 본질적으로 $C{=}1$ online write이므로, 학습을 작은 $C$로 하거나(품질↑, 처리율↓) chunk-consistency를 명시적으로 감사. 두 chunk 크기를 config에 고정해 전 단계 공유.

### X16 — 재현 실패 시 비공식 구현 oracle 대조 절차
- **트리거**: HOPE ppl이 baseline을 못 이김(G-1 실패), 또는 inner-state가 이상(X12).
- **절차**: (1) obekt/HOPE-nested-learning(84★) 또는 erikl2/nested-learning(76★)을 **소형·single-GPU**로 돌려 정상 학습 시 loss curve·inner-state 통계·equivalence test 값을 **참조선**으로 확보(fork·신뢰 아님, 대조만; `impl-availability.md` §5 리스크 (ii)). (2) 우리 구현과 **동일 소형 config**에서 컴포넌트별(C1→C2→C3) 수치 대조 — 어느 컴포넌트에서 갈라지는지 이분 탐색. (3) erikl2의 "paper-exact vs surrogate" 구분을 확인해, 그들이 surrogate로 안정화한 지점을 우리도 paper-exact로 재현 가능한지 판정. (4) 갈라진 컴포넌트를 §3의 unit 검증(finite-difference gradient check, equivalence test)으로 격리.

### X17 — baseline(FLA) 대조가 안 맞음
- **증상**: DeltaNet/Transformer++ baseline ppl이 FLA 공개 수치와 크게 다름 → HOPE 대조의 기준선이 무너짐.
- **조치**: baseline은 **자체 구현 금지**, FLA+flame의 검증된 모델·config 그대로. tokenizer·data pipeline·token budget이 HOPE와 100% 동일한지 감사(§2.2). baseline이 먼저 공개 수치 근처로 나와야 HOPE 순위 주장이 성립.

### X18 — 노드 부분 실패 / 재시작 (elastic 없음)
- **증상**: 2노드 중 하나가 죽으면 collective가 hang→전체 크래시. 스케줄러가 없어 자동 재스케줄 없음(D5).
- **조치**: (1) **frequent atomic checkpoint**(§X5)에서 **수동 재개** — 마지막 성공 checkpoint로 두 노드 재런칭. (2) `torchrun`에 `--max-restarts` 주더라도 2노드 static rendezvous라 노드 교체는 수동. (3) 재개 스크립트: latest checkpoint 심볼릭 → 두 노드 동일 step에서 resume 확인(step mismatch면 X3). (4) 장시간 실행은 데몬화 대신 `tee` 로그 + 주기 checkpoint로 crash-safe.

> **이 절이 문서화한 예외 시나리오 = 18개 (X1–X18).**

---

## 6. 체크리스트 · 예상 소요 · 비용 · 이월

### 6.1 실행 전 체크리스트

- [ ] 이미지 빌드·`import torch,fla` sanity 통과(§2.1), NCCL 버전 확인
- [ ] Singularity `.sif` NFS 배치(§2.1)
- [ ] HF 자산(tokenizer·FineWeb-Edu·WikiText·LAMBADA·BABILong) NFS 사전 물질화(§2.2)
- [ ] tokenizer·data pipeline이 **baseline과 동일**(§2.2, X17)
- [ ] NFS 레이아웃·권한, checkpoint atomic write 경로(§2.3, X5)
- [ ] IB 실측: `ibstat` LinkUp, `NCCL_IB_HCA`/`GID_INDEX` fabric 정답으로 교체(§4.2, X4)
- [ ] 2노드 `nc -vz $MASTER_ADDR $MASTER_PORT` 통과(X2), NTP 동기(X3)
- [ ] `--gpus all`/`--nv`·`--network host`·`--device=/dev/infiniband`·`memlock=-1`(§4.3, X1/X4)
- [ ] C1 equivalence test(chunkwise==naive), C2 finite-diff gradient check 통과(§3.1)
- [ ] FSDP fast-weight wrapping 분리 확인(§4.4, X9)
- [ ] 소형(≤130M single-GPU) 수렴·inner-state 통계 정상(X12) → 스케일 승인
- [ ] baseline(DeltaNet/Transformer++)이 FLA 공개 수치 근처(X17)

### 6.2 예상 소요 — C1–C9 주차 배분 (1인 풀타임 기준; agent-assisted 시 단축)

`impl-availability.md` §5. HOPE 재현(fast-path)은 **C1–C5 + C8 + C9**. C6/C7(Sleep)은 범위 밖(§1.4).

| # | 컴포넌트 | 공수 | 주차(누적) |
|---|---|---|---|
| C1 | deep neural memory 코어(§3.2) | 3–5일 | W1 |
| C2 | self-modifying Titans(§3.3, 최난도) | ~1주 | W2 |
| C3 | CMS(§3.4) | 1–2일 | W2 |
| C4 | HOPE block 조립 + LM 골격(§3.5) | 3일 | W3 |
| C5 | 병렬 학습 경로 naive→chunkwise(§3.6; Triton +2주 옵션) | 상(옵션 별도) | W3–W4 |
| C8 | eval harness(WikiText/LAMBADA/NIAH/BABILong) | 3–5일 | W4 |
| C9 | multi-node 실측 배선(FSDP/ckpt/예외; §4·§5) | ~1주 | W5 |
| — | **760M/30B 실제 학습 + 안정화(§5 예외 소진)** | 학습 벽시계 별도 | W6+ |

**소계(구현): 약 5주(C5 Triton 제외).** 이후 760M/30B **학습 벽시계는 실측 항목**(carry-forward #1; 논문이 wall-clock을 안 줘 사전 예측 불가) — 8장·30B·inner-loop 오버헤드로 **수 일~2주+ 규모로 예상하되 절대 예측은 X10/X8 조치 후 실측**한다. 로컬 소형 semantics 검증(C1–C4 축소판)은 2–3주로 분리 가능.

### 6.3 비용

- **GPU**: fast-path 8×A100-80G 상시 점유(760M/30B 학습 기간). 1.3B/100B로 확대 시 최대 ~32장·수 배 벽시계.
- **스토리지**: NFS에 HF 캐시(FineWeb-Edu shard 수백 GB) + checkpoint(step별 sharded, 주기적 pruning 필요 — §X5 atomic write와 별개로 오래된 ckpt 정리).
- **인적**: 구현 ~5주 + 실측·트러블슈팅. self-modifying Titans(C2)와 inner-loop 그래프(X10)가 리스크 상위.

### 6.4 이 runbook이 닫는 carry-forward 실측 (로컬 실험 → A100)

HOPE가 학습되면, `experiments/results.json`의 `carry_forward_to_A100_runbook` 6항목을 측정한다. 로컬 실험(hatir/hat-schema analytic cost model + CPU microbench)은 **ratio·crossover·tier 순서·bound class만 load-bearing**이고 **절대 µs/token·mJ/token은 roofline 하한**으로 유보했으며, novel scratchpad/PIM twin은 `simulation_ready=False` directional이다. A100에서 실제 kernel로 닫을 것:

1. **절대 per-token decode wall-clock**(ms/token, tok/s) — 논문·로컬 모두 하한만. A100에서 실 decode latency + achieved HBM BW 측정. (로컬 앵커: 1.3B RMW 6.44 GB/token, roofline 1.92 ms/token 하한 — **하한임을 명시, A100 실측으로 대체**.)
2. **achieved-vs-roofline 효율**(kernel-launch·scheduling·decode tail) — 로컬 E1.1은 per-kernel only.
3. **실 energy/token**(r/w-asymmetric DRAM) — twin은 symmetric 7 pJ/B 가정.
4. **S\* 검증**: 실 KV read kernel vs 실 TTT-state kernel head-to-head(로컬은 roofline half). 로컬 앵커 S\*_read≈65k, S\*_rmw≈131k(directional).
5. **novel-twin(scratchpad/PIM)**: silicon 전까지 directional 유지 — A100 실측은 HBM 실수치만, scratchpad/PIM 수치는 여전히 DSE.
6. **B_max under full serving stack**(weights+activations+KV co-resident) — 로컬 E3 B_max(1.3B 5.2 seq @10ms)는 upper bound.

> 이 6항목은 Part III의 pair thesis 중 **decode=memory-centric** 절반을 실측으로 뒷받침한다. runbook은 학습 재현이 1차, 이 서빙 실측이 2차 목표(§1.1). 정직성: 표에 올릴 때 로컬 유래 절대치는 항상 "roofline 하한(A100 실측 대기)"으로 태깅.

### 6.5 이월 — Dreaming/Sleep 후속 runbook 포인터

Sleep(wake-sleep consolidation, Dreaming self-improvement, periodic parameter (de)activation)은 **공개 구현 0**(`impl-availability.md` §2.6) → 세계 최초 공개 구현이 될 영역이자 이 스터디의 잠재적 기여 포인트(C6/C7, 각 1–1.5주 추정). **전제**: 이 runbook의 HOPE 재현이 성공해야 그 위에 wake-sleep cycle을 얹는다. 후속 runbook은 (i) Memory Consolidation = Knowledge Seeding 상향 distillation(small-self memory → larger network, replay buffer), (ii) Dreaming = self-generated rollout → self-modification loop + periodic fast-block 파라미터 교체/재활성를 다룬다. 리스크: Sleep 논문의 Dreaming은 수식 대비 해석 재량이 커 monograph ch17에서 해석을 먼저 고정한 뒤 구현할 것(`impl-availability.md` §5 리스크 (i)).

---

## 부록 A. HOPE hyperparameter 결정 원장 ([미명시] 항목)

논문이 고정 안 한 항목과 우리 결정·근거를 한곳에. 재현 실패 시 여기부터 sweep.

| 항목 | 논문 앵커 | 우리 시작값 / 결정 | sweep 우선순위 |
|---|---|---|---|
| CMS level 수 | 4-level | 4 | 저 |
| CMS lowest frequency $C^{(\ell)}$ | 2K sweet spot(512 최상·2K near-parity·저렴) | $\{1,64,512,2048\}$ | 중 |
| memory MLP hidden dim | 미명시(2-layer residual, expansion 4) | expansion 4, GELU | 중 |
| $C_{\mathrm{mem}}$ / $C_{\mathrm{aux}}$ chunk | 미명시(두 개 사용) | 64 / 64 | **고**(X10/X11/X15) |
| inner lr $\eta_t$ 초기·상한 | 미명시(memory 출력) | gate producer bias로 낮게 시작(X13) | 고 |
| outer lr / warmup | AdamW(값 미명시) | FLA baseline과 동일 계열, warmup 수천 step | 중 |
| tokenizer / data mix | 32K vocab, FineWeb-Edu+long-ctx(비율 미명시) | **baseline과 동일**(X17) | 고 |
| $q$ projection | static(freeze-q가 ppl 우수) | static 고정 | 저(논문 준수) |

## 부록 B. 최소 재현 커맨드 시퀀스 (요약)

```bash
# 1) 이미지·자산 (빌드/로그인 노드 1회)
docker build -t hope:cu124-torch25 /nfs/hope/env
singularity build /nfs/hope/env/hope.sif docker-daemon://hope:cu124-torch25
HF_HOME=/nfs/hope/hf_cache python /nfs/hope/code/prefetch_assets.py   # §2.2

# 2) 단위 검증 (single GPU)
python -m hope.tests.equivalence   # C1 chunkwise==naive
python -m hope.tests.gradcheck     # C2 finite-diff

# 3) baseline (대조군, FLA/flame; 동일 data)
#    두 노드 §4.3 런칭, --config configs/deltanet-760m.yaml

# 4) HOPE 760M/30B (두 노드 §4.3)
#    node0: torchrun --node_rank=0 ... train_hope.py --config configs/hope-760m.yaml
#    node1: torchrun --node_rank=1 ... (MASTER_ADDR=node0)

# 5) 평가
lm_eval --model hf --model_args pretrained=/nfs/hope/ckpt/hope-760m/final \
        --tasks lambada,wikitext,<commonsense set>
python /nfs/hope/eval/babilong/run.py --model /nfs/hope/ckpt/hope-760m/final --lengths 4k,16k,64k

# 6) carry-forward 서빙 실측 (§6.4)
python /nfs/hope/code/bench_decode.py --model hope-760m --measure ms_per_token,hbm_bw,energy
```
