# Part III 실험 — 환경·재현·신뢰성 증명

> 이 문서의 목적은 하나다: 본서 Part III의 "저자 자체 실험" 8개가 **무엇으로 어디서 어떻게** 나왔는지를
> 남이 그대로 재현할 수 있게 못 박고, 각 숫자가 **어느 등급의 증거인지**를 숨김없이 등급화하는 것.
> 결론을 먼저 말하면 — 이 실험들은 **silicon 측정이 아니다.** 실제 GPU wall-clock은 한 줄도 없다(이 host엔 NVIDIA GPU가 없다).
> 신뢰성의 근거는 "측정했다"가 아니라 **(1) 결정론적 재현, (2) 3중 교차검증, (3) host CPU 실측으로 roofline *모양* 확인,
> (4) 모든 숫자의 정직한 등급화 + 반증 목록**이다. 아래는 그 넷을 각각 증명한다.

---

## 0. 한 눈에 — 무엇을 신뢰하고 무엇을 신뢰하지 말 것인가

| 신뢰해도 되는 것 (load-bearing, 본문 승격) | 신뢰하면 안 되는 것 (이월/directional) |
|---|---|
| **비율** (write/read %, traffic 대 compute 배수) | 절대 µs/token, mJ/token (= roofline **하한**) |
| **crossover 위치** (S\*, C\*, d\*, B_max) | novel twin(scratchpad/PIM) 절대치 (`simulation_ready=False`) |
| **tier 순서** (cadence → HBM/CXL/scratchpad 배정) | E1.2 Q2b `spill_MB_per_token` (directional sim) |
| **bound 분류** (memory-bound / compute-bound) | E2.x host GB/s·GFLOPS 절대치 (**모양만** 이송) |

이 표가 "정직성 계약"이다. 본서 본문은 왼쪽 열만 단정문으로 쓰고, 오른쪽 열은 전부 "하한/방향성/사내 A100 runbook 이월"로 표기한다.

---

## 1. 실험 환경 지문 (Environment fingerprint)

이 실험들이 실행된 정확한 환경. 재현 시 아래와 다르면 **E2.x host 실측치는 당연히 달라진다**(그게 정상이다, §3 참조).

| 항목 | 값 |
|---|---|
| repo commit | `neural-memory-study` @ `d709014` (2026-07-14) |
| Python | **3.14.3** (venv: `./.venv`) |
| numpy / scipy / matplotlib | **2.5.1 / 1.18.0 / 3.11.0** |
| `hatir` (cost model) | editable `~/repos/hatir/impl/hatir` @ `10aa2a2` (2026-07-11) |
| `hat_schema` (device twin SoT) | editable `~/repos/hat-schema/hat_schema` @ `c8b8913` (2026-07-10) |
| **device twin** | `~/repos/hat-schema/twins/dgx_h100_x4.json` — sha256 `5cb5fe1e42972a30…` |
| host CPU | AMD Ryzen 7 255 (Radeon 780M), 16 core |
| host RAM | 27 GiB |
| **host GPU** | **없음** (NVIDIA GPU 부재 → 실제 H100/A100 측정 불가) |
| 실행일 | 2026-07-12 (재현·결정론 검증 2026-07-14) |

**device twin이 무엇인가.** `dgx_h100_x4.json`은 hat-schema가 SoT로 관리하는 **pre-silicon 장치 모델**이다 —
DGX-H100(×4)의 HBM3 대역폭·용량·에너지 계수와 compute roofline(ridge 295 FLOP/B)을 기술한 JSON.
E1.x는 이 twin 위에서 `hatir`로 roofline을 계산하고, E3는 같은 twin의 상수를 닫힌 형태 공식에 넣는다.
**즉 "H100에서 쟀다"가 아니라 "H100 twin의 물리로 roofline 하한을 계산했다"**가 정확한 표현이다.

---

## 2. 재현 절차 (Reproduction)

전제: 위 세 repo(`neural-memory-study`, `hatir`, `hat-schema`)를 위 commit으로 두고, `neural-memory-study/.venv` 활성화.

```bash
cd ~/repos/neural-memory-study
PY=.venv/bin/python

# 8개 실험 — 반드시 PYTHONHASHSEED=0 (analytical 실험을 bit-deterministic으로 고정; §3 참조)
for d in E1.1-decode-baseline E1.2-state-placement E1.3-kv-vs-ttt E1.4-frequency-tiers \
         E1.5-kvmgr-gap E2.1-chunk-intensity E2.2-rmw-cliff E3-analytical E4-scaling; do
  ( cd experiments/$d && PYTHONHASHSEED=0 ../../$PY run.py )
done

# 8개 결과를 per-number provenance + caveat 태그로 통합
( cd experiments && PYTHONHASHSEED=0 $PY consolidate.py )   # -> experiments/results.json

# 그림 5장 재생성
$PY figures/render_experiments.py                            # -> figures/exp-*.png
```

산출물: 각 `experiments/<exp>/{results.json, stdout.txt}` + 통합 `experiments/results.json`(machine-readable warrant) +
서술 `experiments/RESULTS.md` + `figures/exp-*.png`.

---

## 3. 결정론·재현성 증명 (실측 2026-07-14)

8개를 재실행하고 저장본과 **bit-diff**한 결과 — 재현성의 패턴이 정확히 실험의 성격과 일치한다:

| 실험 | 종류 | 재실행 시 | 왜 |
|---|---|---|---|
| E1.1, E1.3, E1.4, E1.5, E3, E4 | HATIR/closed-form (analytical) | **bit-identical** ✅ | 순수 계산, 입력 고정 → 결정론적 |
| E1.2 (Q1/Q2 crossover, energy/time) | HATIR analytical | **bit-identical** ✅ | 위와 동일 |
| **E1.2 (Q2b `spill_MB_per_token`)** | HATIR spill **sim** | seed 없으면 흔들림 → **`PYTHONHASHSEED=0`으로 고정** | 아래 ⚠️ |
| **E2.1, E2.2** | **host CPU 실측** | **run마다 변동** (정상) | 진짜 wall-clock → timing jitter |

**E2.x가 흔들리는 건 결함이 아니라 정직성의 증거다.** 이들은 host CPU에서 실제 시간을 재므로 당연히 jitter가 있다.
그래서 본서는 이들의 절대 GB/s·GFLOPS를 **절대 이송하지 않고** "곡선 *모양*"(AI(C) 상승, RMW 대역폭 절벽)만 본문에 올린다.

**⚠️ 유일하게 손봐야 했던 비결정성 — 발견·격리·고정.**
E1.2의 Q2b whole-model spill sim은 `hatir.live_set_spill_bytes(..., policy="largest")`를 쓴다.
그 함수(`hatir/impl/hatir/residency.py:82`)의 eviction 대상 선택 `max(live, key=size)`에서 `live`가 `set`이라,
**같은 크기 텐서의 tie-break이 문자열 hash 랜덤화(`PYTHONHASHSEED`)에 의존하는 set 순회 순서로 갈린다.**
그 결과 spill 총량이 run마다 요동친다(예: 570.4 vs 637.5 MB/token). 검증:

- `PYTHONHASHSEED=0` 고정 → 2회 재실행 spill = `{0.0, 536.9, 1962.9, 4429.2, 22548.6, 193273.5}` **완전 동일**
- seed 미지정 → 매번 다른 draw

이 필드는 **load-bearing이 아니다** — 본서는 E1.2에서 crossover d\*(≈2896, 결정론적), scratchpad의 6.8× 에너지/14.9× 시간 이득,
"규모가 커지면 per-layer/streamed"라는 *정성적* 결론만 인용하고, spill의 절대 MB 값은 인용하지 않는다.
그럼에도 재현성을 위해 (a) 재현 절차에 `PYTHONHASHSEED=0`을 못 박아 이 필드까지 bit-deterministic으로 고정했고,
(b) hatir의 해당 docstring이 스스로 `'largest' (default; byte-identical)`이라 주장하므로,
tie-break을 안정 정렬(2차 키=tensor id)로 바꾸는 것을 **hatir upstream 권고 사항**으로 남긴다(이 repo는 seed 고정으로 이미 해결).

---

## 4. 교차검증 — 이것이 신뢰성의 중심 기둥 (3중 <1% 일치)

silicon 측정이 없으므로, 신뢰성은 "**서로 독립인 계산 경로들이 같은 답에 수렴하는가**"로 세운다.
Anchor 설정 `neural-mem-1.3B (d=2048, m=16, L=24, GQA-8 kv_dim=1024, bf16)`에서
세 방법 — **닫힌 형태 공식(E3)**, **hatir 도구(E1.1/E1.3)**, **plan 손계산** — 이 1% 이내로 일치:

| 양 | E3 (closed-form) | hatir (E1.1/E1.3) | plan (hand-calc) | 일치 |
|---|---|---|---|---|
| RMW GB/token | 6.442 | 6.4425 | 6.4 | ✅ |
| RMW ms/token | 1.923 | 1.9231 | 1.9 | ✅ |
| S\*_read (tokens) | 65536 | 65471 | 65000 | ✅ |
| state MB/layer | 134.2 | 134.218 | 134 | ✅ |
| B_max @10ms | 5.2 | 5.2 | — | ✅ |

**이 교차검증이 증명하는 것과 못 하는 것을 정확히.**
✅ 증명: 세 코드 경로 어디에도 **구현 버그가 없다** — 같은 roofline 물리를 서로 다른 방식으로 계산해 같은 값을 낸다.
❌ 못 함: twin의 HBM BW(2 TB/s)·에너지 계수가 **실제 H100 silicon과 얼마나 맞는지**는 이 일치로 증명되지 않는다
(E3와 hatir는 같은 twin BW를 *입력*으로 공유하므로). 그 fidelity gap이 바로 §5의 "roofline 하한 → A100 runbook 이월" 이유다.
즉 교차검증은 **"우리 계산이 옳다"**를 증명하고, twin fidelity는 별도로 A100 실측으로 메운다.

C\*는 ridge 의존이라 값이 아니라 **모양**만 이송된다: host CPU에서 C\*≈32(discrete), H100 twin에서 C\*≈337 —
CPU ridge가 H100보다 ~9× 낮아 값은 다르되 "작은 C=memory-bound → 큰 C=compute-bound" 곡선 모양은 동일(E2.1이 실측으로 확인).

---

## 5. 명시적으로 주장하지 않는 것 — 사내 A100 runbook 이월 목록

본서가 **의도적으로 주장하지 않고** `runbook/hope-reproduction-a100.md`로 넘긴 6가지(=반증·검증 목록):

1. 절대 per-token decode wall-clock(ms/token, tok/s) — 원 논문 6편도 하나도 안 냄.
2. RMW step의 achieved-vs-roofline 효율(kernel-launch/scheduling/tail 오버헤드).
3. 실제 r/w 비대칭 DRAM 에너지/token (twin은 대칭 7 pJ/B 가정).
4. S\* 검증 — 진짜 KV 커널 vs 진짜 TTT-state 커널 head-to-head.
5. novel twin(scratchpad/PIM) 절대치 — silicon 전까지 directional.
6. 전체 serving 스택(weights+activations+KV 공존) 하의 B_max — E3의 B_max는 상한.

이 목록이 있다는 것 자체가 신뢰성의 일부다: **무엇이 아직 증명 안 됐는지를 이 문서가 먼저 밝힌다.**

---

## 6. 8개 실험 한 줄 요약 (→ 본서 연결 지점)

| 실험 | 무엇 | 방법 | → 본서 |
|---|---|---|---|
| E1.1 | decode = whole-state RMW, memory-bound | hatir @ twin | ch12/14/19 |
| E1.2 | state placement DSE (HBM/SRAM/scratchpad/PIM), 잔류 crossover d\* | hatir @ twin | ch20/24, ch10 |
| E1.3 | KV vs TTT traffic 대비 + S\* crossover + B_max | hatir @ twin | Part III intro, ch01 |
| E1.4 | update cadence → memory tier 배정 | hatir @ twin | ch19/24, ch16/17 |
| E1.5 | KV-manager 의미론이 RMW state에 category mismatch | hatir + hat-kv-manager 코드 | pair-thesis SW 문단 |
| E2.1 | chunk C = roofline x축 (AI(C) 모양) | **host CPU numpy 실측** | prefill/training 절, ch09/15 |
| E2.2 | on-die/off-die RMW 대역폭 절벽 | **host CPU numpy 실측** | ch20, ch10 |
| E3 | 닫힌 형태 cost model + crossover(S\*,C\*,B_max) × 6논문×5규모 | closed-form | Part III 분석 백본 |
| E4 | scaling (state·traffic ∝ m·d²·L) 170M→70B | closed-form + hatir | ch19 scaling |

---

## 7. 검증 도우미

```bash
# 결정론 재확인 — analytical 실험은 재실행해도 git diff가 비어야 함
for d in E1.1-decode-baseline E1.3-kv-vs-ttt E1.4-frequency-tiers E1.5-kvmgr-gap E3-analytical E4-scaling; do
  ( cd experiments/$d && PYTHONHASHSEED=0 ../../.venv/bin/python run.py >/dev/null 2>&1 )
done
git diff --stat -- experiments/   # analytical 6개 + E1.2(seed 고정) → 변화 없음이 기대값
                                  # E2.1/E2.2만 재실행 시 host-timing으로 변동(정상)
```
