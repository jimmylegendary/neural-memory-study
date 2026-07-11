# 로컬 실험 + HAT 툴 DSE 계획 (Part III 뒷받침)

> 상태: **계획 수립 완료, 코드 실행 전** (2026-07-11). 이 문서는 Part III의 (b) 로컬 host 실험,
> (c) HAT IR 기반 decode 워크로드 modeling/DSE, (d) scaling 분석의 실행 계획이다 (PLAN.md P3 참조).
> 실행 산출물은 `experiments/` 아래에, 본문 인용 수치는 각 결과 파일에서 가져온다.

## 0. 전제: 툴 능력 실사 결과 (README + 인터페이스 직접 확인)

이 호스트: GPU 없음, 16 cores, 27GB RAM, numpy 2.4.2 설치됨, torch 미설치 (CPU wheel 설치 가능).
HAT 툴 4종은 전부 pure-Python 분석 모델이라 **GPU 없이 전 실험 실행 가능**하다.

| 툴 | 이번 실험에서 쓰는 능력 (확인된 것) |
|---|---|
| **hatir** (`~/repos/hatir/impl`) | `tile(pattern, sizes, hw, dtype_bytes, macs_per_point)` einsum++ front → `derived`에 `total_backing_bytes` / `energy_pj` (r/w asymmetric: `energy_pj_per_byte_wr`) / `kernel_time_us` / `bound`(compute\|memory). `serving.op_cost`/`phase_cost`/`serve` = prefill+decode end-to-end TTFT·ms/tok·tok/s·에너지. `residency.live_set_spill_bytes` = sw_managed tier 용량 초과 시 spill cascade (largest/lru/belady). `movement.edge_cost` = 이동 비용. **핵심 선례 2개가 이미 리포에 있다**: `examples/stateful_writeback_dse.py` (write-heavy state RMW — docstring이 명시적으로 "future TTT state"를 일반화 대상으로 선언, r/w asymmetry·PIM·on-die residency 4문항) + `examples/kv_cache_memory_dse.py` (decode KV read: precision/device/GQA/context-wall 4문항). 우리 DSE는 이 두 파일의 **직접 확장**이다. |
| **hat-schema** (`~/repos/hat-schema`) | `.hw` twin SoT. 기준 twin `twins/dgx_h100_x4.json` = ib(400GB/s)→nvlink(900GB/s)→**hbm 80GB 3.35TB/s 7pJ/B (sw_managed)**→l2 50MB 10TB/s (hw_cache)→smem 228KB 33.4TB/s→reg→mxu 3.748 TMAC/s. novel twin 2종: `examples/novel-pim-cim.hw` (DRAM-bank 상주 GEMV, off-die weight movement 0) + `examples/novel-swscratchpad-npu.hw` (**scratchpad 268MB @ 60TB/s, sw_managed** — residency spill이 1차 DSE knob). `residency` 필드(hw_cache\|sw_managed)가 memory-device DSE 축. |
| **hat-kv-manager** | `KVCacheManager.from_twin(twin, kv_bytes_per_token, block_tokens)` — tier-aware placement/LRU spill/prefix reuse/hit-rate. **append-once/read-many KV 전용** 시맨틱임을 확인 (아래 E1.5의 honest gap 참조). |
| **hat-vllm-mock** | 이번 계획에서는 **비사용** (serving-loop believability는 Part III 범위 밖; E1.5에서 gap 분석의 참조점으로만 인용). |

준비 작업 (실행 시작 시 1회, ~15분): `pip install -e ~/repos/hat-schema && pip install -e ~/repos/hatir/impl`,
`cd ~/repos/hatir/impl && pytest -q` (101 tests) + `python3 examples/stateful_writeback_dse.py`로 선례 재현 확인.

## 0.5 워크로드 정의 (전 실험 공통)

**TTT decode step (Titans/Atlas/HOPE의 per-token state RMW)**, layer 하나 기준:

- state: TTT-MLP 계열 $W_1 \in \mathbb{R}^{d \times 4d}, W_2 \in \mathbb{R}^{4d \times d}$ → $8d^2$ elems; momentum $S_t$ 포함 시 $16d^2$. bf16 기준 state bytes $= m \cdot d^2 \cdot 2$, $m \in [8,16]$.
- per-token 연산: gradient용 GEMV 2–3회 ($O(d^2)$ MACs) + momentum/decay elementwise epilogue ($S_t = \eta S_{t-1} - \theta \nabla \ell$, $M_t = (1-\alpha)M_{t-1} + S_t$, state 전체 $O(md^2)$ elems).
- per-token 트래픽: state **전체를 read + write back** (RMW) → $2 \cdot m d^2 \cdot 2$ bytes. arithmetic intensity $\approx O(1)$ MAC/byte → 확정적 memory-bound.
- 구체 수치 (계획 검증용 암산): $d=2048$, $m=16$ → state 134MB/layer; $L=24$ → per-token 전 layer RMW = $2 \times 134\text{MB} \times 24 \approx 6.4$ GB/token. HBM3 3.35TB/s에서 state 이동만 ~1.9ms/token. 같은 모델 GQA-8($H_{kv} \cdot h_d = 1024$) KV read가 이 트래픽에 도달하는 context는 $S^* = m d^2 / (H_{kv} h_d) \approx 65\text{k}$ tokens. **이 두 숫자(1.9ms, 65k)가 pair thesis의 정량 골격이고, 전 실험이 이를 정밀화한다.**
- 대비 워크로드 (KV): per-token per-layer $2 H_{kv} h_d \cdot b$ bytes **append(write-once)** + context $S$ 전체 **read** — write는 미미, read는 $S$에 비례, block 단위 공유/재사용(prefix reuse) 가능. TTT state는 **unshared(sequence당 사본), write가 read와 동량, 재사용 불가** — 질적으로 다른 메모리 시스템 요구.

---

## E1. HAT 기반 DSE (hatir + hat-schema twins)

### E1.1 TTT-decode step의 baseline 비용 — DGX-H100 twin

- **목적**: Titans/Atlas/HOPE decode 한 step의 time/energy/bound를 `dgx_h100_x4.json` 위에서 산출하고, "TTT decode는 state RMW가 지배하는 memory-bound 워크로드"를 트래픽 분해로 입증.
- **방법**: `hatir.tile()`로 step을 3개 op로 조립 — (i) GEMV 계열 `"d e, e -> d"` (retrieval + gradient, $d \times 4d$, $4d \times d$ 각각), (ii) state RMW `"w -> w"` with `macs_per_point`=epilogue MAC 수 (stateful_writeback_dse.py의 `writeback()` 재사용), (iii) 합산은 `serving.op_cost` 방식(roofline_us). sweep: $d \in \{1024, 2048, 4096, 8192\}$ × $m \in \{8, 16\}$ × dtype_bytes $\in \{2, 1\}$ × $L \in \{12, 24, 48\}$. 산출: us/token, uJ/token, GEMV-vs-RMW 비율, bound 판정.
- **필요한 것**: hatir+hat-schema editable install만. 신규 스크립트 `experiments/e11_ttt_decode_baseline.py` (~150줄, stateful_writeback_dse.py 골격 복사).
- **예상 산출물**: `experiments/e11_*_results.txt` — (d, m, dtype) grid의 per-token 표 + "state RMW가 step 비용의 X%" 한 줄 결론.
- **뒷받침**: Part III pair-thesis 본문(P3-e)의 decode 쪽 절반 — "decode = per-token fast-weight state 전체의 read-modify-write" 주장의 정량 근거. ch12/ch14의 decode-cost 박스에도 인용.
- **소요**: 0.5일.

### E1.2 State 배치 DSE — HBM vs SRAM/scratchpad vs 계층, novel twin 포함

- **목적**: state를 어느 tier에 상주시키는가가 decode 비용을 얼마나 바꾸는지 — on-die residency의 crossover(몇 layer의 state까지 on-chip에 들어가나)와 PIM의 적용 구간을 수치화.
- **방법**: 세 후보 위에서 E1.1 workload 재실행 — (a) H100 twin (state는 HBM, L2는 hw_cache라 residency 모델링 대상 아님 — roofline이 처리), (b) `novel-swscratchpad-npu.hw` (268MB @ 60TB/s sw_managed: $d=2048, m=16$이면 layer당 134MB → **2 layer만 fit**, 초과분 spill → `residency.live_set_spill_bytes`로 spill bytes 산출; state를 activation-class로 태깅해야 함, 아래 위험 참조), (c) `novel-pim-cim.hw` (elementwise epilogue를 in-array로: stateful_writeback Q3의 macs/elem sweep을 TTT epilogue의 실제 macs/elem(~2–4)에 고정해 PIM energy win 확인). 추가로 twin의 scratchpad 용량을 64MB–1GB로 패치하며 fit-crossover 곡선.
- **필요한 것**: 위 install + novel twin 2종 (이미 존재). `experiments/e12_state_placement_dse.py`.
- **예상 산출물**: tier별 (traffic, uJ/token, us/token) 랭킹 표 + "scratchpad 용량 vs 수용 가능 state(layer 수)" 곡선 + PIM energy-win 표.
- **뒷받침**: Part III의 memory-centric 기회 절(P3-c) — "TTT state는 KV와 달리 크기가 context와 무관하게 고정이라 on-chip 상주가 *설계 가능한* 선택지" 논지. ch10 cost-model cheat sheet의 확장.
- **소요**: 0.5–1일.
- **⚠ 실행 가능성 주의**: `residency.py`는 weights/KV를 persistent로 보고 transient working set에서 **제외**한다. TTT state는 "persistent인데 매 token 변이"라는 제3의 클래스 — activation-class로 태깅해 우회하거나, stateful_writeback Q4처럼 capacity check를 수동으로 하는 fallback이 필요. **모듈 수정 없이도 fallback으로 결과는 나온다** (정밀도만 낮아짐: eviction policy 비교 불가).

### E1.3 KV-workload와의 정량 대비 + batch scaling

- **목적**: 같은 twin, 같은 모델 크기에서 KV-attention decode와 TTT decode의 per-token 트래픽/시간을 context $S$와 batch $B$의 함수로 나란히 놓고, crossover context $S^*$와 batch scaling의 질적 차이(둘 다 unshared지만 KV는 $S$에 비례 성장, TTT는 상수)를 산출.
- **방법**: KV 쪽은 `kv_cache_memory_dse.py`를 파라미터만 맞춰 재실행 ($S \in \{4k, 16k, 64k, 256k\}$, fp16/fp8, GQA-8). TTT 쪽은 E1.1 결과. batch는 두 쪽 모두 트래픽 선형 배수 ($B \in \{1, 8, 32, 128\}$) — aggregate 요구 BW가 HBM 3.35TB/s를 넘는 지점(= latency 목표별 B_max)을 표로. `serving.serve()`로 end-to-end 대조(dense attention 모델)도 1회 뽑아 sanity check.
- **필요한 것**: 위 install. `experiments/e13_kv_vs_ttt.py`.
- **예상 산출물**: 2-패널 표 — (i) per-token GB: KV(S 증가) vs TTT(상수) 교차점 $S^*$ (config별), (ii) B_max @ {10ms, 50ms}/token 목표. "KV cache는 append-once/read-many, TTT state는 RMW/write-heavy/unshared"의 트래픽 성분 분해(read vs write bytes).
- **뒷받침**: pair thesis의 핵심 정량 절 — Part III 도입부의 "왜 decode/serving state 관리가 memory-centric 기회인가". ch01 Rosetta-Stone 표의 정량 후속.
- **소요**: 0.5일.

### E1.4 Frequency-tiered placement — CMS 레벨별 DDR/CXL 허용 DSE

- **목적**: Nested Learning(HOPE)/Sleep의 update-frequency 연속체(레벨 $k$는 $C^k$ token마다 update)를 **메모리 계층 배치 사양으로 직역** — "update 주기가 $P$ token인 state는 BW $\geq 2 S_k / (P \cdot t_{tok})$인 가장 싼 tier에 두면 critical path를 늘리지 않는다"를 twin 위에서 검증.
- **방법**: dgx_h100_x4 twin에 outer tier 2개를 패치로 추가 — DDR5(~0.4TB/s, ~12pJ/B)와 CXL pool(0.5TB/s, 15pJ/B; `memory_device_dse.py` Q3 수치 재사용) — 한 twin의 hbm→ddr→cxl cascade. 레벨별 state $S_k$와 주기 $P_k \in \{1, 64, 4096, 256\text{k}\}$에 대해 update 트래픽을 주기로 나눈 유효 BW 요구를 산출하고, 각 tier에 배치했을 때의 per-token 추가 시간(`tile("w -> w")` @ 해당 tier BW / P로 상각)을 표로. Sleep의 offline consolidation은 "P → ∞(idle window)" 행으로 포함.
- **필요한 것**: 위 install. twin 패치는 JSON 사본 (`experiments/twins/dgx_h100_x4_tiered.json` — schema는 additive라 안전). `experiments/e14_freq_tiered_placement.py`.
- **예상 산출물**: **placement 표** (행=CMS 레벨/update 주기, 열=tier, 셀=per-token 상각 비용과 허용 여부) — Part III에 그대로 실리는 그림 1장감.
- **뒷받침**: Part III의 "update-frequency 연속체 = memory-hierarchy 사양" 절 (ch16 Nested Learning + ch17 Sleep의 시스템 귀결). P3-d scaling projection의 입력.
- **소요**: 0.5일.

### E1.5 hat-kv-manager gap 분석 (실험 아님, 서술)

- **목적**: "KV manager의 시맨틱(append/prefix-reuse/LRU spill)이 TTT state에 왜 안 맞는가"를 코드 레벨 근거로 서술 — pair thesis의 시스템-소프트웨어 측 논거.
- **방법**: `kv_manager/manager.py` 인터페이스 대조 서술만: `place()`(append-once) / `prefix_reuse()`(content-addressed, TTT state는 content-addressing 불가 — 매 token 변이) / `_evict_from()`(clean eviction 가정, TTT state는 항상 dirty → writeback 필수) / checkpoint·rollback 부재. "TTT-state manager가 필요로 하는 새 이벤트 타입" 목록 도출.
- **예상 산출물**: 본문 1–2p 분량 서술 + (부산물) hat-kv-manager 로드맵 메모.
- **뒷받침**: pair thesis 절의 소프트웨어 스택 문단; ch10 systems bridge.
- **소요**: 0.25일. **실행 리스크 없음** (코드 읽기만).

---

## E2. CPU 마이크로벤치 (numpy, roofline 검증용 소규모 데이터 포인트)

### E2.1 chunkwise scan vs sequential update — arithmetic intensity 실측

- **목적**: ch09의 "chunk size는 semantic hyperparameter이면서 동시에 roofline의 x축"을 실측으로 뒷받침 — chunk $C$가 커질수록 유효 intensity(측정 GFLOP/s)가 오르는 **알고리즘적 곡선**은 하드웨어 불문이고, CPU에서도 그 형태가 재현됨을 보인다.
- **방법**: numpy fp32, delta-rule/linear-attention 꼴 최소 구현 — (a) sequential: token마다 $S \mathrel{+}= v_t k_t^\top$ (rank-1 RMW) + $o_t = S q_t$ (GEMV); (b) chunkwise: $C$ token을 GEMM 2–3개($K_c^\top V_c$, intra-chunk attn)로 처리 후 state update 1회. $C \in \{1,4,16,64,256,1024\}$, $d \in \{512, 1024, 2048\}$, $T=8192$. FLOPs/bytes는 해석적으로 계산, wall time으로 나눠 유효 GFLOP/s → intensity 곡선. CPU roofline 캘리브레이션: 대형 `np.dot` 1회(peak GFLOP/s)와 STREAM-triad 꼴 `x += a*y`(peak GB/s) 실측으로 ridge point를 이 호스트 좌표에 놓는다.
- **필요한 것**: numpy(있음)만. in-place ops(`out=`, `+=`)로 numpy 임시 배열 억제. BLAS thread 고정(`OMP_NUM_THREADS=16`), 반복 중앙값. `experiments/e21_chunkwise_intensity.py`.
- **예상 산출물**: CSV + 곡선 데이터 — "측정 GFLOP/s vs $C$" (d별), 이 호스트의 ridge point, sequential($C=1$)이 memory-bound 평탄역에 붙는 모습. ch09/ch10의 roofline-vs-chunk 그림에 실측 점으로 삽입.
- **뒷받침**: ch09 §chunk-size semantics의 실측 보강 + Part III prefill/training 쪽 절반("chunk를 키우면 accelerator 영역으로 넘어간다")의 검증 데이터.
- **소요**: 0.5일 (스크립트 0.25 + 측정/정리 0.25).
- **⚠ 정직 표시**: (i) CPU 측정은 **H100 절대치를 대변하지 않는다** — 곡선의 형태(intensity가 $C$에 어떻게 비례하나)만 이식 가능하고, 본문에도 그렇게만 쓴다. (ii) numpy BLAS의 GEMV는 single-thread로 떨어질 수 있어 $C=1$ 점이 thread 효과와 교락됨 — thread 1 고정 세트를 별도 측정해 분리. (iii) 실제 DRAM 트래픽은 `perf stat`(LLC-misses) 가용 시에만 검증 — sandbox에서 perf가 막히면 해석적 bytes로 대체하고 그 사실을 결과에 명기.

### E2.2 state 크기별 RMW 대역폭 — cache-cliff 측정

- **목적**: "state가 on-chip(cache)에 들어가는 동안은 RMW가 싸고, 넘치면 DRAM BW로 떨어진다"는 E1.2 residency crossover의 로컬 아날로그를 실측 — 계층 경계에서 유효 BW가 계단식으로 떨어지는 것을 보인다.
- **방법**: in-place RMW kernel `x *= a; x += b` (read+write 2×)를 state 크기 1MB→8GB sweep, GB/s vs size 곡선. `lscpu`로 L2/LLC 용량 기록, 곡선의 cliff 위치와 대조. single-thread와 16-thread 두 세트.
- **필요한 것**: numpy만. `experiments/e22_rmw_bandwidth_cliff.py`.
- **예상 산출물**: CSV — size vs GB/s, cliff 주석 (L2, LLC, DRAM 평탄역). E1.2의 "scratchpad fit → spill" 서사에 붙는 실측 삽화.
- **뒷받침**: ch10 cost-model cheat sheet(메모리 계층은 용량-경계에서 불연속) + Part III state 배치 절.
- **소요**: 0.25일.

### E2.3 (선택) torch-CPU 교차 검증

- **목적**: TTT inner-loop을 autograd로 실제 실행해 forward/backward FLOP 비율(≈1:2)과 E2.1 numpy 손계산의 일치 확인.
- **방법**: `pip install torch --index-url .../cpu` (~200MB), TTT-Linear 1-layer inner loop, `torch.profiler`로 op별 시간.
- **⚠ 정직 표시**: **기대 이득이 작다** — 대역폭/intensity 결론은 E2.1로 충분하고, torch 설치는 27GB RAM 호스트에 부담은 아니지만 시간 대비 본문 기여가 한 문단(“backward 포함 시에도 memory-bound 결론 불변”) 수준. **기본 계획에서 제외, Part I ch02 집필 중 backward 비용 수치가 필요해지면 그때 실행.**
- **소요**: 0.5일 (실행하는 경우).

---

## E3. 분석 모델 — 논문 공개 수치 기반 crossover 계산기

### E3.1 파라미터화된 비용 모델 + crossover 표

- **목적**: 실험 없이도 본문 전체에서 재사용할 닫힌꼴 비용 모델 하나를 만들어, 세 crossover를 config별로 산출 — (i) $S^*$: TTT per-token 트래픽이 KV read와 같아지는 context, (ii) $C^*$: chunkwise intensity가 target HW ridge를 넘는 chunk (H100: ridge ≈ 990 TFLOPs / 3.35 TB/s ≈ 295 FLOP/B), (iii) $B_{max}$: latency 목표 하에서 aggregate state RMW가 HBM BW를 포화시키는 batch.
- **방법**: "스프레드시트"는 실제로는 **Python 스크립트 + CSV/Markdown 표 출력**으로 구현 (버전 관리·재현성 우선; 필요 시 CSV를 그대로 스프레드시트로 열면 됨). 파라미터: $d$, $L$, $H_{kv} h_d$, $m$(state multiplier 8–16), dtype bytes, $C$, $B$, $S$, HW($BW_{hbm}$, peak FLOPs — dgx_h100_x4 twin 값을 SoT로 import해 E1과 좌표 일치). 입력 config: 6편 논문의 공개 아키텍처 수치(Titans 170M–760M, Atlas, TNT의 chunk 설정, HOPE 레벨 구조 — `notes/25*.json` dossier에서 추출) + 가상 7B/70B scaling 행.
- **필요한 것**: 논문 dossier(있음), twin JSON(있음). `experiments/e31_analytic_model.py` + `experiments/cost-model.csv`.
- **예상 산출물**: config × {per-token GB, us(모델), $S^*$, $C^*$, $B_{max}$} 표 — Part III scaling projection 절의 골격이자 ch10 cheat sheet의 수치 원천. E1(HAT)·E2(CPU) 결과와 3-way 상호 검증 문단.
- **뒷받침**: P3-d scaling 분석·projection 절 전체 + ch09 $C^*$ 논의 + ch15 TNT(chunkwise training 개선)의 비용 프레임.
- **소요**: 0.5일.
- **⚠ 정직 표시**: 6편 논문 대부분 **H100 wall-clock decode 수치를 공개하지 않는다** — 이 모델의 절대치는 외부 검증 불가이며, 본문에서는 비율(ratio)과 crossover 위치만 주장한다. 절대치 검증은 P3-(a) 사내 A100 runbook 실측의 몫으로 명시적으로 이월.

---

## 4. 실행 순서와 총 소요

| 순서 | 항목 | 소요 | 의존성 |
|---|---|---|---|
| 1 | 설치 + 선례 재현 (stateful_writeback, kv_cache_memory) | 0.25일 | — |
| 2 | E3.1 분석 모델 (먼저: 이후 실험의 기대값 제공) | 0.5일 | — |
| 3 | E1.1 → E1.3 (baseline → KV 대비) | 1일 | 1 |
| 4 | E1.2 (배치 DSE, novel twins) | 0.5–1일 | 3 |
| 5 | E1.4 (frequency-tiered placement) | 0.5일 | 3 |
| 6 | E2.1 + E2.2 (CPU 마이크로벤치) | 0.75일 | — (병행 가능) |
| 7 | E1.5 (kv-manager gap 서술) | 0.25일 | — |
| | **합계** | **약 4–4.5일** | |

## 5. 전 항목 공통의 정직한 한계 (본문에도 그대로 명기할 것)

1. **hatir 수치는 exploration-grade/relative** — 리포 스스로 "ranking + crossover용, silicon-accurate 절대치 아님"을 선언한다 (stateful_writeback_dse.py docstring). 우리는 랭킹·crossover·비율만 본문 주장으로 승격한다.
2. **novel twin 2종은 verify committee가 simulation_ready=False / PPA abstain 판정** (CATALOG.md) — PIM/scratchpad 결과는 "방향성 DSE"로만 인용, 실존 디바이스 주장 금지.
3. **hatir는 per-kernel 분석 모델** — decode step의 kernel-launch/스케줄링 오버헤드, tail effect는 미포함. per-token 절대 latency가 아니라 트래픽·에너지 구조를 본다.
4. **residency 모듈의 persistent-state 공백** (E1.2 참조) — fallback 경로로 결과는 확보되나 eviction-policy 수준 정밀도는 포기.
5. **CPU 실측은 형태 검증** — H100으로의 수치 이식은 하지 않는다 (ridge 좌표가 ~100배 다름).
6. 실행 단계에서 twin JSON을 패치할 때는 **원본 불변** (사본을 `experiments/twins/`에) — hat-schema가 SoT라는 제품 라인 규율 유지.
