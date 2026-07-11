# 그림 스펙 (figures/SPEC.md) — 단일 SoT

**지위**: STYLE-NOTATION v1.1 §5.1이 지정한 그림 스펙의 단일 SoT. 생성(작도)은 후속 figure pass 라운드에서 하며, P2 보정 라운드(W2)의 각 장 fixer는 **본문에 `<!-- FIG: chNN/fig-MM-slug -->` 주석만 삽입**한다(fixplan의 위치 지시를 따를 것). 주석 삽입 위치 = 최종 그림 삽입 위치.

**공통 규격**

- 파일: `study-kr/figures/chNN/fig-MM-<slug>.svg` (STYLE §5.1). 벡터 SVG, 폭 기준 900px 안팎, 빌드 폭 제한 고려.
- 본문 참조: "그림 NN-MM" + 캡션 필수(STYLE §5.4). 원 논문 그림 재작도 시 "([X Fig. n] 재구성)" 표기, 원본 복사 금지.
- 텍스트: 라벨은 로마자 기술 용어(STYLE §2.1), 설명 구는 한국어. 수식 라벨은 통일 표기만.
- 재사용 규칙: "draw once, reuse everywhere" 대상 그림(F-09)은 소유 장에만 이미지가 들어가고, 재사용 장은 "(그림 9-2, → 9장)" 텍스트 참조만 한다. 이미지 중복 삽입 금지.
- 우선순위: **P0** = 커리큘럼(prereq-curriculum B-모듈)이 명시 요구 — figure pass에서 필수. **P1** = 감사·fixplan이 필요로 하는 고가치 그림. **P2** = 여유 시.

| ID | 소유 장 | 우선순위 | 근거 |
|---|---|---|---|
| F-01 ch01/fig-01-two-loops | ch01 | P1 | ch01 감축 후 지면 절약(산문 대체) |
| F-02 ch02/fig-01-loss-landscape | ch02 | **P0** | B1 "one figure's worth" |
| F-03 ch03/fig-01-loss-geometries | ch03 | **P0** | B2 "one figure of loss shapes" |
| F-04 ch05/fig-01-capacity-chain | ch05 | P1 | continuity F-02 보정 지원 |
| F-05 ch06/fig-01-fwp-timeline | ch06 | **P0** | B5 "proper timeline figure" |
| F-06 ch07/fig-01-ssd-three-paths | ch07 | **P0** | B5b "1–2 figures" |
| F-07 ch07/fig-02-selectivity-gate | ch07 | P2 | B5b 두 번째 슬롯(선택) |
| F-08 ch08/fig-01-ttt-layer | ch08 | P1 | decode 속 backward의 시각화 |
| F-09 ch09/fig-02-three-regimes | ch09 | **P0** | B7 "three-regime diagram (draw once, reuse everywhere)" |
| F-10 ch09/fig-01-chunkwise-dataflow | ch09 | P1 | (M4) stale-snapshot의 시각화 |
| F-11 ch10/fig-01-state-budget | ch10 | P2 | ch10 감축 지원(표 시각화) |
| F-12 ch12/fig-01-mac-dataflow | ch12 | P1 | STYLE §5.4의 예시로 이미 예약된 ID |
| F-13 ch13/fig-01-design-space | ch13 | P2 | 4축 설계 공간 지도 |
| F-14 ch14/fig-01-omega-window | ch14 | P1 | Omega rule window·banded mask |
| F-15 ch15/fig-01-tnt-hierarchy | ch15 | P1 | global/local·reset·shard 병렬화 |
| F-16 ch16/fig-01-cms-spectrum | ch16 | P1 | update-frequency 연속체(라인 종합) |
| F-17 ch17/fig-01-wake-sleep-lifecycle | ch17 | P1 | 세 번째 regime의 좌표화 |

---

## F-01 ch01/fig-01-two-loops (P1)

- **내용 스케치**: 좌우 2패널. 좌 = 독자의 세계(inference: weights 동결, KV cache append→lookup, prefill/decode 화살표). 우 = 이 라인의 세계(inner loop: $W_{t-1}\to W_t$ RMW 사이클 + read $y_t=\mathcal{M}(q_t;W_t)$; outer loop: $\Theta$가 게이트 $\eta_t,\beta_t,\alpha_t$를 공급하는 바깥 고리). 중앙에 (M) 수식을 배치하고 각 기호에서 두 패널로 연결선. "폐기되는 불변식: weight update 없음"을 좌→우 전환 라벨로.
- **삽입 위치**: ch01 §1.4 언저리, 표 1-1(Rosetta) 직전. `<!-- FIG: ch01/fig-01-two-loops -->`
- **형식 제안**: 2-column 블록 다이어그램, 화살표 + 수식 라벨. 회색조 + 강조 1색.

## F-02 ch02/fig-01-loss-landscape (P0 — B1 요구)

- **내용 스케치**: loss surface 등고선(2D) 위에 (a) SGD 지그재그 경로, (b) momentum 경로(관성으로 계곡 진동 감쇠), (c) 곡률 축 스케일링(AdamW류) 경로 3개 병치. 인셋으로 1D 단면(계곡/평지/절벽). 라벨: "loss surface는 지도, optimizer는 주행 정책".
- **삽입 위치**: ch02 loss surface 문단(§2.3 부근, "지도" 비유 도입 직후). `<!-- FIG: ch02/fig-01-loss-landscape -->`
- **형식 제안**: 등고선 + 경로 곡선. matplotlib 생성 후 SVG 정리 가능.

## F-03 ch03/fig-01-loss-geometries (P0 — B2 요구)

- **내용 스케치**: 1D 잔차 $r$ 대비 손실값 곡선 4개 오버레이 — $\ell_2$( $r^2$ ), $\ell_1$( $|r|$ ), $\ell_p$ ($1<p<2$ 한 개), Huber(threshold $\delta_t$ 표시, 이차→선형 전환점 강조). 우측에 gradient 곡선(민감도) 소패널: outlier 한 개가 $\ell_2$ gradient를 지배하는 것 vs Huber의 포화. 라벨: "attentional bias의 선택 = write의 outlier 정책"(→ 13장 예고).
- **삽입 위치**: ch03 loss geometry 표(§3.7) 직전 또는 직후. `<!-- FIG: ch03/fig-01-loss-geometries -->`
- **형식 제안**: 2패널 함수 그래프.

## F-04 ch05/fig-01-capacity-chain (P1)

- **내용 스케치**: 왼→오 4단 사슬: 고전 Hopfield($F(z)=z^2/2$, capacity $0.14d$) → dense($F=z^n$, $d^{n-1}$) → exponential($F=\exp$, $e^{O(d)}$, Ramsauer) → softmax attention(= 1-step retrieval). 각 단 아래에 feature-map 해석($\phi_n$ lift 차원 증가)과 "지불 통화: state byte·FLOP" 라벨. 마지막 단에 "[Atlas §3.1] Prop 1/2, $\phi^*$"로 이어지는 화살표(→ 14장).
- **삽입 위치**: ch05 §5.4 사슬 요약 문장("사슬을 한 줄로 요약한다") 직전. `<!-- FIG: ch05/fig-01-capacity-chain -->`
- **형식 제안**: 수평 단계 다이어그램. ch14 §14.3.1의 콜백(fixplan D1)이 이 그림 번호를 텍스트 참조.

## F-05 ch06/fig-01-fwp-timeline (P0 — B5 "proper timeline figure" 요구)

- **내용 스케치**: 수평 타임라인 1992→2025: Schmidhuber FWP(1992) — linear attention(2020) — FWP 동치 발견/DeltaNet(2021) — RetNet/GLA(2023) — Mamba-2/Longhorn(2024) — Gated DeltaNet·RWKV-7(2024–25) — TTT 계열 합류점(→ 8장). 각 노드에 한 줄 update rule(§1.6 카탈로그 표기)과 "무엇이 추가됐나"(decay→data-dep. gate→delta→retention→벡터 gate) 배지. 두 갈래(압축 관점 / FWP 관점)가 2021에 합류하는 구조를 명시.
- **삽입 위치**: ch06 §6.2 timeline 표 자리(표는 그림의 보조로 축약 유지 가능). `<!-- FIG: ch06/fig-01-fwp-timeline -->`
- **형식 제안**: 타임라인 + 배지. 폭이 넓으므로 2단 접이(1992–2021 / 2021–2025) 허용.

## F-06 ch07/fig-01-ssd-three-paths (P0 — B5b 요구)

- **내용 스케치**: Mamba-2 SSD의 동일 계산에 대한 3경로 — (a) recurrent(순차 scan, $O(L)$ 단계), (b) quadratic(masked attention 형태, semiseparable mask 시각화), (c) chunkwise(블록 대각 = intra-chunk GEMM + 블록 하삼각 = inter-chunk state 전달) — 를 같은 $L\times L$ 행렬 그림 위에 3분할로 표시. "3경로 = 같은 함수, 다른 스케줄(exact)" vs "(M4)의 chunk는 함수가 바뀜(semantic)" 대비 라벨(→ 9장 예고).
- **삽입 위치**: ch07 §7.5 3경로 손계산 직전. `<!-- FIG: ch07/fig-01-ssd-three-paths -->`
- **형식 제안**: 행렬 블록 색칠 다이어그램 3개 병치.

## F-07 ch07/fig-02-selectivity-gate (P2 — B5b 두 번째 슬롯)

- **내용 스케치**: S4(LTI, conv 모드 가능) vs Mamba(selective, conv 모드 소멸)의 대비: 입력열 위에 gate가 열리고 닫히는 히트맵, "저장할 token을 고른다 = eviction의 학습" 라벨.
- **삽입 위치**: ch07 Mamba selectivity 절. `<!-- FIG: ch07/fig-02-selectivity-gate -->`
- **형식 제안**: 히트맵 + 화살표. 여유 시에만.

## F-08 ch08/fig-01-ttt-layer (P1)

- **내용 스케치**: TTT layer 해부도: token stream이 layer를 통과할 때 layer 내부에서 (i) inner loss $\ell(W;k_t,v_t)$ 평가, (ii) backward(층내 국소), (iii) $W$ update, (iv) read $y_t$가 한 forward 안에서 도는 순환 고리. 바깥에 "decode 안에 backward가 들어온다"를 프레임으로: 독자의 decode 파이프라인 그림(기존 불변식) 위에 새 박스가 삽입되는 형태.
- **삽입 위치**: ch08 §8.3 TTT layer 3요소 정리 직후. `<!-- FIG: ch08/fig-01-ttt-layer -->`
- **형식 제안**: 블록+고리 다이어그램.

## F-09 ch09/fig-02-three-regimes (P0 — B7 "draw once, reuse everywhere" 요구)

- **내용 스케치**: 가로축 = chunk 크기 $C$ (1 → L), 3구간 밴드: (1) $C=1$ sequential(품질 기준선, MFU 최악), (2) 중간 $C$ chunkwise(stale-snapshot 근사 — 함수가 $C$에 의존, AI($C$)≈$C$ 상승), (3) $C=L$ full-batch 극한. 각 구간에 품질 곡선(개념적)과 MFU 곡선을 겹쳐 "품질 최적 $C$(작음) vs MFU 최적 $C$(큼)의 긴장" 교차점 표시. exact family(GLA/DeltaNet: $C$는 스케줄일 뿐)와의 대비를 하단 스트립으로. [TNT Fig. 2]의 mismatch 화살표(train $C$ ≠ serve $C$)를 구간 (2) 위에 오버레이.
- **삽입 위치**: ch09 표 9-1 부근(§9의 regime 정리 절). `<!-- FIG: ch09/fig-02-three-regimes -->`
- **재사용**: ch10(비용 모델 절), ch15(§15.2 mismatch)에서 "(그림 9-2, → 9장)" 텍스트 참조만. 이미지 재삽입 금지.
- **형식 제안**: 축+밴드+개념 곡선. 이 책에서 가장 재사용이 많을 그림이므로 라벨 밀도를 낮게(구간명·축·교차점만).

## F-10 ch09/fig-01-chunkwise-dataflow (P1)

- **내용 스케치**: (M4)의 시각화: chunk $n$의 token들($nC{+}1\ldots(n{+}1)C$)이 전부 chunk 시작 상태 $W_{\xi(t,C)}$(앵커, 굵은 노드)에서 gradient를 평가받아 합산되는 데이터 흐름 + FlashAttention tiling(bit-exact, 결과 동일)과의 좌우 대비. 라벨: "tiling은 순서만 바꾼다, chunk는 함수를 바꾼다".
- **삽입 위치**: ch09 (M4) 재게 직후(§9.2). `<!-- FIG: ch09/fig-01-chunkwise-dataflow -->`
- **형식 제안**: 데이터 흐름 그래프 2패널.

## F-11 ch10/fig-01-state-budget (P2)

- **내용 스케치**: 문맥 길이 4K→64K→1M(로그축) 대비 상태 크기: KV cache(선형 증가 직선) vs fast-weight state(수평선, 모델별 $d_v\times d_k$·deep memory 몇 종) — 교차점(crossover $L^*$) 표시. ch10 budget 표의 시각화 버전.
- **삽입 위치**: ch10 4K/64K/1M budget 표 직후. `<!-- FIG: ch10/fig-01-state-budget -->`
- **형식 제안**: 로그-로그 선 그래프. ch10 감축과 상충하지 않게 표를 대체하지 않고 보조.

## F-12 ch12/fig-01-mac-dataflow (P1 — STYLE §5.4 예시로 예약된 ID)

- **내용 스케치**: Titans MAC의 데이터 흐름: 입력 segment(크기 $C$) → persistent tokens $P$ 전치 결합 → attention(문맥 조립) → deep memory $W_t$ read/write 경로 → 출력. write filter·게이트 3종($\eta_t,\beta_t,\alpha_t$)이 slow weights $\Theta$에서 내려오는 화살표를 명시(§12.4의 "누가 학습하는가" 표와 대응). MAG/MAL은 인셋 미니어처 2개로 변형 위치만 표시.
- **삽입 위치**: ch12 §12.5(MAC/MAG/MAL) 도입부. `<!-- FIG: ch12/fig-01-mac-dataflow -->`
- **형식 제안**: 블록 다이어그램 + 인셋 2개. ([Titans Fig. 2] 재구성 표기.)

## F-13 ch13/fig-01-design-space (P2)

- **내용 스케치**: Miras 4축(attentional bias / retention / memory 구조 / inner optimizer)의 방사형 또는 2×2 격자 지도 위에 기존 모델들(linear attn, DeltaNet, GDN, Titans-LMM)과 신모델(Moneta/Yaad/Memora)을 좌표점으로 배치. "기존 모델 = 이 공간의 한 점"이라는 논문 주장 시각화.
- **삽입 위치**: ch13 4축 도입 절 끝. `<!-- FIG: ch13/fig-01-design-space -->`
- **형식 제안**: 좌표 지도. 여유 시에만.

## F-14 ch14/fig-01-omega-window (P1)

- **내용 스케치**: 상단 = Omega rule의 sliding window(길이 $c$): token $t$에서 objective가 window 내 $c$쌍을 재방문, $\gamma_{t,i}$ gate가 쌍별로 열림/닫힘(in-context pruning) 히트맵. 하단 = 병렬화 시 banded mask(대역폭 $c$의 띠행렬)로 나타나는 구조. (M2)의 per-token(=$c{=}1$ 특수화)과의 대비 화살표.
- **삽입 위치**: ch14 §14.3 Omega rule 정식화 직후. `<!-- FIG: ch14/fig-01-omega-window -->`
- **형식 제안**: 히트맵 + 띠행렬. ([Atlas] 해당 그림 재구성 아님 — 자체 작도.)

## F-15 ch15/fig-01-tnt-hierarchy (P1)

- **내용 스케치**: 시퀀스 축 위 2단 구조: 상단 global memory $W^{\mathrm{g}}$(큰 $C_{\mathrm{g}}$, 끊기지 않는 사슬) + 하단 local memories $W^{\mathrm{l}(i)}$(shard 길이 $L_{\mathrm{s}}^{(i)}$마다 $W_{\mathrm{init}}$으로 reset — 가위 표시). reset이 sequential chain을 끊어 shard들이 병렬 lane으로 갈라지는 것(context parallelism)을 lane 분기로 표현. $W_{\mathrm{init}}$이 outer에서 학습되는 화살표.
- **삽입 위치**: ch15 §15.4(계층 구조) 도입부. `<!-- FIG: ch15/fig-01-tnt-hierarchy -->`
- **형식 제안**: 타임라인 2단 + lane 분기. ([TNT Fig. 1] 재구성 표기 가능.)

## F-16 ch16/fig-01-cms-spectrum (P1)

- **내용 스케치**: update frequency 스펙트럼(로그축): $f=\infty$(attention/KV cache) — inner fast weights($f$=per-token·per-chunk) — CMS level들($C^{(\ell)}$ 계단) — $f=0$(동결 MLP·persistent memory). 각 지점에 "이 좌표에 사는 것들"을 스택: 12장(3분: contextual/persistent), 15장(global/local 계층), 16장(연속체 CMS), 17장(sleep이 $f=0$ 쪽을 주기적으로 깨움 — 점선 화살표). 라인 6편의 종합 지도 역할.
- **삽입 위치**: ch16 CMS 절(§16.5 부근) 도입부. `<!-- FIG: ch16/fig-01-cms-spectrum -->`
- **형식 제안**: 수평 스펙트럼 + 계단형 주기 표시. ch17이 텍스트 참조("그림 16-1, → 16장")로 재사용.

## F-17 ch17/fig-01-wake-sleep-lifecycle (P1)

- **내용 스케치**: 원형(또는 수평 반복) lifecycle: wake phase(serving: inner-loop write/read, 경험 축적) → chunk 경계 → sleep phase(offline job: KS/SKS distillation, LTI, Dreaming, synaptic-pruning reset, parameter expansion) → 갱신된 $\theta^{(\ell)}$로 복귀. 하단에 독자 어휘 번역 스트립: "서빙 fleet의 background compaction job" — GPU 점유·주기·중단 가능성 라벨. 세 번째 regime(배포 중 offline 훈련)의 좌표를 그림 왼쪽 미니 축(F-16 축약판)에 점으로 표시.
- **삽입 위치**: ch17 §17.2(lifecycle 정식화) 도입부. `<!-- FIG: ch17/fig-01-wake-sleep-lifecycle -->`
- **형식 제안**: 사이클 다이어그램 + 번역 스트립. ([Sleep Fig. 1] 재구성 표기 가능.)

---

## figure pass 실행 노트 (후속 라운드용)

1. 우선순위 P0 5장(F-02, F-03, F-05, F-06, F-09)부터 — 커리큘럼 명세 미충족(coverage major-3)의 직접 해소분.
2. F-09는 재사용 장(ch10/ch15)의 텍스트 참조가 이미 W2에서 삽입되므로 가장 먼저 확정할 것(라벨 변경이 세 장에 파급).
3. 모든 그림은 대체 텍스트 성격의 캡션(STYLE §5.4: 이미지 다음 줄 캡션 반복)을 fixer가 아니라 figure pass가 최종 작성한다 — W2의 FIG 주석에는 캡션을 넣지 않는다.
4. 작도 도구는 자유(SVG 직접 작성, mermaid→SVG, matplotlib→SVG 정리). 회색조 기본 + 강조 1색, 한 그림 안 폰트 2종 이하.
