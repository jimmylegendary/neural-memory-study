# Part III 연속성·정직성 감사 보고서 (ch18–25 + ch17 bridge)

**감사자**: Part III 연속성·정직성 감사자
**날짜**: 2026-07-12
**대상**: `study-kr/part3/ch18–ch25`, bridge-in 대상 `study-kr/part2/ch17-sleep.md`
**기준**: `style/STYLE-NOTATION.md` v1.1, `dossier/PRE-RESEARCH.md` §3–§5,
`experiments/{RESULTS.md, results.json}`
**판정**: **PASS (조건부)** — 척추·정직성·수치 일관성은 견고하다. medium 2건 + minor·open 수건의
수정이 필요하나 어느 것도 서사·정직성 계약을 무너뜨리지 않는다.

---

## 0. 요약

Part III(ch18–25)는 D4 pair thesis를 척추로 8장 전체가 일관되게 서술하며, memory-centric 논증을
정직 판정의 두 지점(decode RMW 트래픽 / update-frequency↔tier)에 **엄격히 가둔다**. FORCED 두 방향
(훈련용 새 소자, 일반 PIM)은 **다섯 곳에서 능동적으로 배제**된다(ch19 §19.3, ch20 §20.4/§20.5,
ch21 §21.3, ch24 §24.6, ch25 §25.2). 정직성 계약(비율 only / 절대치=roofline 하한 / novel-twin
directional)은 전 장이 태그와 함께 지킨다. 헤드라인 수치(6.44 GB/token, 0.59 FLOP/B, 394×,
S\*=65k/131k, 1.92 ms, 46.5 mJ, 6.8×/14.9×, 7.4×, d\*≈2896, 3.9× cliff, 3.05 mJ/130 µs,
0.45→6.44→343, 16k→1.05M, B_max 20.8/5.2/0.97)는 **8장 전체 + results.json과 완전 일치**한다.

수정 대상은 두 가지 실질 항목(L/L_layer 표기, claim-5 장 매핑)과 minor/open 항목이다.

---

## (a) ch17 → ch18 bridge — 맞물림: **YES**

ch17-sleep.md는 "초대장" 문단으로 닫는다:

> "[Titans]가 'test time에 외우는 법을 배우자'로 열었던 질문은 … '모델은 언제 깨어 있고 언제 자야
> 하는가'로 바뀌었다. 그 질문의 답이 논문 한 편이 아니라 serving 시스템의 설계 문서처럼 생겼다는 것 —
> 그것이 이 라인이 inference 엔지니어에게 남긴 초대장이고, 이 책의 Part III가 그 초대에 응한다."

ch18 §18.1은 이 문단을 인용부호로 되받아 "이 장은 그 초대에 응하는 첫 걸음"으로 연다. 완성형 5성분도
ch17 §17.8 [평가]와 ch18 §18.2가 동일하게 나열하고, ch18은 "17장이 정식화한 완성형"·"17장의 [평가]"로
소유권을 정확히 넘겨받는다. bridge는 **맞물린다**.

- **[P3-LOW-1] 인용 미세 부정확.** ch18 §18.1의 인용문은 ch17 원문 "생겼다**는 것** —"을 "생겼다 —"로
  줄여 인용부호 안에 넣는다. 축약 자체는 무해하나 인용부호는 verbatim을 함의하므로 원문과 일치시키는
  것이 낫다. (위치: ch18 §18.1)

---

## (b) pair thesis 척추 + FORCED 규율 — **PASS**

**척추 일관성.** 8장 전부가 D4 pair thesis를 명시적 축으로 쓴다: ch18(정의) → ch19(scaling law
안에서 드러남) → ch20(roofline 두 절반) → ch21(hardware-lottery split과 정확히 겹침) → ch22(player
지도 위 분업) → ch23(서빙 청구서 두 장) → ch24(제안을 pair로) → ch25(재확인). bridge-in/다음-장 체인도
ch18→19→20→21→22→23→24→25로 끊김 없이 이어진다.

**memory-centric 두 지점 준수.** 모든 장이 memory-centric 논증을 (①) decode state RMW 트래픽,
(②) update-frequency↔tier 두 지점에만 편다. ch19 §19.3 [해설]은 식 (19-1)이 "새 메모리 소자 요구가
아니다"라고 자기 방어까지 한다.

**FORCED 배제 확인(넘은 곳 없음).**

| 장 | FORCED 배제 위치 | 내용 |
|---|---|---|
| ch19 | §19.3 [해설] | state-bytes는 "새 소자 요구"가 아님을 명시 |
| ch20 | §20.4 [평가], §20.5 | 일반 PIM 반증(대부분 FLOP은 GEMM); training에 memory 소자 옹호 안 함 |
| ch21 | §21.3 [평가] | "training에 새 소자 필요" 주장 금지(dossier forced 3) |
| ch24 | §24.6 | 두 FORCED 방향을 명시적 소절로 배제 |
| ch25 | §25.2 | 결론에서도 두 FORCED를 재차 배제 |

PIM은 전 장에서 write-heavy elementwise epilogue(소수 FLOP)에서만, directional로 등장한다. **위반 없음.**

---

## (c) 정직성 계약 — **PASS**

비율·crossover·tier·bound만 load-bearing / 절대 µs·mJ = roofline 하한(A100 runbook 이월) /
novel scratchpad·PIM = directional. 전 장이 준수한다.

- ch18 §18.6이 계약을 정식화, ch20 §20.1·ch23 §23.1·ch24 §24.1·ch25 §25.2가 각각 재선언.
- 절대값(1.92 ms, 46.5 mJ, 6.8×/14.9×, 7.4×, 3.05 mJ/130 µs, 265/68 GB/s)은 등장할 때마다
  "하한/이월" 또는 "directional/NOVEL-SIM-FALSE" 꼬리표를 단다.
- CPU 실측(C\*≈32, cliff GB/s)은 전 장에서 "shape만 이전, 절대 이전 금지(CPU-SHAPE)"로 처리.
- ch23 §23.9는 `results.json`의 `carry_forward_to_A100_runbook` 6항을 그대로 이월 목록으로 옮긴다.

**과장 사례 없음.** 절대치를 주장 근거로 승격한 문장을 전 장에서 발견하지 못했다.

---

## (d) 장 간 수치 일관성 — **강함(1 medium)**

results.json/RESULTS.md와 대조한 헤드라인 수치가 8장에 걸쳐 일치한다(아래 표본).

| 양 | 값 | 등장 장(일치) | source |
|---|---|---|---|
| RMW/token(anchor) | 6.44 GB | 18,19,20,23,24,25 | 6.4425 |
| arith. intensity | 0.59 FLOP/B | 18,20,21,22,24,25 | 0.59 |
| state/GEMV 비 | 394× | 18–25 전부 | 393.8 |
| state/layer | 134 MB | 19,20,23,24 | 134.218 |
| S\*_read / S\*_rmw | 65k / 131k | 18,19,22,23,25 | 65471/130942 |
| decode 하한 | 1.92 ms, 46.5 mJ | 18,19,20,23,25 | 1.9231/46.5 |
| RMW scaling | 0.45→6.44→343 | 18,19,20,23,24,25 | 0.453…343.6 |
| S\* scaling | 16k→1.05M | 18,19,23,25 | 16384…1048576 |
| B_max@10ms(340M/1.3B/7B) | 20.8/5.2/0.97 | 19,23 | 동일 |
| B_max@50ms | 104/26/4.87 | 23 | 104/26.0/4.87 |
| scratchpad | 6.8× E / 14.9× t | 18,20,23,24,25 | 동일 |
| per-layer scratchpad | 7.4× E | 20,23,24 | 7.42 |
| residency crossover | d\*≈2896 | 19,20,23,24 | 2896 |
| RMW cliff | 3.9×, 265→68 GB/s | 20,24 | 3.9 / 264.8·67.9 |
| dirty writeback | 3.05 mJ / 130 µs | 22,23,24 | 3053.5 µJ/130.2 µs |
| stale accretion | 256× | 22,23,24,25 | 256 |
| PIM epilogue | 1.6× E / 2.1× t @3MAC | 18,20,24 | 1.64/2.09 |
| retrieval gap | 53.55/43.70, 67.3/41.9 | 17,18,19,21,22,25 | (Atlas/NL) |
| C\* | 32 host / 306–430 H100 | 19,20,21,22,23,24 | 32 / 337.1 |

**[P3-MED-1] $L$ vs $L_{\mathrm{layer}}$ 표기 불일치 (STYLE v1.1 위반 + 장 간 불일치).**
STYLE §1.2/§1.5는 무첨자 $L$을 **sequence 길이 전용**으로 예약하고 layer 수는 $L_{\mathrm{layer}}$를
쓰도록 v1.1에서 명문화했다(“layer 수·MLP 깊이 의미로 사용 금지”).
- **ch19만 준수**: 식 (19-1) `B_state = m d² L_layer s`, 본문 "$L_{\mathrm{layer}}$는 layer 수
  (sequence 길이 $L$이 **아님**)", anchor를 `L_layer=24`로 표기.
- **ch20·21·23·24·25는 위반**: 모두 bare $L$을 layer 수로 사용.
  - ch20 §20.1 "layer 수 $L$", §20.2 "$L$개 layer", 표 20-1 "RMW=$2\,m\,d^2\,L$"
  - ch21 §21.5 "$m\,d^2\,L$의 RMW"
  - ch23 §23.2/§23.4 "$m\,d^2\,L$"
  - ch24 §24.3.1 "$m\,d^2\,L$"
  - ch25 §25.2 "$m\,d^2\,L$"
  - 추가로 anchor 문자열이 ch18/20/23/25에서 `L=24`(bare), ch19에서만 `L_layer=24`로 갈린다.
  → **수정 방향**: ch20–25의 prose·수식·표의 layer-count $L$을 전부 $L_{\mathrm{layer}}$로 교체
  (ch19에 정렬). code-font anchor 문자열의 `L=24`는 데이터 라벨로 관용 허용 가능하나, 일관성을 위해
  함께 맞추길 권장. (results.json/RESULTS.md는 데이터 파일이라 STYLE 비적용 — 원본 유지 무방.)

**[P3-LOW-2] TNT 배속 정밀도 불일치.** ch22 §22.2는 "17.37×", ch21/23은 "17.4×", dossier는
"up to 17.4×". 반올림 차이라 모순은 아니나 책 표준(17.4×)에 맞추는 편이 낫다. perplexity·chunk-mismatch
수치(23.09 vs 25.07; 13.78→36.45; 0.96h vs 1.12h; 1.3× vs FlashAttention)는 장 간 일치.

---

## (e) dossier §4 ToC 커버리지 — **충족(1 medium)**

| dossier §4 항목 | 대응 장 | 충족 |
|---|---|---|
| ch18 arc/convergence/완성형+2유보 | ch18 | ✓ |
| ch19 scaling + state-bytes/capacity 축 + A3 | ch19 | ✓ (A3 falsifier 3조건 명시) |
| ch20 decode roofline·backward primitive·NS-5·chunk·MFU | ch20 | ✓ 전부 |
| ch21 hardware-lottery | ch21 | ✓ |
| ch22 player-strategy(3 player) | ch22 | ✓ |
| ch23 TNT→7-70B·prefill/decode split·per-session cache class·sleep job·비용모델 | ch23 | ✓ 전부 |
| ch24 proposals(algo + HW pair + serving SW) | ch24 | ✓ (제안 A–I + 배제) |
| ch25 conclusion·agenda | ch25 | ✓ (5 어젠다) |

**[P3-MED-2] claim-5 장 매핑 오류(표 18-1 ↔ 실제 본문 불일치).**
ch18 표 18-1은 claim 5(update cadence → memory-tier)의 "주 담당 장"을 **"19, 24"**로 적고, figure
(d)도 claim 5에 연결한다. 그러나:
- **ch19에는 cadence→tier / frequency-tier 내용이 없다**(ch19는 state-bytes·capacity·RMW/crossover
  scaling만 다룸). figure exp-d도 ch19에 등장하지 않는다.
- claim 5의 실제 본진은 **ch23 §23.6**(제목 "Frequency-tiered placement: 서빙 메모리 계층 그 자체",
  표 23-2, 그림 23-1=exp-d)이며 **ch24 §24.3.2**(제안 E, 그림 24-3=exp-d), 부차적으로 ch20 §20.6.
- §18.5 로드맵도 자기모순: 19장 설명에 frequency-tier가 없고, 23장 설명은 claim 3만 credit하고 claim 5
  (§23.6의 주제)를 누락한다.
→ **수정 방향**: 표 18-1 claim 5 행의 "주 담당 장"을 **"23, 24"**(또는 "20, 23, 24")로 교정하고,
§18.5의 19장·23장 로드맵 설명을 실제 배치(frequency-tier=23·24)에 맞춰 조정.

---

## 기타 open 항목

- **[P3-OPEN-1] ch22 미확정 인용(TODO-VERIFY).** ch22 line 49의 주석: Gated DeltaNet의 arXiv ID와
  저자 소속(NVIDIA) 미확정. §2.5는 첫 인용 시 arXiv ID 병기를 요구하는데 현재 "Yang et al. 2024"만
  있음. 또한 ch21 §21.4는 "NVlabs/GatedDeltaNet 공식 구현"을 단정하는데, ch22가 NVIDIA affiliation을
  미확정으로 둔 것과 표면상 긴장 — 원저자와 구현 호스팅은 별개일 수 있으므로 모순은 아니나 확정 필요.
- **[P3-OPEN-2] figure 임베드 방식 불일치(cosmetic).** ch19/20/23/24는 `![](../../figures/…png)`로
  이미지를 임베드하나, ch21 §21.3(exp-c)·ch22 §22.1–22.2(exp-c, exp-a)는 `<!-- FIG -->` 주석 +
  plain-text 캡션만 두고 이미지를 임베드하지 않는다. figure 소유가 다른 장이라 의도적일 수 있으나,
  렌더 시 두 장에서 그림이 보이지 않으므로 통일 여부 결정 필요.
- **[P3-OPEN-3] state multiplier $m$ 전역 예약 부재(minor).** Part III는 $m$을 state multiplier로
  일관되게 쓰고 ch19 §19.3·ch20 §20.1이 정의하나, STYLE §1.2는 $m_t$(outer moment)·Atlas의
  $m$(저장 쌍 수, §1.7.3)로 이미 사용 중이다. Part III-국소 기호로서 문제는 없으나, ch18이 anchor
  문자열에서 정의 없이 "$m\,d^2$"를 먼저 쓰므로 ch18에 1줄 정의를 추가하면 깔끔.

---

## 판정 근거 요약

- **연속성**: ch17→ch18 bridge 맞물림, 8장 bridge/다음-장 체인 완결, 완성형 소유권 정확 인계 → PASS.
- **척추/FORCED**: pair thesis 전 장 일관, memory-centric 두 지점 준수, FORCED 5곳 능동 배제, 위반 0 → PASS.
- **정직성 계약**: 비율/하한/directional 태그 전 장 준수, 절대치 승격 사례 0 → PASS.
- **수치 일관성**: 헤드라인 20+ 수치 8장·results.json 일치. 결함 1건(L/L_layer 표기) → PASS w/ fix.
- **ToC 커버리지**: dossier §4 전 항목 충족. 결함 1건(claim-5 장 매핑) → PASS w/ fix.

수정 우선순위: **P3-MED-1(L_layer 표기)** 과 **P3-MED-2(claim-5 매핑)** 를 먼저 반영하고,
open 항목(GDN 인용 확정, figure 임베드 통일)은 후공정(P4/서지)에서 처리 권장.
