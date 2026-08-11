# OUTLOOK-REVIEW — 외부 전망 문서 적대적 검토

**검토 대상**: `Sleep-time Compute 전망 — Offline Consolidation, Dreaming, CMS 및 HW 확장` (2026-08-11, 10쪽, 한국어). 이하 [문서].
**판정 기준**: `dossier/stc-v2/PRE-RESEARCH.md`, `study-kr-stc/` 30장, `experiments/stc/X2·X3`, `notes/stc-v2/` 32편(2026-08-11 추가 3편 포함).
**검토 시점**: 2026-08-11. **검토자**: 이 책의 편집자.
**문체**: `style/STC-STYLE-NOTATION.md` §6.

---

## 0. 요약 판정

[문서]는 하나의 뼈대 위에 서 있다 — **CMS(Nested Learning)가 작동하는 multi-timescale 기제이고, Sleep은 그 level 사이의 transition operator다.** 이 뼈대가 §4에서 세워지고 §5(Scheduler)·§6(continuum)·§7(2030 전망)·§9(HBF 계층)이 전부 그 위에 올라선다.

**그 뼈대는 서 있지 못한다.** 세 개의 독립 판정이 같은 방향을 가리키고, 갈리지 않는다.

1. 이 책 ch18: "[NL]이 구현한 모든 갱신은 (U-W) 모양이고, $\Theta$는 그 축 위에서 test-time에 아무것도 움직이지 않는 $f=0$ 끝점으로만 등장한다. … **연속체는 표기의 다리이지 기제의 다리가 아니다**"(ch18 §18.1, §18.4).
2. 새 서베이(2607.25380): NL을 Table II "Optimization-based writing" 행에 TTT-E2E·Titans·In-Place TTT와 함께 놓고, 그 행의 기제 정의를 "Memory parameters or **fast weights** are updated"라 쓰며, "updates are temporally and functionally separable from the **static base model weights**"[Memory for LLMs §IV-A-b, p.10], "backbone parameters remain **frozen or adapt slowly**"[동 §V-A-d, p.15]라 쓴다.
3. NL 논문 자신: "in this work, we focus on the first stage: memory consolidation as an online process"[NL §1] — 오프라인 단계를 이름 붙여 부르고 명시적으로 범위 밖에 둔다.

세 판정이 갈리는 것이 결과가 아니라, **세 판정이 갈리지 않고 [문서]만 갈린다는 것이 결과다.** [문서]는 서베이의 "frozen or adapt slowly"에서 "frozen or"를 떨어뜨리고 "adapt slowly"만 가져왔는데, 서베이에서 그 갈래는 기제도 인용도 예시도 없는 쪽이다(→ §2 반증 R1).

그럼에도 [문서]는 이 책이 갖지 못한 좌표 하나를 정확히 짚는다 — **Consolidation(보존)과 Dreaming(생성)은 실패 모드가 다른 두 연산이다.** 이 책은 그 구분을 ch06·ch22·X4에서 이미 **쓰고 있으면서 축으로 세우지 않았다.** [문서]의 목적별 축은 이 책의 층별 축을 대체하지 못하지만, 층 축의 **두 번째 좌표**로는 성립하고, 격자로 놓는 순간 빈 칸 넷을 즉시 드러낸다(→ §5). 이것이 [문서]의 최대 기여이며 이 책이 채택해야 할 것이다.

정직성 항목 하나를 먼저 인정한다. [문서]는 §7 전체를 "제 전망", §4의 CMS 해석을 "나는 이렇게 해석하는 게 맞다고 봅니다", §2 도입을 "제 판단으로는"으로 표시하고, 말미에 "전망/해석 부분은 위 논문들의 결과를 바탕으로 한 분석적 extrapolation"이라 못 박는다. **표시된 것은 정직하다.** 이 검토가 결함으로 세는 것은 표시가 **첫 등장에만 붙고 사용처에는 붙지 않는** 경우뿐이다.

**계수 판정**: 확증 11항 · 반증 6항 · 근거 없음 8항 · 코퍼스 밖 7항.

---

## 1. 확증 — 이 책이 독립적으로 같은 결론에 도달한 것

| # | [문서]의 주장 | 이 책의 근거 |
|---|---|---|
| A1 | 2025년 원 Sleep-time Compute는 offline consolidation/dreaming을 제안한 논문이 아니라, 질의 전 persistent context를 미리 추론해 future query의 test-time compute를 줄이는 개념이다 | PRE-RESEARCH §2.2 — Letta STC는 $E$-경로, 시계는 "질의 전 1회", $\mathrm{wr}$은 파괴적 덮어쓰기. ch14가 "상속 없는 시작"으로 서술한다. **[문서]가 이 라인의 출발점을 정확히 좁힌다** |
| A2 | 항상 sleep compute를 돌리면 "그냥 training compute가 하나 더 생긴 것"이다 | ch28 §28.3.1 반복 행 — "$B_s$는 라운드마다 다시 낸다. 그러므로 '이 방법의 $B_s$'라는 말은 라운드 수를 지정하지 않으면 정의되지 않는다." ch17 §17.4.1 — 헤드라인 설정은 $N_q=1$이라 식 (A)가 붕괴한다. **이 책이 더 강한 형태를 이미 갖고 있다** |
| A3 | "언제 자야 하는가"를 정하는 metacontroller가 없다 | Letta STC §7이 스스로 남긴 future work("optimally allocating inference compute between sleep-time and test-time")와 서베이 §VI-D(pp.17–18, "Most existing models rely on static compositions… A promising direction lies in adaptive memory orchestration")가 **서로를 인용하지 않고 같은 공백에 도달**한다. 두 경로의 독립 수렴이므로 이 공백은 실재한다 |
| A4 | Consolidation과 Dreaming은 실패 위험이 다르다 — 전자는 forgetting, 후자는 hallucinated learning | ch06(model collapse — $\mathcal{R}$이 자기생성일 때의 제약), X4 F6(replay를 보존된 원본에 앵커하면 자기 출력 앵커보다 **모든 혼합비에서** 오래 버틴다), Continual Facts §6.2 p.14–15($\Theta_0$ 고정 교사 +2pp·유지율 54% 대 자기 누적 merge 교사 −31pp·유지율 21%). **부호가 세 곳에서 같다**(→ ch29 판정 4) |
| A5 | Auto-Dreamer는 "이름은 Dreamer지만 기능적으로는 상당 부분 consolidation"이다 | 확증이며 [문서]가 자기 주장보다 더 옳다. 논문 자신이 각주로 이름 충돌을 차단한다 — "Auto-Dreamer is distinct from the Dreamer family of world models; our method operates on memory entries and source trajectories"[Auto-Dreamer fn.2, p.2]. 그리고 전문에 `synthetic` 0회·`self-play` 0회다[본서 관찰 — `notes/stc-v2/auto-dreamer.json`]. **생성 성분이 부분적인 것이 아니라 아예 없다** |
| A6 | Auto-Dreamer는 여러 session의 memory를 offline으로 모아 raw trajectory provenance까지 확인한 뒤 memory region을 통째로 rewrite한다 | 확증. $E_{k+1}=(E_k\setminus\mathcal{W}_k)\cup\hat c_k$[Auto-Dreamer Eq. 1, §3, p.4]; `get_source_trace`가 task agent는 회수할 수 없는 원시 action–observation trace를 읽는다[동 §3, p.4; App. E.2, p.23]; "old entries do not persist by default, and information survives only if it is re-synthesized into the replacement set. As a result, abstraction, deduplication, contradiction resolution, and omission-based forgetting become default behaviors."[동 §1, p.2] |
| A7 | Auto-Dreamer는 consolidator 자체를 GRPO로 학습한다 | 확증. "only the consolidator parameters are updated during training, and only the memory bank is updated after deployment"[Auto-Dreamer §4, p.4]. corpus에서 **$E$-경로 쓰기 연산자 자체가 학습 파라미터를 갖는 첫 사례**다 |
| A8 | Behrouz `LM Need Sleep`이 Sleep을 두 단계(Memory Consolidation = Knowledge Seeding → Dreaming)로 명시한다 | 확증. 이 책이 읽은 판이 같은 v2다(`arXiv:2606.03979v2 [cs.LG] 10 Jul 2026`). (S-9) Knowledge Seeding objective와 (S-10) Dreaming이 한 sleep 사이클 안에서 (U-$\Theta$)를 **두 번** 인스턴스화한다[LM Need Sleep Sec. 3.3, 3.4] |
| A9 | Consolidation은 latency-insensitive·batch 가능·throughput 지향이고, wake는 latency-critical이다 | ch29 표 29-6에서 **두 경로에 대해** 확증. $E$-경로 sleep = 오프라인 추론 배치 잡, 선점 가능성 높다. $\Theta$-경로 sleep = 훈련 잡, 라운드 단위 체크포인트. 세 번째 경로는 반대다(→ §2 반증 R5) |
| A10 | 서베이가 offline/online, short/long-term, implicit/explicit를 독립 design axis로 놓고, multi-timescale memory와 hardware-algorithm co-design을 향후 핵심 문제로 본다 | 서베이의 내용으로서 확증. $\tau(P)=(\text{representation},\text{update-dynamics},\text{persistence})$; §VI-E, p.18 — "memory bandwidth and storage hierarchies increasingly dominate system performance… memory operations themselves may become primary computational primitives." 단 두 유보가 붙는다(→ §3 U7) |
| A11 | 원 논문에서 query predictability가 높을수록 sleep compute의 가치가 높다 | 논문의 보고로서 확증[Letta STC §5]. 이 책도 그 진술을 기록한다. **보고의 존재는 확증이고 값의 지위는 다르다**(→ §2 반증 R6) |

---

## 2. 반증 — 이 책에 반대 증거가 있는 것

### R1. CMS를 작동하는 multi-timescale 기제로 놓고 그 위에 L0–L4를 세운 것 — [문서] §4

**[문서]의 주장.** "CMS-L0 working state/KV(milliseconds) → L1 recurrent/fast weight(seconds~minutes) → L2 episodic/semantic memory(minutes~hours) → L3 LoRA/expert(hours~days) → L4 stable parametric knowledge(days~months) → base model/very slow memory."

**판정문.**

> "표 18-3에 (i)을 만족하는 행이 없고 (ii)를 만족하는 행은 통제군 하나뿐이다. 따라서 [NL]의 연속체는 **표기의 다리**다." (ch18 §18.4 [평가])

> "**정의.** 연속체 주장이 **기제의 다리**이려면 두 조건이 필요하다. (i) 축 위의 어떤 갱신이 문맥 경계를 넘어 존속하고, (ii) 그 산출물이 배포 아티팩트에 되돌아 쓰인다. 둘 중 하나라도 없으면 그 연속체는 **표기의 다리**다." (ch18 §18.4)

**무엇이 어긋나는가.** 넷이다.

1. **실측 상한이 2K 토큰이다.** "실측된 가장 느린 CMS 눈금이 2K 토큰이다. 연속체 자체의 실증 상한이 2K 토큰이라는 뜻이고, 세션은 물론이고 sleep 라운드와는 자릿수가 세 자리 이상 떨어져 있다"(ch18 §18.6.2, [NL §9.1] 근거). [문서]의 L2·L3·L4 세 층 — minutes~hours, hours~days, days~months — 은 NL에 대응물이 **없다.**
2. **NL의 nested 변형은 반대 방향으로 움직인다.** CMS level의 running state는 $\lceil C^{(s)}/C^{(s+1)}\rceil$ step 뒤 초기값으로 **재초기화**된다[NL Eq. 71–72, §9.1]. 층 사이로 흘러 올라가는 것이 아니라 제자리로 되돌아간다.
3. **논문 자신이 그 자리를 비웠다.** "in this work, we focus on the first stage: memory consolidation as an online process"[NL §1]. ch18 §18.2가 이 문장을 "이 장의 판정에서 가장 무거운 증거"로 지목한다.
4. **ch18이 이 오독을 미리 차단해 두었다.** "[NL §7.3]의 ad-hoc level stacking은 $\eta^{(\ell)}\to0$으로 두면 … 진짜 다리라면 이렇게 생겼을 것이다. 그러나 보간의 한쪽 끝은 '$\Theta$처럼 **행동하는** $W$'이지 $\Theta$가 아니다. 체크포인트에 되돌아 쓰이는 경로가 식 어디에도 없다"(ch18 §18.4).

**세 번째 판정이 이 책 편이다.** 새 서베이는 NL을 Table II 1행 "Optimization-based writing"에 놓고, 그 행의 기제 정의를 "Memory parameters or **fast weights** are updated by minimizing an explicit objective"라 쓴다. 동반자는 TTT-E2E·Titans·In-Place TTT — 세 편 모두 이 책이 $W$로 분류하는 계열이다. 그리고 서베이는 NL을 Table I에서 **Online**으로 표시한다. 서베이의 online은 "updates during inference"를 뜻하므로, **서베이의 축 위에서도 NL은 판별식 조건 (1)을 만족하지 않는다.**

**[문서]의 인용이 어디서 어긋났는가.** [문서] §4는 서베이를 근거로 "backbone은 훨씬 느린 단위로 바뀌는 구조"라 쓴다. 서베이의 원문은 이렇다.

> "activation states evolve at token-level granularity, explicit memory modules update at interaction or batch level, and backbone parameters remain **frozen or adapt slowly**." [Memory for LLMs §V-A-d, p.15]

[문서]는 "frozen or"를 떨어뜨렸다. 그리고 서베이는 "adapt slowly" 갈래에 **기제도 인용도 예시도 붙이지 않는다**[본서 관찰 — `notes/stc-v2/memory-survey-2607.json` nested_learning_classification]. 즉 [문서]가 뼈대로 삼은 층은 외부 서베이에서도 근거 없는 절반이다.

### R2. "Sleep은 각 level 사이의 transition operator다 — $M_{i+1}\leftarrow C_i(M_i, M_{i+1})$" — [문서] §4

**판정.** 이 연산자는 **실재하고, 귀속이 틀렸다.**

corpus에 정확히 한 사례가 있다 — `LM Need Sleep`의 wake $W$ → sleep $\Theta$ 순서다[PRE-RESEARCH §2.2: "$\Theta$ (←$W$) | sleep 라운드 | wake $W$ → sleep $\Theta$ 순서"]. [문서]는 이 연산자를 CMS에 귀속시키는데, CMS에는 없고 **CMS 위에 graft된 논문에** 있다. ch18 §18.8.1이 그 이송 경로를 이미 적었다 — "[NL]이 남긴 것 중 이 책에 가장 중요한 항목은 [NL §1]의 범위 진술이다. … 이것이 뒤 장에서 상속으로 실현된다(→ ch21)."

그리고 이 사실이 [문서] 결론("HOPE/Nested Learning은 이걸 구현하기에 굉장히 좋은 architecture abstraction입니다")을 직접 반증한다.

> **THE FULL PARADIGM IS NEVER RUN ON THE ARCHITECTURE IT IS DEFINED FOR.** Sec. 3 defines everything over a CMS/MoE backbone; Sec. 4.1's consolidation results use Hope; but the Dreaming results use "Sleep (Transformer)" and "Sleep (Transformer + four-level)" [LM Need Sleep Table 3, p.12] and Llama-3.2-1B [동 App. B.3, p.25].

CMS 위에서 Sleep을 정의한 유일한 논문이 그 Sleep의 절반을 CMS가 아닌 곳에서 돌렸다. 그리고 PRE-RESEARCH §4의 의무 caveat가 같은 방향이다 — "기제는 pre-trained Llama/Qwen 위의 graft"(ch21 배정).

### R3. `Do LMs Need Sleep?`을 "state 자체를 바꾼다"의 대표로 세우고 그 결과를 무조건 인용한 것 — [문서] §1

**[문서]의 주장.** "sleep compute를 늘리면 특히 reasoning depth가 깊은 문제의 성능이 좋아졌습니다. … 이건 이미 sleep-time compute를 memory consolidation mechanism으로 바꿔버린 것입니다."

**판정문 셋.**

> "**[평가]** ch11이 이 논문을 통과로 분류한 것은 옳고, 그 통과는 **Depo 설정이 지고 있다.** 헤드라인 실험은 판별식의 조건 (1)과 (4)를 동시에 약화한 구성이며, 그 대가는 정확히 상각의 소멸이다 — 질의 조건부 consolidation에는 나눌 $N_q$가 없다. … 이 논문은 $W$-경로가 sleep-time compute이 될 수 있음을 보인 유일한 정면 사례이면서, **자기 최대 성과를 그 성질을 포기한 설정에서 얻었다.**" (ch17 §17.4.1)

> "계산량을 맞춘 baseline이 없다 — 캐시를 유지한 full attention은 어느 과제·어느 스케일에서도 실행되지 않았다." (ch27 B3, ch17 §17.6)

> "축 위에 점이 넷 있어도 그 축이 예산 축이 아니면 기울기는 예산에 대한 기울기가 아니다." (ch28 §28.2.3)

**무엇이 어긋나는가.**

- 헤드라인 GSM-Infinite 설정은 질문을 문맥 **앞**에 놓는다 — "This order gives the model the query before it reads the long problem context"[Do LMs Need Sleep? §6.3]. 상태를 바꾸는 연산이 질의 도착 **뒤**에 실행되므로 조건 (1)이 문자 그대로 성립하지 않고, 요청당 질의가 하나이므로 조건 (4)도 약해진다.
- 개선의 크기가 설정의 산물이다. "sliding-window 2연산의 52%는 0.596에서 0.905로 가는 값인데, **같은 모델**이 hard eviction $C=2000$에서 같은 2연산을 loop 없이 0.857–0.868로 푼다. … **새 능력이 아니라 설정이 만든 구멍을 메운 값이다**"(ch17 §17.6.4).
- **"memory consolidation mechanism으로 바꿔버렸다"는 서술이 이 논문의 상태 수명과 어긋난다.** ch27 표 27-3의 $W$-경로 "풀지 않는 문제" 칸: "**요청을 넘는 지속성.** 상태가 시퀀스마다 0으로 초기화되어 세션과 함께 죽는다." [문서]의 lifecycle 다이어그램은 이 논문의 산출물을 "persistent model state → next WAKE cycle"로 흘려보내는데, 그 화살표가 이 논문에 없다.

### R4. SleepGate를 micro-sleep의 실증 사례로 놓은 것 — [문서] §6

**[문서]의 주장.** "SleepGate 같은 연구는 이미 inference 중에 adaptive trigger로 sleep micro-cycle을 실행해서 stale KV를 선택적으로 evict/compress하고 surviving information을 consolidate하는 방향을 보여줍니다."

**판정.** 네 성분 중 셋이 이 논문에 없다.

1. **evict/compress는 실행된 적이 없다.** "Theorem 1, Corollary 1 and Proposition 1 all presuppose eviction; the evaluated soft-biasing variant evicts nothing"[`notes/stc-v2/sleepgate.json` UF-7]. 평가된 변종은 캐시를 고치지 않고 bias 벡터를 만든다[SleepGate Eq. 10–11; Algorithm 1 lines 16–21].
2. **adaptive trigger가 무엇을 gate하는지 논문 안에서 결정되지 않는다.** "every forward pass computes both unbiased (wake) and soft-biased (sleep) logits"[SleepGate §4.2 Stage 2, p.9]. Table 2 캡션은 보고된 수치가 biased pass에서 읽힌 값이라 밝힌다 — "SleepGate uses post-sleep (soft-biased) evaluation."[동 Table 2 caption, p.12]. Algorithm 2 line 12의 `trigger(t)` 게이팅과 정면 충돌한다[동 UF-6].
3. **지속하는 유일한 성분의 기여가 0에 가깝다.** Decay Only ablation은 전 depth 최대 12.5%이고 논문 자신이 "performs comparably to other baselines"라 쓴다[SleepGate §7.2, p.13].
4. **$N_q=1$이 구성이다.** 평가되는 모든 에피소드가 질의 하나로 끝난다[SleepGate §6.2, p.11]. 이후 질의가 없으므로 조건 (4)가 성립할 자리가 없다.

**판별식 판정: FAIL, 조건 (3).** $\Theta$·$W$·$E$ 어느 층도 바뀌지 않는다. 그리고 비용이 전부 $L_w$에 실린다 — 이 책의 시금석("sleep은 정의상 $L_w$에 0을 기여한다", ch11 §11.2.3, ch01 표 1-4)의 정반대다. MemGPT는 decode 예산을 사용자 턴 안에서 쓰고, SleepGate는 attention 예산을 **같은 forward pass 안에서** 쓴다.

부수 사실 하나를 적는다. 논문은 gate의 파라미터 오버헤드를 "< 0.01%"라 쓰는데 자기 Table 1에서 9.36%다[동 UF-15]. [문서]가 이 논문을 "micro-sleep의 비용이 작다"는 증거로 쓴 것은 아니지만, 그 방향의 인용도 성립하지 않는다.

### R5. "accelerator를 latency critical wake partition + throughput optimized sleep partition으로 나눌 수 있다" — [문서] §8

**판정문.**

> "$W$-경로가 가장 곤란하다 — sleep이 요청 안에서 일어나므로 유휴 시간으로 밀 수 없고, ch17이 확인한 대로 한 요청이 자고 나머지가 decode하는 배치에는 커널 모양이 둘 살아 있다." (ch29 §29.3.7)

표 29-6 $W$ 행의 선점 가능성 칸: **"낮다. 요청 경계에 묶인다."** [문서]의 파티션은 $E$·$\Theta$ 두 경로에서 성립하고 $W$에서 성립하지 않는다. 그런데 [문서] §1이 뼈대로 세운 논문이 정확히 $W$-경로 논문이다. **같은 문서 안에서 §1과 §8이 충돌한다.**

### R6. "wake-time에는 거의 read-only, sleep-time에만 controlled write가 일어난다" — [문서] §9

**판정문.**

> "**판정 2 — 이 자리는 Neural Memory가 $W$-경로에서 지목한 HBM read-modify-write 기회와 다른 자리다. 두 결론을 잇지 않는다. 조건 없다.** … 특히 **두 요구는 공존할 때 상쇄되지 않고 합쳐진다.** fast weight를 쓰는 모델에 사용자별 delta를 얹으면 HBM은 여전히 토큰당 whole-state RMW로 포화하고 그 위에 요청당 웜 스테이징이 추가로 얹힌다." (ch29 판정 2)

[문서]가 세운 세 경로 중 $W$-경로는 **토큰마다 전 상태를 읽고 쓴다.** "wake=read-only"는 그 경로에서 거짓이고, [문서]는 그 경로를 §1의 중심에 놓았다.

그리고 [문서]가 기회의 자리를 잘못 짚는다. ch29 판정 1:

> "**기회는 실재한다. 그리고 그 자리는 HBM 대역폭이 아니라 요청 지연 안에 delta를 스테이징하는 웜 계층과 그 계층의 쓰기 내구성이다.**"

세 나눗셈이 [문서]의 tiering 논지와 다른 축을 만든다. (i) delta는 같은 요청의 base 가중치 읽기 대비 약 **0.105%**이므로 대역폭 예산에서 보이지 않는다[ch29 §29.3.1, 본서 산술]. (ii) 매체의 하루당 전체 재기록 횟수는 $\mathrm{DWPD}=1\ \text{day}/T_k$로 **용량·사용자 수·dtype과 무관하게 라운드 주기 하나**가 정한다(식 (29-3)). (iii) 총량 16.777 TB는 데이터센터 규모에서 제약이 아니다. [문서] §9의 "frequently changing → SRAM/HBM, medium-term → HBM/DRAM, stable → HBF"는 **총량과 대역폭 축으로 계층을 나누는데, 이 책의 회계에서 그 두 축은 제약이 아니다.**

### R7. Letta STC의 5×·2.5×를 무조건 인용한 것 — [문서] 서두

[문서]는 "실제로 동일 정확도에서 test-time compute를 약 5× 줄였고, 같은 context에 여러 query가 들어오는 경우 query당 비용을 2.5× 줄였으며"라 쓴다. PRE-RESEARCH §4의 의무 caveat 넷이 걸린다.

- 이 값들은 저자 제작 벤치(Stateful GSM-Symbolic/AIME, Multi-Query)에서 나왔고, Stateful AIME은 문맥이 비어 있는 인스턴스를 포함하며 그 개수를 밝히지 않는다.
- 비용모델은 **생성 토큰만 센다.** prefill 항이 없고 $\kappa_{\text{cost}}=10$은 벤더 문서 인용이며 민감도 분석이 없다.
- **이득은 높은 test-time 예산에서 역전한다(두 과제 모두). 기제 미제시.**
- Multi-Query 데이터셋의 90.9%가 모델 생성 문항이고 생성 답의 검증이 보고되지 않았다.

이 책은 이 값을 부정하지 않는다 — ch25 판정 3이 "짧은 문맥·큰 절감률 레짐에서 5×는 뒤집히지 않는다"고 확정했다. 반증되는 것은 **"실제로 줄였다"는 무조건 서술**이며, 특히 세 번째 항목(역전)은 [문서]의 §5 스케줄러 논지가 정면으로 필요로 하는 정보다 — 자야 할 때를 정하려면 이득이 뒤집히는 지점을 알아야 한다.

---

## 3. 근거 없음 — 제안인데 기정사실처럼 딛고 올라간 것

**표시된 것과 표시 없는 것을 가른다.** [문서]는 §7 전체, §4의 CMS 해석, §2 도입, lifecycle 다이어그램, Scheduler의 learned-policy 예측을 전부 "제 전망/내 생각"으로 표시했다. **그것들은 결함이 아니다.** 아래는 표시 없이 전제로 쓰인 것과, 표시가 첫 등장에만 붙고 사용처에는 붙지 않은 것만 센다.

### U1. Sleep Scheduler 목적함수 — 다섯 항이 전부 오늘 채울 수 없다 (표시 없음)

[문서] §5는 "따라서 future system에서는 대략 이런 목적함수를 가지는 metacontroller가 필요합니다"로 연결해, 도출된 것처럼 놓는다.

$$\max_\pi\ \mathbb{E}[\text{future utility}] \;-\; \lambda\,C_{\text{sleep}} \;-\; \mu\,C_{\text{wake latency}} \;-\; \nu\,M_{\text{footprint}} \;-\; \xi\,I_{\text{interference}}$$

항마다 이 책의 계수를 댄다.

| 항 | 이 책의 대응 | 인쇄 실태 |
|---|---|---|
| $C_{\text{sleep}}$ | $B_s$ | **절대 FLOPs 또는 토큰으로 인쇄한 논문 1편**(ReasoningBank, 과제당 3748.4 토큰)이고, 그 1편조차 backbone 크기가 없어 FLOPs로 환산되지 않는다[X3 Q1, Q2]. 19편 침묵, 그중 9편은 $B_s$가 **정의조차 되지 않는다** |
| $C_{\text{wake latency}}$ | $L_w$ | **요구 단위로 인쇄한 논문 0편**[ch25 §25.2.1]. 표 29-4의 세 스테이징 방식 어디에도 측정값이 없다 |
| $M_{\text{footprint}}$ | $C_{\text{cap}}$ | **바이트로 인쇄한 논문 0편**[ch26 판정 6, PRE-RESEARCH §4] |
| $I_{\text{interference}}$ | **대응 칸 없음** | 이 책의 비용 4종에 자리가 없다. 서베이가 "robustness to interference"를 평가 축으로 제안하되 **단위를 주지 않는다**[Memory for LLMs §VI-F, p.18] |
| $\mathbb{E}[\text{future utility}]$ | 식 (A)의 분모 $N_q$ | **실서빙 $N_q$ 분포를 보고한 논문 0편**[ch25 §25.3.3] |

**간극의 정확한 형태.** 세 겹이다.

1. **값이 없다.** 다섯 항 중 하나도 오늘 수로 채워지지 않는다. 최적화는커녕 목적함수를 **평가**할 수 없다.
2. **계수가 정의되지 않는다.** $\lambda,\mu,\nu,\xi$는 FLOPs·ms·bytes·(미정)를 하나의 스칼라로 합치는 환산 계수인데, 네 단위 중 셋이 미인쇄이고 하나는 단위 자체가 없다. 그리고 ch28 판정 2대로 **$B_s$의 눈금은 워크로드 의존이므로 $\lambda$는 실험의 불변량이 아니다** — 같은 FLOPs가 $N_q$에 따라 다른 비용이 된다.
3. **부호가 경로마다 다르다.** ch28 판정 3: "$E$-경로에서 $B_s$를 키우면 $\hat c$가 길어지고 매 질의의 프롬프트 비용이 **늘어난다**. $\Theta$-경로에서는 merge 뒤 토큰당 계산이 그대로다. $W$-경로에서는 $N_q=1$이라 상각이 성립하지 않고 총계산이 재귀 횟수에 비례해 는다. … **경로를 지정하지 않은 $B_s$ 곡선은 부호가 정의되지 않는다.**" 그러므로 $C_{\text{sleep}}$과 $C_{\text{wake latency}}$를 **독립된 두 항으로 놓는 것 자체가** 이 회계에서 성립하지 않는다 — $E$-경로에서 둘은 같은 방향으로 함께 움직인다.

**그럼에도 이 간극은 하드웨어를 요구하지 않는다.** ch28 R1–R9 중 여덟, ch29 판정 6의 인쇄값 셋이 **이미 존재하는 실행 로그의 인쇄**로 해소된다. 그리고 새 논문 셋이 그 진단을 강화한다 — Auto-Dreamer는 $B_s$를 **측정해 놓고 인쇄하지 않았다**("`summary.json` reports total wall-clock time, and per-role LLM call and token counts"[Auto-Dreamer App. B.2, p.18]).

### U2. CMS-L0~L4의 시간 눈금 — 표시가 사용처에 붙지 않는다

[문서] §4는 5층 계층과 눈금(ms / s~min / min~hr / hr~day / day~month)을 "나는 CMS를 앞으로 이렇게 해석하는 게 맞다고 봅니다"로 표시한다. **그 표시는 정직하다.** 그러나 §5의 "어떤 memory level을 consolidate 할지", §6의 timescale 표, §9의 "HOPE CMS처럼 update frequency가 level마다 다르면"이 그 계층을 **표시 없이 전제로 받는다.** 특히 §9의 tiering 표는 L0–L4가 실재한다는 전제 위에서만 성립한다.

어느 논문에서도 오지 않은 눈금이다. NL의 실측 최저 주파수는 **2K 토큰**이고[NL §9.1], 그것이 연속체의 실증 상한이다.

### U3. "Validation / Replay → promote / rollback / forget"

lifecycle 다이어그램의 필수 단계로 그려지되, corpus에 대응 기제가 없다.

- ch27 C4: "$\Theta$-경로의 쓰기 검증은 문맥마다의 라벨 또는 이름 없는 외부 frozen reward model에 의존한다"[ch20 §20.7, ch21 §21.3.2]. `LM Need Sleep`의 semantic reward model은 "never named, sized, or counted"이고, **그것을 제거하면 AIME-25가 오른다(69.2 대 69.0)**[LM Need Sleep Table 1, p.11].
- ch27 판정 4 조건 (iii): "라벨 없이 쓰기를 판정하는 절차가 나오고 그 정확도가 측정되는 것" — corpus에 없다.
- rollback 단위의 소실은 $\Theta$-경로가 **지불하는 것**이지 갖고 있는 것이 아니다[ch27 표 27-3].

[문서]는 이 미해결을 상자 하나로 그린다.

### U4. "weight를 직접 건드리지 않으므로 rollback, safety, multi-user isolation이 쉽습니다" — [문서] §7 (2026–2027)

방향은 이 책과 같다 — $E$-경로는 shared-weight batching을 보존한다[ch27 A5, ch29 표 29-3 첫 행]. **"쉽다"가 미검증이다.** 반대 방향 증거가 이 책에 있다.

ch27 표 27-2 T8("파괴적 쓰기가 미검증 판정기와 결합한다"): ReasoningBank의 판정기 정확도는 **72.7%**이고 견고성 실험은 실제 판정기 편향 대신 대칭 잡음으로 대체했다. Mem0 Alg. 1의 DELETE는 동결 LLM의 감사되지 않은 판단으로 항목을 하드 삭제하는데 **삭제 정확도·오삭제율·DELETE 비활성 ablation이 전부 없다.** 롤백이 쉬운 것은 **층**이지 절차가 아니다.

### U5. HBF 매체 지목 — [문서] §9

ch29 §29.1이 명시적으로 답이 될 수 없다고 못 박은 것이다.

> "**소자 종류를 지목하는 것은 답이 아니다.** 요구의 형태(입도·마감·쓰기 주기)는 연역되지만, 어떤 매체가 그 요구를 만족시키는지는 이 책이 가진 어떤 증거로도 판정되지 않는다. 요구 사양을 쓰는 것과 제품을 고르는 것은 다른 일이며, 이 장은 앞의 것만 한다."

[문서]는 요구 형태를 건너뛰고 매체로 간다. 그리고 이 책이 실제로 연역한 요구 — 식 (29-3)의 DWPD가 라운드 주기 하나로 정해진다 — 는 [문서]의 tiering 논지에 나타나지 않는다.

### U6. "$C_{\text{total}} = C_{\text{prefill}} + C_{\text{decode}} + C_{\text{online update}} + C_{\text{consolidation}} + C_{\text{dreaming}}$" — [문서] §8

다섯 항을 한 합으로 놓는다. 이 책의 회계에서 그 합은 정의되지 않는다. 뒤 세 항은 서로 다른 분모로 나뉘고(식 (A)의 $N_q$가 문맥당·세션당·과제당으로 다르다, ch11 §11.3.1 [해설]), $B_s$가 wake 항에 결합하는 부호가 경로마다 다르다(ch28 판정 3). 이 책이 실제로 쓰는 형태는 라운드에 대한 합이다.

$$C_{\text{avg}} = C_{\text{wake}}(\hat c) + \frac{1}{N_q}\sum_k B_s^{(k)} \tag{28-2}$$

[문서]의 합은 $1/N_q$가 없다. 그 인자가 빠지면 **[문서] §5가 목적함수로 최적화하려는 대상 자체가 사라진다** — 언제 자야 하는가의 답은 정확히 그 분모가 정한다.

### U7. 서베이의 세 축이 "독립적"이라는 서술 — [문서] §7

서베이가 세 축을 독립 design axis로 **놓는 것**은 사실이다(→ 확증 A10). 그러나 서베이 자신의 Table I이 그 직교성을 반증한다 — `explicit / * / short-term` 셀이 **0편**이고, 구조적으로 빈 셀이 셋이다[`notes/stc-v2/memory-survey-2607.json` F3]. 그리고 서베이는 자기 1차 축이 정의되지 않는 지점을 스스로 인정한다.

> "Recent analysis of TTT with key–value binding argues that a broad class of such inner-loop updates can be reformulated as learned linear attention. This does not invalidate the memory interpretation of TTT-style modules, but it suggests that only variants with a clearly designated, autonomously updated storage substrate should be treated as explicit memory." [Memory for LLMs §IV-A-c, p.11]

즉 서베이의 1차 축은 **설계 의도**로 그어져 있고, 환원 정리가 기제의 일치를 보이면 포장으로 후퇴한다. [문서]는 이 축의 독립성을 자기 논지의 보강으로 인용하는데, 그 축은 [문서]가 딛는 바로 그 지점($W$층)에서 서베이가 미해결로 남긴 축이다.

### U8. "query predictability가 높을수록 sleep compute의 가치가 높다"를 스케줄러 입력으로 쓴 것 — [문서] §5

논문의 보고로서는 확증이다(A11). 스케줄러의 **입력**으로 쓰려면 실서빙에서 그 예측가능성을 측정해야 하고, 그 양의 회계적 대응물이 $N_q$인데 **실서빙 $N_q$ 분포를 보고한 논문이 0편이다**[ch25 §25.3.3]. 그리고 ch30 표 30-2는 그 한 줄이 인쇄되면 ch25 판정 1의 첫 층위가 해소되고 ch28 판정 2의 실무적 무게가 결정된다고 적는다 — **새 실험을 요구하지 않는 항목**이다.

---

## 4. 코퍼스 밖 — 이 책도 그 소스도 판정할 수 없는 것

| # | 항목 | 왜 판정 불가인가 |
|---|---|---|
| O1 | 2027–2030 단계 전망 자체(persistent adaptive state → autonomous self-improvement) | 미래 진술. ch27 §27.1 — "예측은 단정이 아니다. … 조건 없는 예측은 이 책의 문장이 아니다." [문서]가 "제 전망"으로 표시했으므로 결함이 아니라 범위 밖이다 |
| O2 | "Wake/Sleep dual-phase learning architecture"라는 용어가 자리 잡을지 | 용어 채택은 공동체의 일이다. ch24 §24.5 F6이 "새 논문의 정의문이 네 조건 중 몇 개를 명시하는지 센다"는 관측을 지정했을 뿐이다 |
| O3 | HBF 소자의 실제 지연·내구성·비용 특성 | ch29 §29.1이 소자 판정을 명시적으로 범위 밖에 둔다. 이 책에 wall-clock이 한 줄도 없다(ch30 §30.2) |
| O4 | "GPU를 background cognitive engine으로 쓰게 된다"는 운영 관행 예측 | 배포 관행에 대한 진술. 이 책은 스케줄러 정책의 **형태**만 연역한다(ch29 표 29-6) |
| O5 | HAT에서 Wake/Sleep phase를 first-class execution primitive로 만드는 판단 | 제품 설계 결정. 이 책의 증거는 요구 사양까지만 간다 |
| O6 | micro-sleep의 실제 지연 예산(100 ms~seconds 대역이 성립하는지) | 이 corpus에 그 대역을 측정한 논문이 없고, 이 책의 실험 넷 중 GPU 실측이 하나도 없다(ch30 판정 2) |
| O7 | "$I_{\text{interference}}$"라는 양의 정의와 단위 | 이 책의 비용 4종에 대응 칸이 없고, 서베이도 이름만 제안하고 단위를 주지 않는다[Memory for LLMs §VI-F, p.18]. **양쪽 코퍼스가 함께 비어 있는 유일한 항이다** |

---

## 5. 두 축 비교 — 목적별 축과 갱신 층별 축

이 검토에서 가장 값진 부분이다. 결론을 먼저 적는다.

> **두 축은 직교하지 않는다. 부분적으로만 교차한다. 목적 축은 판정 도구가 될 수 없고, 층 축의 두 번째 좌표로는 성립한다. 합치는 순서는 층 → 목적 하나뿐이며, 반대 순서는 이 책이 판별식으로 세운 것을 전부 잃는다.**

### 5.1 격자에 실제 논문을 놓는다

가로: [문서]의 목적 축 {Precompute, Consolidation, Dreaming}. 세로: 이 책의 층 축 {$E$, $W$, $\Theta$}. 칸에 넣는 것은 판별식 통과 10편(Auto-Dreamer 포함)이다.

| | **Precompute** | **Consolidation** | **Dreaming** |
|---|---|---|---|
| **$E$** | Letta STC | Mem0, Zep, ReasoningBank, SCM, Multi-Timescale, **Auto-Dreamer** | **비어 있음** |
| **$W$** | **비어 있음** | `Do LMs Need Sleep?` | **비어 있음** |
| **$\Theta$** | **비어 있음** | `LM Need Sleep`(Knowledge Seeding) | SEAL, `LM Need Sleep`(Dreaming) |

**아홉 칸 중 넷이 비어 있다.**[본서 관찰] 이것이 격자를 그리는 이유다 — 어느 축 단독으로도 이 네 공백은 보이지 않는다.

읽어야 할 것 셋.

1. **Precompute 열은 여전히 한 편이다.** [문서]가 "원 논문은 왼쪽 Precompute 하나에 가까웠다"고 쓴 것은 맞고, **2026-08 기준으로도 그 열은 그 한 편뿐이다.** $W$나 $\Theta$에서 질의 전 예상 계산을 한 논문이 0편이다. [문서]의 서사("2026년 연구들은 가운데와 오른쪽으로 빠르게 확장되고 있습니다")는 가운데 열에 대해서만 참이다.
2. **Dreaming 열은 $\Theta$에만 있다.** 우연이 아니라 구조다. 생성물을 소비하려면 gradient가 필요한데, $E$는 gradient를 쓰지 않고 $W$는 문맥과 함께 죽는다.[본서 추론] 그러므로 [문서]가 §7의 2028–2030 그림에서 "consolidation → KG/procedural memory/expert → counterfactual generation → synthetic trajectory → RL/distillation → LoRA/expert/slow CMS"로 그리는 파이프라인은 **마지막 두 화살표에서만 실증 사례를 갖는다.**
3. **Consolidation 행만 세 층을 덮는다.** 즉 [문서]의 "Sleep $\supset$ {Precompute, Consolidation, Dreaming}"은 세 원소가 대등한 분해처럼 보이지만, 격자 위에서 Consolidation 하나만 3층을 가로지르고 나머지 둘은 각각 한 층에 갇힌다. **세 원소를 대등하게 놓으면 이 비대칭이 사라진다.**

### 5.2 각 축이 더 잘 가르는 것

**목적 축이 더 잘 가르는 것: 실패 모드.**

같은 $\Theta$ 층 안에서 `LM Need Sleep`의 Knowledge Seeding과 Dreaming은 **다른 방식으로 실패한다.** 전자는 보존 실패(forgetting), 후자는 생성 오염(hallucinated learning)이다. 층 축은 이 둘을 같은 칸에 넣는다. 그리고 이 책은 그 구분을 **이미 쓰고 있다.**

- ch06: model collapse — $\mathcal{R}$이 자기생성일 때 걸리는 제약.
- X4 F6: replay를 보존된 원본에 앵커하면 모델 자기 출력에 앵커할 때보다 **모든 혼합비에서** 오래 버틴다.
- Continual Facts §6.2, p.14–15: $\Theta_0$ 고정 교사가 역량 +2pp·유지율 54%, 자기 누적 merge 교사가 −31pp·유지율 21%.
- ch29 판정 4: "반복이 성립하려면 replay가 보존된 원본에 앵커되어야 한다."

**이 네 항목이 전부 "Consolidation 대 Dreaming"의 경계에 서 있다.** 이 책은 그것을 축으로 부르지 않고 "원본 앵커 대 자기 출력 앵커"라는 국소 대비로만 썼다. [문서]의 공로가 여기 있다.

**층 축이 더 잘 가르는 것: 비용·용량·롤백·배칭 단위.**

- 식 (A)의 분모, $B_s$가 wake에 결합하는 부호, 상각 가능성 — 전부 층이 정한다(ch28 판정 3).
- 상태 바이트의 자릿수와 메모리 계층 배치 — 층이 정한다(ch26 판정 2, ch29 표 29-7).
- shared-weight batching이 어디서 어떻게 깨지는가 — 층이 정한다(ch29 표 29-3, 네 단계).
- sleep 라운드가 어떤 워크로드 클래스인가 — 층이 정한다(ch29 표 29-6: 배치 추론 잡 / 요청 내부 forward / 훈련 잡).

목적 축은 이 넷 중 어느 것도 결정하지 못한다. Consolidation이라는 같은 목적이 $E$에서는 오프라인 LLM 배치 잡이고 $W$에서는 선점 불가능한 요청 내부 단계다.

### 5.3 목적 축이 판정 도구가 될 수 없는 이유

**목적 축은 자기 규정을 입력으로 받는다.**

> "판별식이 **묻지 않는 것** 두 가지를 먼저 밝힌다. 첫째, **논문의 자기 규정을 묻지 않는다.** 제목에 sleep이나 consolidation이 들어 있는지, 저자가 자기 절차를 무엇이라 부르는지는 판정에 들어오지 않는다." (ch11 §11.2.1)

SleepGate를 목적 축에 놓으면 어디로 가는가. 제목이 "Sleep-Inspired Memory Consolidation"이고 method 이름에 sleep이 있고 부록 C에 5행짜리 생물 대응표가 있으므로 **Consolidation 칸에 들어간다.** 그 순간 이 책이 판정한 사실 — 어떤 층도 바뀌지 않고, 비용이 전부 $L_w$에 실리며, evict 변종은 실행된 적이 없다 — 이 전부 사라진다. NL도 같다. CMS가 "Continuum Memory System"이므로 목적 축에서는 Consolidation이다.

**그리고 이것이 이 책의 척추 논지가 겨냥한 병이다.**

> "**'sleep-time compute'은 축이 아니라 세 개의 서로 모르는 축이며, 그 이름은 기제보다 넓다.**" (PRE-RESEARCH §5)

[문서]는 §3에서 "Sleep Compute $\supset$ {Precompute, Consolidation, Dreaming}으로 보는 게 가장 정확합니다"라 쓰고 §1의 lifecycle 박스 전체를 Sleep-time Compute로 부를 것을 제안한다. **이름을 더 넓히는 처방이다.** ch24 판정 2가 진단한 병이 정확히 이름이 기제보다 넓다는 것이고, SleepGate가 방금 그 진단의 외부 표본이 되었다 — ch24 §24.5 F6의 계수 절차("새 논문의 정의문이 네 조건 중 몇 개를 명시하는지 센다")를 이 corpus의 가장 새로운 논문에 돌리면 **0/4**다.

### 5.4 합칠 수 있는가 — 예. 순서가 정해져 있다

**층 축이 먼저, 목적 축이 그 안의 두 번째 좌표로.** 이유 셋.

1. **목적 축 단독은 판정 도구가 아니다**(§5.3). 층 축으로 먼저 거르지 않으면 SleepGate·NL·Memory Caching이 들어온다. 그리고 ch30 §30.3.3 변형 (3)이 그 대가를 이미 계산했다 — 조건 (3)을 완화하면 H2O·StreamingLLM·sliding window·FastGen까지 들어와 "벤치마크도 비용 단위도 인용도 공유하지 않는 큰 네 번째 집단"이 생긴다.
2. **목적 축은 회계로 번역되지 않는다**(§5.2 후반).
3. **목적 축이 회수하는 것은 층 축이 놓치는 정확히 하나** — 자기생성 데이터가 들어오는 자리다. 그 자리는 이 책이 이미 네 곳에서 다루면서 축으로 세우지 않았다(§5.2 전반).

**권고.** ch24에 "층 × 목적" 2차 격자를 부록 표로 신설한다. **축이 아니라 층 축의 두 번째 좌표로** 도입하고, 격자의 빈 칸 넷을 명시한다. 이 격자는 판별식을 대체하지 않고 판별식 **통과 이후에만** 적용된다.

---

## 6. 새 논문 3편의 영향

세 편 모두 이 책의 판정을 **바꾸지 않고 계수를 바꾼다.** 방향은 셋 다 이 책 쪽이다.

### 6.1 Auto-Dreamer (2605.20616) — 통과 10편째, 그리고 판정 2에 대한 유일한 위협

**판별식: PASS ($E$).** 결정하는 조건은 (3)이다 — GRPO로 14B 모델을 학습하지만 그 대상은 쓰기 연산자 $\theta_C$이고, 배포 시 움직이는 것은 memory bank뿐이다. "only the consolidator parameters are updated during training, and only the memory bank is updated after deployment"[Auto-Dreamer §4, p.4].

**위협**: corpus에서 **$E$-경로 기억 시스템이 품질 1위인 첫 사례**다. ScienceWorld 41.07 [37.53, 44.70] 대 No memory 28.69 [25.18, 32.14] — 비겹침. ALFWorld 60.21 [54.65, 65.59] 대 30.83 [26.40, 35.31] — 비겹침. ch27 A2의 "반복적으로"에 반례가 생겼고, 그것도 비용 열이 아니라 품질 열의 반례다.

**그럼에도 ch27 판정 2는 선다.** FE2가 요구한 셋 중 하나만 충족된다. (i) **full-context 팔이 없다** — 전문 grep 0회이고, top-$k{=}3$/1500 토큰 회수 예산 아래에서 full-context 팔은 설계상 배제된다. (ii) bootstrap CI는 인쇄되되 시드가 없다 — "We report point estimates without seed or task-order variance"[동 App. A, p.15]. (iii) 대화 길이 스윕이 없다. **판정이 서는 이유가 정확히 ch27이 FE2를 그렇게 쓴 이유다.**

**확증 셋.** (a) 독립 재평가가 이 책의 $E$-경로 진단을 재현한다 — 같은 frozen task agent·같은 writer·같은 encoder로 열 시스템을 재실행한 결과, ScienceWorld에서 Mem0 26.79 < No memory 28.69, ExpeL 28.33 < 28.69, LightMem 28.08 < 28.69이고 열 baseline 전부의 95% 구간이 memoryless 팔과 겹친다. (b) CLS를 1차 문헌으로 인용하고 면책까지 인쇄하는 **첫 논문**이다 — "We adopt CLS not as a biological claim about language models, but as an operational design principle"[동 §1, p.2]. ch23 §23.8이 요구한 구분을 실제로 한다. (c) $E$-경로에 **세 고리짜리 사슬이 처음 성립한다** — Letta STC 인용, LightMem을 이길 대상으로 지목, Letta STC §7이 미룬 다라운드 레짐 실행.

**비용 실태는 악화시킨다.** $B_s$가 absent이고, 논문은 그것을 로그로 **측정해 놓았다**. $C_{\text{cap}}$을 토큰으로 11개 시스템 × 3 도메인에 걸쳐 인쇄한 것은 corpus 최고의 상태량 보고이지만(WebArena에서 LightMem 370,874 토큰 대 Auto-Dreamer 927 토큰), 바이트도 사용자당 값도 아니므로 ch26 판정 6은 살아남는다.

### 6.2 SleepGate (2603.14517) — 척추 논지의 외부 표본

**판별식: FAIL, 조건 (3).** B군, Memory Caching 옆에 놓는다(→ §2 R4).

**ch24 판정 2에 대한 가장 강한 확증이다.** 제목에 sleep, method 이름에 sleep micro-cycle, 부록 C에 5행 생물 대응표를 갖춘 논문이 (i) 세 층 어느 것도 바꾸지 않고, (ii) 비용의 100%를 $L_w$에 싣고 $B_s$에 0을 싣고, (iii) **이 책이 이 범주로 지목한 아홉 편 중 한 편도 인용하지 않는다** — 11개월 앞서 이 이름을 만든 Letta STC를 포함해서. 판정 2의 조건절은 "이 책이 원문 대조로 읽은 29편에 대한 것"인데, **30번째가 바깥에서 도착해 아무것도 읽지 않은 채 같은 방식으로 행동한다.** 이것은 또 다른 사례가 아니라 독립 재현에 가깝다.

**침묵 지도의 반전 사례이기도 하다.** 기존 행들은 생물의 이름만 빌리고 논증은 빌리지 않은 논문들이다. 이 행은 거울상 — 생물·CLS 11편을 인용하고 §1.1에 기제 수준 서술을 두면서, 이 라인에는 완전히 침묵한다. 그리고 새 열이 필요하다 — 이 논문이 유일하게 조밀하게 읽은 문헌은 KV cache·서빙 계열이며, corpus의 다른 어떤 논문도 그 문헌을 읽지 않는다.

**이 책의 apparatus 한 곳을 위협한다.** ch11 §11.2.4가 회색지대를 둘로 선언했는데 셋째가 있다 — **세 층 중 어디에도 속하지 않는 상태(KV cache)를 바꾸는 절차.** 적용 절차 2단계("배포 아티팩트이면 $\Theta$, 요청·세션에 붙는 가중치 형태의 상태이면 $W$, 모델 밖 레코드이면 $E$")가 상태 전체를 덮지 않는다. 판정은 옳게 나오지만 이유가 인쇄되지 않는다. ch17 표 17-2가 이미 "| KV cache $K_t,V_t$ | 비모수 상태 |" 행을 갖고 있다는 사실이 이 공백이 실재함을 보인다.

### 6.3 Memory for LLMs 서베이 (2607.25380) — ch18의 외부 확인

**§0에서 이미 적었다. 판정은 CONFIRMS다.** 추가로 셋을 기록한다.

1. **두 프레임은 거의 닿지 않는다.** 이 책의 29편 중 서베이가 인용하는 것은 3편이고, 판별식 통과 10편 중 **0편**이다. 여덟 편이 선언된 범위에서 배제된다. 반대로 서베이의 lookup/routing 계열(kNN-LM, PlugLM, Engram, ExplicitLM, MoE)은 이 책의 프레임에서 보이지 않는다. **두 축은 서로가 날카로운 자리에서 각각 눈이 먼다.**
2. **이 책의 축이 이기는 자리가 둘 있고, 하나는 서베이 자신의 인정으로 이긴다.** (i) NL과 Memory Caching — 같은 저자·같은 해·이 책에서 한 장인데 서베이는 최상위 축에서 갈라 놓는다(explicit 대 implicit). 갈린 것은 포장이고 갱신 규칙은 같다. (ii) Titans 대 Mamba/Gated DeltaNet — 서베이는 explicit/implicit로 가르는데, 환원 정리 앞에서 "clearly designated… substrate"로 후퇴한다[§IV-A-c, p.11]. **(U-W)는 포장에 무관하다.**
3. **비용 침묵이 필드 수준에서 재현된다.** 47개 시스템을 다루는 20쪽 서베이가 **측정값을 0개** 보고하고, memory write가 계산을 얼마나 쓰는지 한 번도 묻지 않는다. 네 평가 축(capacity/fidelity/persistence/efficiency)을 제안하면서 **어느 축에도 단위를 주지 않는다.** X3이 corpus 안에서 잰 진공이 필드의 자기 서술 수준에서 그대로 재현된다.

한 가지는 이 책이 받아야 한다. 서베이의 "retrieval fidelity" 축에 **이 책의 비용 4종은 대응 칸이 없다.** 같은 바이트를 들고도 회수 정밀도가 다를 수 있고, 이 책은 그 자리를 갖고 있지 않다.

---

## 7. 이 책에 필요한 수정 목록

장 번호와 문장 수준으로 적는다. **판정은 하나도 바뀌지 않는다. 계수·인용·범위 한정어가 바뀐다.**

### 7.1 계수 (기계적, 전 장)

**corpus 29 → 32. 통과 9 → 10. 불충족 19 → 21.**
- 통과군에 Auto-Dreamer($E$) 추가. 경로 분포 $E$ 6/$W$ 1/$\Theta$ 2 → **$E$ 7/$W$ 1/$\Theta$ 2**(경계 사례 포함 시 $E$ 8).
- B군에 SleepGate 추가(탈락 조건 (3)). C군에 서베이 추가.
- 대상: PRE-RESEARCH §2.2 카운트 규약, ch11 §11.3 집계 규약·표 11-1/11-2/11-3, ch11 §11.3.1, ch24 판정 1, ch27 §27.3.2 서두("$E$-경로에는 판별식 통과 논문 9편 중 6편이 있다"), ch28 표 28-2 모수, 부록 C, S5 조립 QA 체크리스트.
- **ch28 판정 1은 강해진다** — 모수가 29→32로 늘고 (a)범주는 1편 그대로다. ch24 B1의 "$B_s$ absent 19/29"는 21/32가 된다.
- ch30 §30.3.3 마지막 문단 "A군 9편이 한 편으로 줄고"를 "10편이 한 편으로 줄고"로.

### 7.2 ch18 — 외부 확인 두 인용을 §18.4에 추가

§18.4의 [평가] 블록 뒤에 한 문단. 서베이 §IV-A-b p.10과 §V-A-d p.15를 직접 인용하고, 서베이가 NL을 Table II "Optimization-based writing" 행에 TTT-E2E·Titans·In-Place TTT와 함께 놓았다는 사실을 적는다. 함께 적을 것 — 서베이가 NL에 'Explicit'·'Long-Term' 라벨을 **증거 없이** 주었고, 그 라벨은 서베이 자신의 §III-D-c(cached recurrent snapshots는 "primarily improve retention within an extended sequence rather than introducing fully persistent memory across sessions")와 어긋난다. **주파수 계층이 얼마나 쉽게 2층 기억 시스템으로 쓰이는지의 사례이며, ch18이 존재하는 이유가 그 오독을 막는 것이다.**

### 7.3 ch18 §18.8.2 — 예측된 오독이 실제로 일어났다는 관측 기록

새 [평가] 한 항. ch18의 판정이 예측한 오독 — 축이 있다는 사실만으로 $W$-경로의 결과를 $\Theta$-경로로 옮겨 읽는 것 — 이 2026-08의 외부 전망 문서에서 실제로 관측되었다. 관측의 형태를 적는다: NL의 어휘(CMS, level, update frequency)를 빌려 NL이 구현하지 않은 세 층(episodic/semantic, LoRA/expert, stable parametric)을 얹고, 서베이의 "frozen or adapt slowly"에서 "frozen or"를 떨어뜨린다. **이 기록은 ch18의 반증 관측(표 30-3: "연속체의 저주파 끝점에서 $\Theta$가 실제로 오프라인에 움직이는 구현과 수치가 제시됨")이 여전히 발화하지 않았다는 확인이기도 하다.**

### 7.4 ch21 §21.6 — CMS 위에서 돌지 않은 절반

본문에 수치와 함께 싣는다: "Sec. 3은 CMS/MoE backbone 위에서 전부를 정의하는데, Sec. 4.1의 consolidation 결과는 Hope에서 나오고 **Dreaming 결과는 'Sleep (Transformer)'·'Sleep (Transformer + four-level)'[LM Need Sleep Table 3, p.12]과 Llama-3.2-1B[동 App. B.3, p.25]에서 나온다.**" 이 한 문장이 "CMS는 이걸 구현하기 좋은 abstraction"류 주장의 직접 반증이며, 현재 `unfavorable_facts` 7번에만 있고 본문에 없다.

### 7.5 ch11 §11.2.4 — 회색지대를 셋으로

셋째: **세 층 중 어디에도 속하지 않는 상태를 바꾸는 절차.** KV cache 관리 문헌 전체가 여기 해당하며, 판별식은 이들을 조건 (3)에서 탈락시키되 그 탈락이 "아무것도 바꾸지 않아서"가 아니라 "이 책이 모형화하지 않은 것을 바꿔서"임을 밝혀야 한다. 함께: 표 11-2에 SleepGate 행 추가(`| SleepGate | 2603.14517 | (3) | (R)의 $f$를 바꾼다. 평가된 변종은 캐시를 고치지 않고 bias만 만든다 [SleepGate Eq. 10–11; Algorithm 1 lines 16–21] | ch11·ch18 |`), ch01 층 표에 네 번째 행(이 책의 세 경로 밖) 표시, §11.2.3 조건 (3) 문단에 두 번째 예시 추가.

### 7.6 ch27 판정 2 — 레짐 한정어를 붙인다

현재: "$E$-경로는 유망하다. 단 비용 방법으로서다. **품질 방법으로서는 현재 문헌이 지지하지 않으며**, 그 반증은 이 책이 아니라 논문들 자신의 표에서 나온다."

수정: 품질 절을 두 레짐으로 가른다. **대화형 QA 레짐**에서는 품질 1위가 반복적으로 기억 시스템 없는 구성이고(Mem0·Zep·SCM·ReasoningBank), **다세션 에이전트 스트림 레짐**에서는 $E$-경로 시스템이 memoryless 팔을 비겹침 구간으로 이기지만 **full-context 팔을 실행한 논문이 아직 0편이다.** 정직한 형태는 "한 레짐에서 미지지, 다른 레짐에서 미검증"이다. §27.5 FE2에 한 줄 추가 — Auto-Dreamer가 FE2의 세 요구 중 하나만 충족하므로 발화하지 않는다.

### 7.7 ch27 표 27-2 — 열한 번째 행: 독립 재평가

새 유형이 아니라 새 증거 종류다. 같은 frozen task agent·같은 writer LLM·같은 encoder·같은 프롬프트 형식으로 Mem0·ReasoningBank·LightMem·AWM·Memp·ExpeL·Reflexion·Mem-$\alpha$·UMEM을 제3자가 재실행한 결과가 이 책의 진단을 재현한다. ScienceWorld continual: Mem0 26.79 · ExpeL 28.33 · LightMem 28.08 대 No memory 28.69, 열 baseline 전부의 95% 구간이 memoryless 팔과 겹침[Auto-Dreamer Table 1 Panel A].

### 7.8 ch27 §27.5 FW1 — SleepGate를 선제 배제

한 문장: 캐시 상태를 조작하므로 $W$-경로 두 번째 표본 후보로 보이지만, 네 조건 중 어느 것도 $W$로 통과시키지 않고 fast weight가 없으며 논문 자신이 고정 크기 상태 계열에 반대편으로 선다[SleepGate §8.2]. **$W$-경로 표본은 1로 유지된다.**

### 7.9 ch23 §23.8 — 양방향 갱신

(i) Auto-Dreamer가 CLS를 1차 문헌(McClelland 1995 / Kumaran 2016 / McClelland 2020)으로 인용하고 이 장이 요구한 면책을 인쇄한다. **이 장이 요구한 것을 실제로 한 첫 논문이며, "결함은 이름의 재사용이다"의 일반화는 이제 날짜를 달아야 한다.** (ii) SleepGate는 거울상 — 생물·CLS 11편을 읽고 이 라인 아홉 편을 하나도 읽지 않는다. 두 사례를 함께 싣는다.

### 7.10 ch14·ch15 — 계보 서술의 시제

"Letta STC를 인용하는 후속 논문이 없다"를 날짜를 붙인 형태로 바꾼다. Auto-Dreamer가 Letta STC를 인용하고, LightMem을 "closest architectural counterpart"로 지목하며, Letta STC §7이 미룬 다라운드 레짐을 실행한다(ScienceWorld 약 66라운드, ALFWorld 약 34, WebArena 약 23[본서 산술]). **$E$-경로에 세 고리짜리 사슬이 처음 성립한다.** ch15 §15.6과 PRE-RESEARCH §3.1의 침묵 지도가 같은 처리를 받는다. 경로 **간** 침묵은 그대로다 — Auto-Dreamer 전문에 `KV` 0, `fast weight` 0, `test-time training` 0, `Titans` 0, `LoRA` 0, `gradient` 0이다.

### 7.11 ch28 §28.2.1·§28.3.4 — "측정해 놓고 인쇄하지 않았다"

표 28-2에 Auto-Dreamer를 (f)로 추가하되 특기한다: `summary.json`이 per-role LLM call과 token count를, `dreamer_calls.jsonl`이 task별 writer/dreamer 이벤트를 기록한다[Auto-Dreamer App. B.2, p.18]. **$B_s$는 저자의 로그 안에 있고 표에도 그림 축에도 문장에도 없다.** §28.3.4의 "관습의 부재" 진단에 corpus에서 가장 날카로운 사례다. 함께 ch26 판정 6에 SleepGate 추가 — 917,313 파라미터 모델에 Table 3이 하이퍼파라미터를 전부 인쇄하므로 $B_s$·$C_{\text{cap}}$ 두 칸 모두 곱셈 한 번인데 둘 다 없다.

### 7.12 ch29 §29.3.7 — wake/sleep 하드웨어 파티션이 경로마다 성립하지 않는다

표 29-6 뒤에 한 문단. 가속기를 "latency-critical wake partition + throughput-optimized sleep partition"으로 나누는 서술은 $E$·$\Theta$에서 성립하고 $W$에서 성립하지 않는다 — $W$의 sleep은 요청 안에서 일어나 유휴로 밀 수 없고 선점 불가다. **이 구분이 흐려지는 것이 이 분야 HW 서술의 가장 흔한 과장이며, 표 29-6 셋째 열이 이미 그 근거를 담고 있으나 명시적 반례 서술이 없다.**

### 7.13 ch29 §29.5 — 반증 항목 R-10 신설

현재 ch29 판정 2("두 요구는 합쳐진다")의 반증 관측이 R-7 하나뿐이다. 하나 더: **세 경로가 함께 배포된 구성에서 wake 쪽 상태 쓰기가 실제로 0이 되는 측정.** "wake=read-only, sleep=controlled write" 구성이 구현 가능하다면 판정 2의 합산 논증이 약해진다. 관측 형태 — $W$-경로 fast weight를 sleep 국면으로만 밀고 wake에서 동결했을 때의 품질 손실과 토큰당 지연.

### 7.14 ch24 — "층 × 목적" 2차 격자 부록표 신설

§5.4의 권고를 실행한다. 판별식 통과 논문에만 적용되는 **두 번째 좌표**임을 명시하고, 빈 칸 넷(Precompute×$W$, Precompute×$\Theta$, Dreaming×$E$, Dreaming×$W$)을 관측 형태로 적는다. 이 표는 ch27 판정 6의 "세 경로 합성" 논의와 ch30 표 30-2의 새 결과 유형 목록에 각각 한 줄을 보탠다.

### 7.15 ch30 표 30-2 — 행 둘 추가

- "판별식 통과 논문이 자기 $B_s$를 로그로 측정해 놓고 인쇄하지 않은 사실이 확인됨" → ch28 판정 6(관습의 부재)이 추론에서 관측으로 승격된다. **새 실험 불요.**
- "외부 서베이가 NL을 fast-weight 계열로 분류하고 backbone을 정지로 둠" → ch18 판정이 외부 확인을 얻는다. **새 실험 불요.**

그리고 §30.3.4에 실행 기록 한 문단: 이번 세 편은 표 30-2의 아홉 유형 중 어느 것에도 해당하지 않았고, **판정을 하나도 바꾸지 않고 계수만 바꿨다.** 이것이 표 30-2가 지정한 관측의 정밀도를 확인한다.

### 7.16 이 책이 채택하지 않는 것

명시적으로 기록한다. (i) [문서]의 CMS-L0~L4 계층 — R1. (ii) Sleep $\supset$ {P, C, D}를 **1차 축**으로 쓰는 것 — §5.3. (iii) HBF 등 매체 지목 — ch29 §29.1이 범위 밖으로 선언했다. (iv) 서베이 §VI-E를 ch29의 warrant로 인용하는 것 — **수렴하는 외부 의견으로만 인용하고 근거로는 절대 인용하지 않는다.** 서베이 §VI-E에는 수치도 측정도 시스템 연구 인용도 없다. ch29의 warrant는 X2 하나로 유지한다(NM D4 제약 승계).

---

## 8. 이 검토 자체의 한계

**한계 1 — 원문 대조 범위.** 새 논문 3편에 대한 이 검토의 판정은 `notes/stc-v2/`의 deep-read 노트에 의존한다. 노트는 원문 로케이터를 달고 있으나, 이 검토가 세 편의 vendored 원문을 직접 다시 훑지는 않았다. X3이 자기 최약 고리로 지목한 것과 같은 구조다 — "노트가 놓친 값은 이 감사도 놓친다"[ch28 §28.5 항목 2]. 특히 SleepGate의 trigger 모순(UF-6)과 `LM Need Sleep`의 Dreaming 백본(unfavorable_facts 7번)은 판정의 무게가 크므로, 본문 반영 전 원문 재확인이 필요하다.

**한계 2 — [문서]의 의도를 판정하지 않았다.** 이 검토는 [문서]가 무엇을 주장하는지를 판정했고 **왜 그렇게 썼는지는 판정하지 않았다.** 특히 §4의 CMS 해석은 "이렇게 해석하는 게 맞다고 봅니다"로 표시되어 있으므로, [문서]가 그것을 NL의 기술적 사실로 주장한 것인지 규범적 제안으로 놓은 것인지는 문면으로 갈리지 않는다. 이 검토는 **후속 절들이 그것을 전제로 쓴다**는 사실만으로 결함을 셌다. 저자가 "제안을 제안으로만 썼다"고 답하면 §3 U2의 절반이 사라진다.

**한계 3 — 값을 다투지 않았다.** 이 검토의 모든 판정은 이 책의 등급 규칙을 그대로 승계하므로 **형태 판정**이다 — 부재, 구조 분류, 식의 성질, 자릿수, 순서, 부호. [문서]의 HW 논지가 실무적으로 얼마나 무거운지, HBF 계층이 실제로 어느 정도 성립하는지는 이 검토가 답하지 않으며 답할 수 없다. ch30 판정 2의 대가가 여기서도 그대로다 — "이 책은 자기 판정의 실무적 무게를 스스로 확정할 수 없다."

**한계 4 — 격자는 이 검토의 산물이지 검증된 분류가 아니다.** §5.1의 층×목적 격자는 [문서]의 목적 축을 이 책의 통과군 10편에 적용한 것인데, 목적 배정 자체가 §5.3이 지적한 문제 — 자기 규정 의존 — 를 완전히는 벗어나지 못한다. Auto-Dreamer를 Consolidation에 놓은 것은 논문의 기제(region replacement, 생성 성분 0)에 근거하지만, SCM의 REM dream을 Consolidation에 넣은 것은 그 성분의 ablation이 수치를 바꾸지 않는다는 사실[ch27 T3]에 기댄 판단이다. **빈 칸 넷은 견고하고, 찬 칸의 배정 중 둘은 재검토 대상이다.**

**한계 5 — 두 축의 우열 판정에 이 책 편향이 있을 수 있다.** §5.2·§5.3은 층 축이 판정 도구로 우월하다고 단정했고, 근거는 ch11 §11.2.1(자기 규정 비입력)과 ch28 판정 3(부호가 층으로 정해짐)이다. 두 근거 모두 **이 책이 정한 규칙**이다. 판별식이 자기 규정을 묻지 않기로 한 것이 이 책의 선택이므로(ch30 판정 4), 다른 선택 위에서는 목적 축이 우월할 수 있다. 이 검토는 그 가능성을 계산하지 않았다 — ch30 §30.3.3이 계산한 네 변형에 "목적 축을 1차로 놓는 다섯째 변형"이 없다. **M4의 후보다.**

**한계 6 — [문서]가 인용한 두 편을 이 책이 읽지 않았다.** LightMem과 UMEM은 Auto-Dreamer가 자기 계보의 직전 항으로 지목하는 논문인데 이 corpus에 없다. LightMem은 "online writer + periodic offline consolidation"으로 서술되므로 판별식을 정면 통과할 가능성이 있고, 통과하면 $E$-경로 계수가 다시 바뀐다. **이 검토가 확인하지 않은 채로 남긴 가장 큰 공백이다.**

---

## 9. 한 문단 결산

[문서]는 이 책이 판정한 것을 뒤집지 못한다. 여섯 반증 중 다섯이 [문서]의 뼈대에 걸리고, 그 뼈대에 대해 이 책·외부 서베이·NL 논문 자신이 갈리지 않는다. 그러나 [문서]는 이 책이 **갖고 있으면서 세우지 않은 좌표** 하나를 정확히 짚는다 — 보존과 생성은 실패 모드가 다른 두 연산이다. 이 책은 그것을 ch06·ch22·X4·ch29 판정 4에서 네 번 쓰면서 축으로 부르지 않았다. 층 축의 두 번째 좌표로 놓고 격자를 그리면 아홉 칸 중 넷이 비고, 그 넷이 이 라인의 실제 지도다. **이 검토가 이 책에 요구하는 유일한 실질 변경이 그것이며, 나머지 열다섯은 계수와 인용과 범위 한정어다.**
