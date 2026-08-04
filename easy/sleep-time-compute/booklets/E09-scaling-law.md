# E09 · Sleep-time scaling law는 어떻게 세워야 하는가

## 이 권이 답할 질문

Pretraining에는 compute·data·parameter와 loss의 경험적 scaling law가 있다. Sleep-time compute에도 “더 오래 자면 더 똑똑해진다”는 법칙이 생길까? 현재 답은 **아직 universal law가 없다**다. 먼저 cost accounting identity, reuse break-even, cadence optimum, capacity knee를 분리하고, 이후 workload별 empirical fit을 쌓아야 한다.

이 권은 {{STUDY:STC-S12|scaling-law section}}을 풀며 {{CLAIM:STC-C041}}, {{CLAIM:STC-C043}}, {{CLAIM:STC-C044}}, {{CLAIM:STC-C045}}을 구분한다.

## 먼저 세 문장

1. Raw sleep FLOP, 처리한 memory 수, 생성한 dream token은 좋은 scaling variable이 아니다. 잘못된 기억을 많이 처리하면 utility가 음수가 될 수 있다.
2. 핵심 후보는 **valid reusable evidence per unit sleep cost**다. Validity, reuse, retention, staleness, interference를 함께 측정해야 한다.
3. Sleep cadence와 memory capacity에는 interior optimum이 있을 가능성이 높다. 너무 자주 갱신하면 overhead·noise가 크고, 너무 늦으면 staleness가 커진다.

## 직관

### Identity와 law를 구분한다

다음은 정의에 가까운 accounting identity다.

$$C_{\mathrm{sleep}} = C_{\mathrm{read}} + C_{\mathrm{transform}} + C_{\mathrm{train}} + C_{\mathrm{eval}} + C_{\mathrm{publish}}$$

각 항을 더하면 총비용이 된다는 것은 항상 참이지만 성능이 어떻게 변하는지는 말하지 않는다. Scaling law라면 여러 규모에서 측정한 data로 예측력이 있어야 한다. 예를 들어 utility gap이 sleep compute의 power law로 줄어드는지, 어느 점에서 saturation하는지 fit하고 out-of-sample에서 맞아야 한다.

### Break-even은 reuse 문제다

External evidence를 매번 retrieve하는 비용을 $c_r$, 한 번 sleep/compile하는 비용을 $c_s$, compile 후 매 query 절약을 $\Delta c$라 하자. 단순 break-even reuse는 다음처럼 생각할 수 있다.

$$N^{\star} \approx \frac{c_s + c_{\mathrm{governance}}}{\Delta c}$$

$N>N^{\star}$일 때만 compile이 경제적이다. 하지만 quality loss와 rollback risk를 cost에 포함해야 한다. 이 식은 measurement model이지 검증된 universal law가 아니다.

### 좋은 분모와 좋은 분자

분모에는 GPU FLOP만 넣지 않는다. Joule, wall-clock, accelerator-hour, HBM/SSD bytes moved, index write amplification, validation inference를 기록한다. 분자에는 raw memory count 대신 future task에서 실제로 사용된 **valid retained evidence**를 둔다.

한 후보 metric은 다음과 같다.

$$\eta_{\mathrm{STC}} = \frac{U_{\mathrm{future}} \times R_{\mathrm{retained}} \times P_{\mathrm{valid}}}{C_{\mathrm{sleep}} + C_{\mathrm{serve\ overhead}} + C_{\mathrm{rollback}}}$$

여기서 $U$는 future utility, $R$은 retention, $P$는 provenance/validation을 통과한 비율이다. 이 역시 law가 아니라 연구를 위한 normalization proposal이다.

### Cadence optimum

Sleep interval $\tau$가 작으면 fresh하지만 job launch와 validation이 너무 자주 일어난다. $\tau$가 크면 batch efficiency는 좋아지지만 memory가 stale해지고 wake state가 넘칠 수 있다. Traffic과 update entropy가 변하므로 optimal $\tau$는 constant가 아닐 가능성이 높다.

### Capacity knee

Memory byte $M$을 늘릴 때 초반 utility는 증가하지만 중복·interference·retrieval noise·compaction cost 때문에 marginal gain이 꺾일 수 있다. {{BG:capacity-budget|용량 예산}}의 knee를 넘으면 더 큰 device보다 admission·eviction policy가 중요해진다.

## 예와 반례

### 예 — Adapter compile break-even 실험

동일한 knowledge bundle을 RAG로만 제공하는 system과 adapter로 compile한 system을 비교한다. Reuse 횟수를 1, 10, 100, 1000으로 바꾸고 answer quality, TTFT, energy, adapter load, delete cost를 잰다. Break-even은 quality constraint를 만족하는 최초 reuse 지점이다. Hardware가 바뀌면 $N^{\star}$도 바뀐다.

### 예 — Cadence sweep

Sleep interval을 1분, 1시간, 1일, 1주로 바꾸고 staleness, batch size, queue delay, wake SLA interference, derived-memory error를 측정한다. U-shaped total cost가 나오면 interior optimum 가설이 지지된다. 단일 cadence의 SOTA만 보고 law라고 부르면 안 된다.

### 반례 — Dream token을 10배 늘렸더니 benchmark +1

한 점의 결과는 scaling law가 아니다. 다른 model size, data quality, recursive depth, task에서 반복되고 held-out scale을 예측해야 한다. 또한 extra token이 중복이면 유효 evidence는 거의 늘지 않았을 수 있다.

### 반례 — Memory byte가 크면 capacity 문제 해결

Storage capacity를 늘려도 active index, cache hit, retrieval precision, write endurance가 병목일 수 있다. 유한한 latency와 energy 안에서 usable capacity를 재야 한다. “저장 가능한 byte”와 “정확히 찾아 활용 가능한 기억”은 다르다.

## 그림 읽기

{{FIG:STC-F007}}

Figure는 measured identity, break-even model, fit candidate, untested hypothesis를 색으로 분리한다. 수식이 있다는 이유만으로 theory가 되는 것을 막기 위한 장치다. 연구 보고서에는 각 equation의 status를 함께 적어야 한다.

{{FIG:STC-F008}}

Sleep cost는 한 번 앞에서 지불하고 later reuse에서 절약을 얻는다. Curve가 0을 지나는 지점이 break-even이다. External memory maintenance처럼 read cost를 줄이지 않고 quality를 높이는 경우에는 절약이 아니라 utility gain을 monetary/energy-equivalent로 환산해야 한다.

{{FIG:STC-F009}}

Capacity knee는 finite-memory system의 saturation 가설이다. Knee 이전에는 byte 추가가 유용하고, 이후에는 compaction·eviction·tiering이 더 중요하다. Device solution은 knee를 오른쪽으로 옮기거나 knee 이후 비용을 낮추는 제품으로 정의할 수 있다.

{{FIG:STC-F010}}

Cadence frontier는 staleness와 batching의 tradeoff를 보여 준다. Fixed daily sleep보다 event-triggered/learned cadence가 frontier를 개선할 수 있다. 그러나 scheduler를 학습하려면 delayed future utility를 현재 sleep decision에 연결하는 평가가 필요하다.

## 대안과 비교

| 후보 관계 | 현재 지위 | 필요한 축 | 반증 조건 |
|---|---|---|---|
| sleep FLOP↑ → utility↑ | 미검증 | model, data quality, task | 동일 compute에서 saturation/하락 |
| reuse↑ → compile 가치↑ | break-even model | access count, latency, quality | governance cost가 절약 초과 |
| cadence에 interior optimum | 가설 | staleness, batch, SLA | boundary가 항상 최적 |
| capacity knee 존재 | 가설 | byte, entropy, interference | marginal utility가 계속 일정 |
| valid evidence efficiency | metric proposal | utility, validity, cost | measurement가 불안정 |

{{BG:off-policy-evaluation|오프폴리시 평가}}는 실제로 다른 cadence를 배포하지 않고 logged policy에서 utility를 추정하는 후보지만, hidden confounding과 distribution shift에 취약하다. 초기에는 controlled sweep과 replay simulation이 필요하다.

## 아직 모르는 것

- Future utility를 memory write에 귀속하는 credit assignment가 미해결이다.
- External synthesis와 parametric training의 cost unit을 공정하게 통일하기 어렵다.
- Quality·safety·delete를 하나의 objective로 결합하면 weight 선택에 따라 결론이 달라진다.
- Model size가 커질 때 sleep sample efficiency가 좋아지는지 나빠지는지 공개 scaling sweep이 없다.
- Capacity knee가 device bandwidth, index algorithm, memory entropy 중 무엇에 더 민감한지 모른다.

> **핵심.** 첫 STC scaling law는 “sleep FLOP law”보다 “재사용되는 유효 evidence와 lifecycle 총비용의 관계”에서 나올 가능성이 높다.

## 더 깊이 읽기

- 원 분석: {{STUDY:STC-S12|accounting identity·break-even·hypothesis의 엄격한 구분}}
- Training Background: {{BG:objective|목적함수}}, {{BG:validation-set|검증 세트}}, {{BG:capacity-budget|용량 예산}}, {{BG:rate-distortion|율–왜곡}}, {{BG:off-policy-evaluation|오프폴리시 평가}}
- Claim route: {{CLAIM:STC-C041}} · {{CLAIM:STC-C043}} · {{CLAIM:STC-C044}} · {{CLAIM:STC-C045}}
