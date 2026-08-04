# E05 · Sleep phase 안에서 실제로 무엇을 계산하는가

## 이 권이 답할 질문

“자는 동안 학습한다”를 실제 연산 단계로 풀면 무엇이 남는가? Sleep phase는 하나의 algorithm이 아니라 data construction, selection, replay, distillation, adapter optimization, evaluation, state publication을 엮은 pipeline이다. 어느 단계를 쓰느냐에 따라 compute와 memory traffic이 완전히 달라진다.

이 권은 {{STUDY:STC-S07|sleep mechanism taxonomy}}를 따라 {{CLAIM:STC-C015}}, {{CLAIM:STC-C019}}, {{CLAIM:STC-C023}}, {{CLAIM:STC-C024}}, {{CLAIM:STC-C035}}을 연산 관점으로 설명한다.

## 먼저 세 문장

1. External-memory sleep의 기본 loop는 **episode 수집→추출→중복 제거→충돌 해결→summary/graph/index 갱신→검증**이다. Gradient가 없어도 generation과 search가 크다.
2. Parametric sleep은 **replay data 구성→teacher/student 또는 task loss 계산→backward→optimizer→evaluation→adapter/weight version 배포**다. LoRA로 weight write를 줄여도 activation과 data movement는 남는다.
3. Dreaming이나 synthetic augmentation은 data 부족을 줄이지만, 자기 output을 다시 학습하는 recursive loop이므로 real evidence anchor와 quality gate가 필요하다.

## 직관

### Mechanism A — Replay

과거 data 또는 그것을 흉내 낸 sample을 새 data와 섞어 학습한다. 원 sample을 보존하면 fidelity가 높지만 storage·privacy cost가 있다. {{BG:synthetic-data|합성 데이터}}로 replay하면 저장량을 줄일 수 있으나 rare case와 failure mode를 generator가 놓칠 수 있다. Replay는 “기억을 다시 보여 주는 data plane”이다.

### Mechanism B — Distillation

Teacher의 output distribution이나 behavior를 student state로 옮긴다. {{BG:distillation|지식 증류}}는 raw label보다 richer signal을 줄 수 있고, large context/fast memory를 compact adapter로 옮기는 데 쓰인다. 하지만 compression은 무엇을 버릴지 결정한다. Teacher가 틀렸거나 student capacity가 부족하면 오류도 함께 압축된다.

### Mechanism C — Data augmentation과 dreaming

Model이 스스로 candidate experience를 만들고 evaluator가 useful sample을 고른다. {{BG:data-augmentation|데이터 증강}}의 장점은 실제 log에 없던 corner case를 탐색할 수 있다는 점이다. Reward나 downstream improvement로 sample을 고르면 {{BG:reward|보상}} 설계가 data distribution을 지배한다. “그럴듯함”과 “학습 후 utility”는 다르다.

### Mechanism D — Adapter update

{{BG:lora|LoRA}}는 full weight 대신 low-rank delta를 학습한다. Trainable parameter와 optimizer state가 줄고 version 단위가 작아진다. 그러나 forward activation, backward gradient, base weight read는 여전히 필요하다. 또 user별 adapter가 많아지면 batch가 갈라지고 routing metadata가 커진다.

### Mechanism E — Consolidation과 compaction

여러 fast memory를 slow memory로 옮기거나, 여러 episode를 summary/graph로 압축한다. {{BG:consolidation|공고화}}는 중요한 pattern을 장기 state로 promotion하는 과정이다. External version은 merge·supersede·index rebuild이고, parametric version은 distillation·fine-tuning·expert expansion이다.

### Mechanism F — RL-style optimization

Dream candidate나 adapter policy가 future task score를 높이면 reward를 준다. {{BG:policy-gradient|정책 기울기}}를 쓸 수도 있고 best-of-N selection처럼 gradient 없는 최적화를 쓸 수도 있다. 핵심은 reward evaluation이 sleep cost의 일부이며, reward hacking과 distribution shift를 막아야 한다는 점이다.

> **시스템 모델링 관점.** Sleep job의 FLOP만 계산하면 불완전하다. Episode read bytes, vector/graph random write, teacher rollout token, activation checkpoint, optimizer state, adapter snapshot, validation inference, failed version 폐기까지 포함해야 한다.

## 예와 반례

### External synthesis 한 cycle

1. 지난 24시간의 consented episode를 immutable log에서 읽는다.
2. 후보 fact·preference·strategy를 structured record로 추출한다.
3. 기존 memory와 semantic·temporal conflict를 찾는다.
4. admit/merge/supersede/reject를 결정하고 provenance edge를 남긴다.
5. vector index와 temporal graph를 incremental update한다.
6. held-out query로 retrieval precision과 contradiction을 검사한다.
7. 통과한 generation만 active pointer로 전환한다.

이 cycle은 foundation training보다 가볍지만 embedding, LLM extraction, graph traversal, random write가 섞인 heterogeneous workload다.

### Parametric adapter 한 cycle

1. External system-of-record에서 high-reuse·low-conflict record를 선택한다.
2. Positive example, counterexample, deletion test, old-capability replay를 dataset으로 만든다.
3. Teacher output과 ground truth를 섞어 task/distillation loss를 계산한다.
4. Adapter만 backward·optimizer update한다.
5. New fact, old skill, locality, privacy, adversarial set을 평가한다.
6. Threshold를 통과하면 versioned adapter를 publish하고 canary traffic에 붙인다.
7. Regression이면 pointer를 이전 generation으로 되돌린다.

### 반례 — Training loss가 낮으니 consolidation 성공

Train loss는 dataset을 얼마나 잘 맞췄는지만 말한다. Old capability가 사라졌는지, unrelated answer가 변했는지, delete request가 반영되는지, future query latency가 줄었는지는 별도 평가다. Sleep의 output은 model artifact이므로 release gate가 필요하다.

### 반례 — Dream sample 수를 늘리면 계속 좋아짐

Synthetic sample이 같은 generator bias를 반복하면 quantity가 diversity를 보장하지 않는다. Real data anchor를 잃으면 distribution tail이 줄어드는 model collapse가 가능하다. Generated replay는 free data가 아니라 provenance가 약한 derived data다.

## 그림 읽기

{{FIG:STC-F002}}

Mechanism을 memory medium 위에 배치해 보라. Replay·distillation은 adapter와 weight 쪽에, dedup·graph merge는 external memory 쪽에, recurrent update는 learned state 쪽에 놓인다. 동일 sleep scheduler가 서로 다른 kernel과 storage pattern을 부른다. 그래서 “sleep cluster”는 단일 GPU cluster라기보다 GPU training pool, CPU/near-data preprocessing, graph/index tier를 연결한 fabric이다.

{{FIG:STC-F013}}

Failure-mode 그림은 pipeline의 어느 지점에 control을 넣어야 하는지 보여 준다. Ingest에는 consent·poison filtering, transform에는 provenance·quality score, train에는 replay·regularization, publish에는 regression·canary, serve에는 observability, delete에는 lineage-driven rebuild가 필요하다. Control이 없는 sleep은 오류를 durable하게 만드는 자동화다.

## 대안과 비교

| mechanism | 주된 연산 | 주된 state traffic | 장점 | 핵심 risk |
|---|---|---|---|---|
| summary/dedup | generation, similarity | episode read, record write | 해석·삭제 쉬움 | lossy merge |
| graph update | extraction, traversal | random metadata write | 시간·관계 표현 | write amplification |
| replay SFT | forward/backward | activation, optimizer | 강한 retention baseline | storage/privacy |
| distillation | teacher+student forward, backward | logits/activation | compact behavior transfer | teacher error |
| LoRA | base forward+delta backward | base read, small weight write | 작은 artifact | adapter explosion |
| dreaming/RL | rollout, reward, update | generated data, checkpoints | data exploration | recursive bias |

Mechanism 선택은 “얼마나 AI다운가”가 아니라 memory utility와 system cost의 곱으로 결정해야 한다. Stable factual state는 graph/record가, repeated behavior는 adapter가, rare raw evidence는 cold episode store가 더 적합할 수 있다.

## 아직 모르는 것

- Sleep cycle별 end-to-end energy·wall-clock·data-movement breakdown이 공개된 연구가 드물다.
- Teacher rollout, reward model, LoRA SFT 중 어느 항이 실제 utility를 만드는지 ablation이 충분하지 않다.
- Generated replay가 rare tail을 얼마나 보존하는지 장기 sequence에서 측정되지 않았다.
- Multi-rate neural memory update가 prefix cache reuse와 함께 실제 serving throughput을 높이는지 실증이 부족하다.
- Online consolidation과 scheduled sleep 사이에서 optimal split을 학습하는 scheduler가 없다.

> **핵심.** Sleep은 “한 번 더 fine-tuning”이 아니다. Evidence를 data로 만들고, state를 갱신하고, 검증해 배포하는 작은 ML lifecycle 전체다.

## 더 깊이 읽기

- 원 분석: {{STUDY:STC-S07|replay·distill·RL·external compaction mechanism}}
- Training Background: {{BG:distillation|증류}}, {{BG:data-augmentation|데이터 증강}}, {{BG:synthetic-data|합성 데이터}}, {{BG:lora|LoRA}}, {{BG:reward|보상}}, {{BG:policy-gradient|정책 기울기}}, {{BG:consolidation|공고화}}
- Claim route: {{CLAIM:STC-C015}} · {{CLAIM:STC-C019}} · {{CLAIM:STC-C023}} · {{CLAIM:STC-C024}} · {{CLAIM:STC-C035}}
