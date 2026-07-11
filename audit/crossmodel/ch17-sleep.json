[
  {
    "severity": "major",
    "location": "§17.2 핵심 주장 둘째",
    "issue": "생물학적 CF 해법이 rehearsal이 아니라 neuroplasticity라고 배타적으로 서술하지만, 원전은 replay를 수면 중 offline consolidation의 핵심 기제로 명시한다. Parameter expansion은 replay를 대체하는 것이 아니라 replay 기반 Knowledge Seeding과 함께 사용된다.",
    "evidence": "papers/2606.03979.txt:14-18 — \"stable long-term knowledge with replay\"; 82-85 — offline consolidation이 \"repeats the replay of the recently encoded patterns during sleep\"라고 설명한다.",
    "suggested_fix": "“논문은 replay 기반 consolidation에 neuroplasticity를 본뜬 capacity expansion을 결합한다”로 고치고, rehearsal을 해법에서 배제하는 문구를 삭제한다."
  },
  {
    "severity": "major",
    "location": "§17.3.2, §17.4 regime 2, §17.7 첫째",
    "issue": "1k→5k→10k를 update frequency 사다리라고 부른 것은 차원과 방향이 틀렸다. 이 값들은 token 단위 chunk 길이/update period이며, 더 느린 블록일수록 값이 커진다. 원전의 정의대로 frequency는 단위 시간당 갱신 횟수이므로 chunk 길이와 반비례한다.",
    "evidence": "papers/2606.03979.txt:221-239 — frequency를 \"number of updates per unit of time\"으로 정의하고 C^(ℓ)을 frequency로 나눈 chunk size로 정의한다; 344-350 — 1K 블록이 10K 블록보다 열 번 갱신된다는 예시는 1K/10K가 갱신 주기임을 보여 준다.",
    "suggested_fix": "모든 “frequency 사다리 1k→5k→10k”를 “chunk/update-period 사다리 C=1k→5k→10k”로 바꾸고, 대응 frequency는 반대 방향으로 감소한다고 명시한다."
  },
  {
    "severity": "minor",
    "location": "식 (17-1), 표 17-1",
    "issue": "원전 Eq. 2의 합 하한 t=i-C^(ℓ)을 t=i-C^(ℓ)+1로 바꾸면서 이를 단순한 인덱스 관행이며 알고리즘이 동일하다고 설명한다. 동일한 1-based 인덱스와 상한 i를 유지하면 두 합은 한 항 차이이므로 수학적으로 동일하지 않다. 원전의 첫 경계에서 x_0가 생기는 문제를 고친 것이라면 관행 변경이 아니라 원전의 off-by-one 오탈자 교정이라고 밝혀야 한다.",
    "evidence": "papers/2606.03979.txt:249-255 — Eq. 2는 합을 t=i-C^(ℓ)부터 i까지 둔다. 원전은 입력을 x_1,…,x_T로 정의하므로 첫 i=C^(ℓ) 경계에서 t=0이 포함된다.",
    "suggested_fix": "원식을 그대로 옮기거나, 현재 하한을 유지하되 “원전 Eq. 2의 명백한 off-by-one을 1-based token 인덱스에 맞게 교정했다”고 명시하고 “알고리즘 동일”이라는 설명을 삭제한다."
  },
  {
    "severity": "major",
    "location": "§17.3.6 마지막 문단",
    "issue": "consolidation된 저주파 지식을 Dreaming의 weight update가 침식할 수 없다고 단정한다. 원전의 전체 파라미터 동결은 Knowledge Seeding 최적화에만 적용된다. Dreaming은 별도의 LoRA SFT로 모델을 변경하며, 원전도 CF를 불가능하다고 보장하지 않고 두 단계 설계가 더 robust하다고만 주장한다.",
    "evidence": "papers/2606.03979.txt:408-411 — \"In this optimization process\"에서만 기존 student 파라미터를 동결한다; 464-474 — 반복적 Dreaming이 CF를 일으킬 위험을 인정하고 두 단계 설계가 더 robust하다고 보고한다; 500-504 — Dreaming에서 각 instance를 LoRA SFT로 갱신한다.",
    "suggested_fix": "“먼저 consolidation하면 Dreaming에 의한 forgetting 위험을 줄인다는 것이 논문의 가설·실험 결과다”로 약화하고, 구조적으로 침식이 불가능하다는 주장을 삭제한다."
  },
  {
    "severity": "major",
    "location": "§17.4 표 17-2 데이터 행 및 regime 3 설명",
    "issue": "sleep 데이터가 전부 자기 생성이라고 한 것은 Dreaming 설정을 누락한다. Dreaming은 task 관련 context C와 downstream 성능 측정 τ를 외부 조건으로 요구하며, SQuAD passage나 ARC demonstration 및 정답 판정처럼 실제 실험에서도 외부 task 정보가 들어간다.",
    "evidence": "papers/2606.03979.txt:477-482 — Dreaming은 \"sampled task (C, τ)\"와 task-relevant context C를 조건으로 dream을 생성한다; 503-510 — τ에서 fine-tuning 후 개선 여부를 보상으로 사용한다.",
    "suggested_fix": "consolidation corpus는 teacher/student가 자기 생성하지만, Dreaming은 외부 또는 wake에서 보존된 task context와 평가 함수 τ에 조건화된다고 표와 본문을 수정한다."
  },
  {
    "severity": "major",
    "location": "§17.7 둘째",
    "issue": "Sleep에서는 sequence layer가 고정 크기 state가 되고 지속 상태가 파라미터로 대체된다고 일반화한 것은 틀렸다. CMS의 sequence model은 attention일 수도 있으므로 KV cache의 O(L) 상태가 그대로 남을 수 있다. 또한 BABILong에서 GPT 계열 정확도가 128K–256K 이후 하락한 사실은 KV cache 크기가 감당 불가능하다는 인과 증거가 아니다.",
    "evidence": "papers/2606.03979.txt:221-222,274-276 — CMS의 sequence model로 attention, 다른 memory module, RNN을 모두 허용한다; 1493-1497 — BABILong 부록은 대형 모델의 성능 저하를 보고할 뿐 이를 KV-cache 메모리 고갈로 귀인하지 않는다.",
    "suggested_fix": "Sleep은 장기 지식을 parametric memory로 옮기지만 sequence-layer state를 일반적으로 제거하지 않는다고 고친다. O(L) KV cache 제거는 fixed-state sequence module을 택한 Hope 변형에만 한정하고, GPT 성능 저하와 KV 용량 사이의 인과 문장을 삭제한다."
  },
  {
    "severity": "minor",
    "location": "§17.8 라인의 완성형 평가",
    "issue": "Atlas의 retrieval 평균을 attention 53.6, Atlas 43.7로 반올림해 원전 수치와 다르게 적었다.",
    "evidence": "papers/2505.23735.txt:Table 5 부근(약 2181-2294) — Transformer Average 53.55, Atlas Average 43.70.",
    "suggested_fix": "“attention 53.55 vs Atlas 43.70”으로 수정한다."
  }
]