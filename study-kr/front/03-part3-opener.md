# Part III — 원저 기여

Part II는 여섯 편이 하나의 배포 형태로 수렴함을 보였다 — 세션 상태가 mutable weights인 continually-learning LLM. 그러나 여섯 편 어디에도 그 배포가 실제로 무엇을 요구하는지는 측정되어 있지 않다. decode wall-clock 수치가 라인 전체에 부재하다. Part III는 그 빈 자리를 채우는 원저 기여다: 완성형을 systems 엔지니어의 도구 — roofline, GEMM shape, memory 계층, batching — 로 계량한다.

Part III의 척추는 하나의 명제, **D4 workload-split pair thesis** — 이 라인의 배포가 질적으로 다른 두 부하(memory-centric한 decode/serving-state 관리, accelerator 영역인 training/prefill)로 갈라진다는 것 — 이며, 8개 실측 실험이 그 두 절반을 각각 정량화한다. 이 명제의 정식 진술, 완성형의 재구성과 두 유보, 8개 claim의 실험·figure·장 매핑, 그리고 실측의 정직성 계약(exploration-grade — 비율·crossover·순서는 본문으로, 절대치는 A100 runbook으로 이월)은 **18장이 완결된 형태로 세운다.** 이 도입은 그 위에 얹힌 나머지 장들이 어떻게 이어지는지만 가리킨다.

**18장**이 프로그램 전체(수렴·pair thesis·8 claim 지도)를 세운 뒤, 나머지 일곱 장이 두 절반을 전개한다. **19장** — scaling 분석: state-bytes와 capacity를 params·tokens와 나란한 일급 scaling 축으로 올리고 여섯 편의 published 점들로 후보 scaling law를 맞춘다. **20장** — 하드웨어 병목: 각 아키텍처 클래스의 decode roofline과, backward-pass-at-decode라는 새 serving primitive. **21장** — hardware lottery: attention이 GEMM density로 이긴 역사를 읽고, 이 라인이 스스로를 matmul로 빚은 것이 채택·kernel에 무엇을 예고하는지. **22장** — player-strategy: Google(TPU·JAX·long-context 제품), NVIDIA(Gated DeltaNet 계보), open-source kernel 생태계가 각자 합리적으로 두는 다음 수. **23장** — 대규모 학습·서빙 projection: TNT 경제학을 7–70B로 외삽하고, per-session weight state를 새로운 cache class로 설계한다. **24장** — 제안: 알고리즘 레벨(§5의 소규모 실험)과 하드웨어 레벨(frequency-tiered memory 배치와 fused chunk kernel·grouped-GEMM decode의 pair). **25장** — 결론과 연구 어젠다.

여덟 장을 관통하는 규율은 정직성 계약이다. 본문으로 승격되는 것은 비율·순서·bound 분류이지 silicon 정확 절대치가 아니며, memory-centric 논증은 pair thesis의 두 지점 밖으로 나가지 않는다.
