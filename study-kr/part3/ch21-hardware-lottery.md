# ch21. Transformer/NVIDIA scaling 시대의 교훈: hardware lottery

> **이 장의 목표** — 독자가 이 장을 마치면 (1) attention이 이긴 진짜 이유를 표현력 논쟁이 아니라 **GEMM density**로 설명하는 hardware-lottery 독법을 진술할 수 있고, (2) 이 neural-memory 라인이 본질상 recurrent임에도 chunkwise·Newton–Schulz·reset이라는 네 가지 수(手)로 **matmul 안에 설계되어** 있음을, 그리고 그 순응이 pair thesis의 training/prefill 절반이 accelerator 영역인 근본 이유임을 논증할 수 있으며, (3) 이 가족이 아직 기다리는 **FlashAttention-moment**이 무엇인지, 그것이 채택·kernel·pair thesis의 경계에 대해 무엇을 예고하는지 예측으로 옮길 수 있어야 한다.
> **왜 필요한가** — 20장은 아키텍처 class별 decode roofline과 state placement로 병목을 **측정**했다. 이 장은 그 병목이 왜 지금 이 모양인지를 한 단계 위 — 알고리즘 설계가 하드웨어와 협상해 온 역사 — 에서 읽는다. 이 독법(dossier §5 angle A5)은 falsifiable한 정량 기여가 아니라 **framing**이지만, Part III의 다른 장들이 내놓는 숫자에 "왜 이 라인은 새 소자를 요구하지 않는가"라는 배경을 준다. 그 배경이 없으면 24장의 pair 제안 — memory-centric decode 소자와 accelerator prefill kernel을 쌍으로 — 이 왜 균형인지가 흐려진다.

## 21.1 Bridge-in: 20장이 측정한 병목의 역사적 원인

20장은 이 라인의 decode를 arithmetic intensity 0.59 FLOP/byte의 memory-bound RMW로, prefill을 chunk $C$가 x축인 roofline 곡선으로 분해했다(claim 1·7, → 20장). 그 분해는 참이지만 **왜 알고리즘이 그렇게 생겼는가**를 설명하지 않는다. 왜 이 라인은 여섯 편 내내 recurrence를 dense matmul로 다시 빚었는가? 왜 새 하드웨어를 요구하지 않았는가? 이 장의 답은 하나의 개념으로 압축된다: **hardware lottery**.

**hardware lottery**(Hooker 2020, arXiv:2009.06489)는 어떤 연구 아이디어가 본질적으로 더 낫기 때문이 아니라 그 시점에 가용한 하드웨어·소프트웨어에 우연히 잘 맞기 때문에 이긴다는 관찰이다. 아이디어의 승패는 아이디어의 품질이 아니라 그것이 당대 accelerator의 연산 primitive와 얼마나 정렬되는가로 결정된다. 이 렌즈로 지난 10년의 transformer/NVIDIA scaling 시대를 읽으면, 그리고 이 neural-memory 라인이 그 시대에 어떻게 자세를 잡았는지 읽으면, 20장이 측정한 병목이 **선택된 병목**이었음이 드러난다.

## 21.2 attention이 이긴 진짜 이유: GEMM density

softmax attention의 핵심 연산은 두 개의 큰 행렬곱이다 — score $QK^\top$와 가중합 $PV$. 둘 다 sequence 축 전체에 걸쳐 완전히 병렬이고, batch·head·hidden 차원을 곱해 큰 arithmetic intensity를 만들 수 있다. 이것이 GPU/TPU의 tensor core가 가장 잘하는 모양, 즉 높은 **GEMM density**다. 반대로 고전 RNN은 시점 $t$의 상태가 $t-1$에 의존하는 sequential dependency 때문에 같은 하드웨어에서 죽었다 — 병렬화할 GEMM이 없고, 매 스텝이 작은 GEMV로 쪼개져 tensor core를 놀린다.

여기서 핵심은 이 승부가 **표현력 논쟁과 무관**했다는 점이다. RNN이 원리적으로 무엇을 표현할 수 있는지, attention이 무엇을 못 하는지는 결과를 바꾸지 않았다. attention은 자신을 큰 matmul로 표현할 수 있었기에 scaling 시대의 로또를 이겼고, RNN은 그러지 못했기에 밀렸다. transformer가 지배종이 된 것, 그리고 NVIDIA가 그 지배의 하드웨어 지대(地代)를 걷은 것은 같은 사건의 두 얼굴이다.

> **[해설]** 독자가 매일 다루는 KV cache는 이 승리의 부산물이다. attention은 과거 token을 **압축하지 않고** 통째로 남겨 두는 대가로 GEMM density를 얻는다 — softmax attention은 capacity가 무한한($\phi^*$) associative memory, 즉 압축하지 않는 극한이다(→ 14장). KV cache가 문맥에 비례해 자라는 것은 결함이 아니라 이 거래의 명세서다: 메모리를 무한정 쓰는 대신 연산을 완벽한 matmul로 유지한다. 이 라인이 도전하는 것이 바로 그 거래 — 상태를 고정 크기 weights로 **압축**하되, 압축의 대가로 무엇을 하드웨어에 지불해야 하는가 — 다.

### 21.2.1 GEMM density는 왜 하필 승리 조건이었나 — systolic array에서 tensor core까지

GEMM density가 로또의 당첨 번호가 된 것 자체가 하드웨어 역사의 산물이다. Dennard scaling이 끝나 clock 주파수를 더 올리지 못하게 된 2010년대, accelerator가 지수적으로 키운 것은 clock이 아니라 **병렬 throughput** — 한 사이클에 소화하는 dense·regular한 곱셈-누산의 수 — 였다. 이 throughput을 값싸게 뽑는 표준 구조가 systolic array다: Kung–Leiserson(1978)이 제안한, 데이터가 격자 위를 규칙적으로 흐르며 곱-누산되는 dataflow. TPU의 MXU가 정확히 이 systolic array이고, GPU의 tensor core도 같은 원리의 작은 dense-matmul 엔진이다. 이 구조들의 공통 성질은 하나다: **규칙적인 dense GEMM에 최적이고, irregular·sparse·sequential access에는 페널티를 매긴다.**

그래서 승부는 표현력 이전에 dataflow에서 갈렸다. attention의 $QK^\top$·$PV$는 systolic array가 흡수하도록 태어난 것처럼 규칙적이라, 하드웨어가 넓어질수록 그대로 공짜로 빨라졌다. 반대로 RNN/LSTM은 — 2014–2017 sequence modeling의 실질 SOTA였음에도 — 시점 $t$가 $t-1$을 기다리는 sequential dependency 때문에 넓어지는 tensor throughput을 굶겼다. LSTM이 밀린 것은 무엇을 표현하지 **못해서**가 아니라, 지수적으로 좋아지는 하드웨어를 **먹지 못해서**였다. 이것이 hardware lottery의 교과서적 패배 사례다. 같은 이유로 초기 MoE·sparse attention 류의 conditional-compute 아이디어도 dense GEMM에 맞지 않아 반복적으로 채택이 지연됐다(Hooker 자신의 예).

타이밍의 우연도 논제가 예측하는 대로다. attention을 정의한 transformer(2017)는 tensor core를 처음 실은 Volta(2017)와 같은 해에 도착했고, 그 이전 GPU 딥러닝의 문(AlexNet, Krizhevsky et al. 2012 → cuDNN)은 이미 dense-matmul 특화 쪽으로 열려 있었다. 아이디어와 소자가 우연히 같은 창(窓)에서 만난 것이다. 이후 둘은 **공진화(lock-in)**한다: 지배종 transformer가 하드웨어 로드맵을 끌고(더 큰 MXU, 더 낮은 정밀도의 matmul), 그 하드웨어가 transformer를 더 싸게 만들어 지배를 굳힌다. NVIDIA가 이 시대의 지대를 걷은 구조가 이것이다.

> **[평가]** 이 배경이 이 장의 전제다. GEMM density는 알고리즘의 내재적 미덕이 아니라 **특정 시대의 소자가 값싸게 실행할 수 있는 모양**이고, 그 모양을 통과하지 못한 아이디어는 옳고 그름과 무관하게 지연됐다. neural-memory 라인은 본질상 RNN을 로또에서 떨어뜨린 바로 그 sequential dependency를 품고 있다. 그러므로 이 라인이 살아남으려면 같은 관문 — dense GEMM으로 자신을 표현하는 능력 — 을 통과해야 한다. 21.3은 이 라인이 그 관문을 **어떻게** 통과했는지를, 21.4는 그 통과가 아직 **부분적**임을 읽는다.

## 21.3 이 라인은 matmul 안에 설계되어 있다

neural-memory 라인은 본질적으로 recurrent다. inner loop가 token마다 fast weights $W_t$를 $W_{t-1}$에서 갱신하는 sequential 과정이고(→ 8장·9장), 이는 정확히 RNN을 로또에서 탈락시킨 그 dependency다. 그런데 이 라인은 탈락하지 않았다. 저자들이 hardware lottery의 세(稅)를 **선불**했기 때문이다 — 알고리즘을 기존 accelerator에 맞도록 네 번에 걸쳐 다시 빚었다.

1. **chunkwise-parallel training** (→ 9장). chunk 안의 모든 gradient를 chunk 시작 상태 $W_{\xi(t,C)}$에서 평가하는 stale-snapshot 근사(식 (M4))는 sequential recurrence를 chunk 단위 GEMM으로 바꾼다. 정확성을 내주고 matmul을 산다 — 이 라인 여섯 편 전부의 병렬화 기반이다.
2. **Newton–Schulz orthogonalization** ([Atlas], → 14장). Muon inner optimizer의 $\mathrm{NS}_\kappa$($\kappa=5$)는 momentum buffer를 semi-orthogonal하게 만드는 반복인데, 그 반복 자체가 batched 행렬곱의 연쇄다. [Atlas]가 자기 방법을 "tensorize and maximize matmuls"의 원칙으로 서술하는 것은 우연이 아니라 로또를 의식한 설계 언어다.
3. **periodic reset** ([TNT], → 15장). local memory를 학습된 $W_{\mathrm{init}}$으로 주기적으로 되돌리면 비선형 recurrence의 사슬이 끊겨, deep-memory recurrence를 병렬화하는 유일하게 알려진 길 — context parallelism — 이 열린다. reset은 품질 장치이기 이전에 **병렬화 장치**다.
4. **featurized keys** ([Atlas], → 14장). polynomial feature map $\phi_p$로 key를 들어 올려 capacity $O(d_k^p)$를 사는 것은, capacity마저 더 많은 matmul로 치환하는 수다.

이 네 수의 공통 문법은 하나다: **arithmetic intensity를 제조한다.** claim 7이 이 문법을 정량화한다 — chunk $C=1$(per-token rank-1 RMW = decode 영역)은 AI≈1 FLOP/byte로 memory-bound 평원에 앉아 있지만, $C$를 키우면 roofline을 타고 올라 crossover를 넘어 compute-bound로 옮겨 간다. 같은 알고리즘이 knob 하나로 두 영역을 오간다.

<!-- FIG: exp-c -->

(그림 c — chunk $C$에 따른 AI(C) 곡선이 memory-bound 평원에서 compute-bound로 오르는 모양; 주 담당은 9장·15장, → 20장에서 decode-side와 함께 배치.)

> **[해설]** claim 7의 정직한 사용법을 못박아 둔다. **load-bearing한 것은 곡선의 모양과 memory↔compute crossover의 존재**다: 저자들이 $C$를 키워 GEMM density를 제조할 수 있다는 사실. crossover의 **위치**는 ridge 의존적이어서, host CPU에서 측정된 $C^*\approx32$는 곡선의 모양을 확인할 뿐 H100로 이전되지 않는다(host ridge ≈34 FLOP/byte는 H100 twin ridge 295의 약 1/9; H100 closed-form $C^*$는 $d$에 따라 306–430, → 20장·§18.6 정직성 계약). 이 장이 딛는 것은 오직 "제조가 가능하다"는 구조뿐이다.

**GEMM shape로 읽으면** 이 제조가 눈에 보인다. chunkwise gradient $\nabla_W\ell$은 chunk 내 $C$개 token의 오차를 key에 실은 $dW=(\text{오차})\,K^\top$ 꼴 — contraction 차원이 정확히 $C$인 rank-$C$ GEMM이다(→ Rosetta). 그래서 $C$는 arithmetic intensity를 직접 쥐는 손잡이가 된다. host roofline 위 측정이 이 손잡이의 힘을 보여 준다: 같은 커널이 $d=2048$에서 $C{=}1$일 때 4.56 GFLOP/s로 memory-bound 평원(host roofline의 ~12%)에 앉아 있다가, $C$를 512까지 키우면 689.5 GFLOP/s로 오른다 — 한 손잡이로 약 150× throughput을 제조한 것이다(claim 7, → 20장). 정직성: 이 GFLOP/s 절대치는 host의 것이라 H100로 이전하지 않는다(host ridge ≈34 FLOP/byte는 H100 twin ridge 295의 약 1/9). load-bearing한 것은 곡선의 모양, memory↔compute crossover의 존재, 그리고 이 한 자릿수 knob이 만드는 ~150× swing이라는 비율뿐이다.

이 제조가 실제로 로또를 이길 만큼 강하다는 증거는 [TNT]에 있다: plain JAX로 작성된 TNT가 32K 문맥에서 step당 FlashAttention을 이겼다(→ 15장). recurrent memory가 — 커스텀 CUDA 없이 — 지배종의 최적화된 커널을 특정 조건에서 앞선 것이다.

> **[평가]** 이 라인 전체는 hardware lottery에 대한 하나의 긴 **순응**으로 읽어야 한다. 여섯 편 어디에도 "이 알고리즘을 위해 새 하드웨어가 필요하다"는 요구가 없다. 정반대로, 매 편이 알고리즘을 dense matmul로 다시 빚어 기존 accelerator에 밀어 넣는다. 이것이 pair thesis의 training/prefill 절반이 왜 **accelerator 영역**인지의 근본 이유다(→ 18장, 24장): 그 절반에는 chunk라는 축이 있어 GEMM density를 제조할 수 있고, 제조할 수 있는 곳에서는 memory-centric 소자를 요구하는 것이 아니라 tensor core를 더 잘 먹이는 것이 답이다. 이 라인의 중심 엔지니어링 패턴 자체가 "training에 새 소자가 필요하다"는 주장을 금지한다(dossier §5의 forced 판정 3).

## 21.4 그런데 FlashAttention-moment은 아직 오지 않았다

로또의 세를 선불했다고 해서 이 라인이 곧바로 실용화되는 것은 아니다. attention의 실용화에는 알고리즘 승리 이후 한 번의 **kernel 승리**가 더 필요했다. FlashAttention(Dao et al. 2022, arXiv:2205.14135)은 attention의 수학을 한 글자도 바꾸지 않고 — bit-exact tiling으로 — IO를 재조직해 attention을 메모리 병목에서 풀어냈다. 이후 FlashDecoding이 같은 아이디어를 decode까지 확장했다. attention이 오늘 어디서나 돌아가는 것은 이 kernel-level 돌파 덕이다.

이 neural-memory 가족에는 그런 순간이 **아직 없다.** 그리고 그 부재는 추측이 아니라 생태계에서 관측된다. flash-linear-attention(FLA, 5,325★)은 이 라인의 **matmul-clean한** 계보 — `delta_net`, `gated_deltanet`, `gla`, `rwkv7`, `mamba2`, `mesa_net` 등 — 을 전부 production-grade Triton 커널로 layer·model 수준까지 커버한다. Gated DeltaNet(이하 GDN)은 NVIDIA의 공식 구현(NVlabs/GatedDeltaNet, 619★, ICLR 2025)까지 갖췄다. 그런데 이 라인의 정점인 **deep-memory + momentum** Titans는 FLA에서 `fla/ops/titans`의 **naive PyTorch 참조 구현**에 머물러 있다 — Triton 커널도, layer/model 통합도 없다. 같은 RFC(#107)에서 TTT와 Titans 커널이 함께 발의됐으나 TTT만 Triton화되고 Titans는 #214에서 정체했다.

이 정체의 원인이 이 장의 핵심 증거다: **chunk-level momentum이 chunkwise closed form을 깨뜨린다.** momentum buffer $S_t=\beta_t S_{t-1}-\eta_t\nabla_W\ell$(식 (M2))의 재귀 항은 chunk 경계를 넘어 이어지므로, chunk 내부를 하나의 깨끗한 matmul로 접는 dual form이 성립하지 않는다. 바로 이 어려움이 [TNT]의 존재 이유였고(→ 9장·15장), 지금은 오픈소스 커널 생태계에 남긴 **실물 흔적** — Titans만 naive에 멈춘 자리 — 으로 확인된다.

> **[해설]** 즉 이 라인은 로또의 세를 **부분적으로만** 냈다. matmul-clean한 부분(linear attention, delta rule, gating)은 이미 커널이 있고 이미 채택됐다 — Mamba-2와 GDN은 production에 들어갔다. matmul-clean하지 **않은** 부분(deep MLP memory, momentum, self-modification)은 fused 커널이 없어 아직 연구 코드에 머문다. 이 라인이 논문에서 그린 완성형(→ 18장)은 후자에 크게 기댄다. 그러므로 완성형과 배포 사이에는 아직 하나의 kernel 승리가 통째로 비어 있다.

### 21.4.1 왜 momentum·deep memory는 커널화가 어려운가 (기계적 해부)

이 정체가 우연한 엔지니어링 지연이 아니라 알고리즘의 구조적 성질임을 못박아 둔다. chunkwise 병렬화의 문법을 다시 부르자(→ 9장): sequential recurrence를 (i) chunk 내부를 하나의 큰 matmul로 접는 병렬 파트와 (ii) chunk 요약들만 잇는 inter-chunk sequential scan으로 분해한다. 이 분해가 성립하려면 per-token update가 상태의 affine map이어야 하고, 그 map들의 chunk-내 합성이 **associatively 결합 가능한 compact object(행렬 하나)**로 접혀야 한다. 어떤 알고리즘이 "matmul-clean"한가는 정확히 이 접힘이 되는가로 갈린다.

**되는 쪽 — GDN.** GDN의 update $W_t=\alpha_tW_{t-1}(I-\eta_tk_tk_t^\top)+\eta_tv_tk_t^\top$는 $W$에 affine이고 memory가 linear(행렬)다. chunk 내 $\prod_i(I-\eta_ik_ik_i^\top)$ 곱은 WY/UT 표현으로 하나의 masked matmul(삼각 역행렬 한 번 — 그 자체가 GEMM)로 접힌다. 그래서 `delta_net`·`gated_deltanet`·`gla`·`rwkv7`·`mamba2`가 전부 FLA에 production-grade Triton으로 존재하고, GDN은 NVIDIA 공식 커널(NVlabs/GatedDeltaNet, 619★)까지 갖췄다.

**안 되는 쪽 — Titans(deep memory + momentum).** 두 겹의 장애가 겹친다.

- **장애 1 — deep memory의 gradient는 rank-1이 아니다.** memory가 표준 deep memory(2-layer residual MLP, → §1.3)이면 $\nabla_W\ell$은 outer product $v_tk_t^\top$가 아니라 MLP forward·backward를 통과한 matmul 사슬(중간 nonlinearity 포함)이다. linear-in-$W$ closed form이 존재하지 않는다. stale-snapshot 근사(식 (M4))가 chunk 내 모든 gradient를 anchor $W_{\xi(t,C)}$에서 평가해 chunk를 하나의 batched forward/backward로 만들어 **chunk-내** 병렬성은 복원하지만, 이는 chunk **간** 결합까지 풀어 주지는 않는다.
- **장애 2 — momentum이 scan combine을 companion 행렬로 만든다.** momentum buffer $S_t=\beta_tS_{t-1}-\eta_t g_t^{\mathrm{in}}$(식 (M2))의 재귀는 상태를 pair $(W_t,S_t)$로 확장한다. joint map은 여전히 affine이지만, chunk-내 합성이 momentum decay 사슬 $\beta_t\beta_{t-1}\cdots$을 실어야 하고 각 위치의 $\eta_t$·$\beta_t$ gate가 얽혀, scan의 combine 연산이 **위치별 block(companion) 행렬 곱**이 된다. delta rule의 $(I-\eta_tk_tk_t^\top)$처럼 하나의 clean GEMM으로 접히지 않는다.

이 두 번째 장애가 생태계에 남긴 실물 흔적이 `fla/ops/titans`다. 이 ops는 momentum·decay 곱을 log-space에서 combine하려고 $L\times L$(sequence×sequence) 하삼각 가중 행렬을 그대로 materialize한다(`naive.py`+`log_impl.py`). $O(L^2)$의 메모리·연산은 $C$를 키울수록 chunk 병렬화가 사려던 이득을 도로 반납하므로, Triton 커널이 아니라 **참조 구현**에 머문다. 대비가 결정적이다: RFC #107이 TTT와 Titans 커널을 **함께** 발의했으나, 구조적으로 접히는 TTT만 진짜 Triton 커널(`chunk_ttt_linear`, `fused_chunk_ttt_linear`; group-norm 융합·varlen 지원)이 되고 momentum을 지닌 Titans는 #214에서 naive에 정체했다. 같은 팀·같은 인프라·같은 시점에서 갈린 것은 오직 알고리즘의 matmul-cleanness다. 사실상의 reference 구현인 lucidrains/titans-pytorch(1,966★)조차 dual-form matmul이 아니라 AssocScan(accelerated associative scan) 기반 병렬화로 우회한다 — scan은 돌지만 tensor core를 FlashAttention만큼 먹이지는 못한다.

같은 메커니즘이 momentum 하나에 국한되지 않는다. Miras의 새 attentional bias들도 이 관문에 걸린다: Moneta의 $\ell_p$ norm이나 Yaad의 Huber loss는 inner objective를 nonlinear하게 만들어 순간의 chunkwise closed form을 없애고, $\ell_2$ 계열이 재활용하던 DeltaNet/GLA 커널 대신 naive 순차 루프(또는 논문의 근사)로 후퇴하게 한다. 그래서 FLA에는 Miras의 세 인스턴스(Moneta/Yaad/Memora)가 없다 — 설계 공간의 커널 인프라(`comba`, `mesa_net`, `mom`, `kda`)는 있는데 nonlinear-loss 지점만 비어 있는 것이다. 즉 이 라인의 matmul-cleanness는 momentum·deep-memory·nonlinear-loss라는 **세 축**에서 동시에 시험받고, fused 커널의 공백은 정확히 이 세 축이 겹치는 곳 — 논문들이 그린 완성형(Titans/Atlas/Hope) — 에서 가장 크다.

> **[해설]** 문이 원리적으로 닫힌 것은 아니다. MesaNet(Behrouz 라인 인접, arXiv:2506.05233)은 locally-optimal test-time regression을 chunkwise conjugate-gradient solver로 Triton화했다(`chunk_cg_solver`) — Atlas의 2차·window 방향에 가장 가까운 커널 증거다. 즉 "2차 정보를 쓰는 recurrent memory는 커널화 불가"가 아니라, **deep-MLP memory + momentum이라는 특정 조합에 대한 fused 커널이 아직 쓰이지 않았을 뿐**이다. 이 구분이 다음 절 예측 2의 형태를 정한다.

## 21.5 이 라인의 FlashAttention-moment이 예고하는 것

hardware-lottery 독법은 예측을 낳는다. 네 가지로 정리한다.

**예측 1 — 채택은 kernel 가용성이 정한다, 수학적 우수성이 아니라.** 커널이 있는 부분만 production으로 간다. GDN·Mamba-2는 이미 갔고, deep-memory Titans/HOPE는 fused 커널이 나오기 전까지 못 간다. 이것은 hardware lottery의 **재현**이다: 라인 내부에서조차 승자는 더 나은 memory가 아니라 커널이 있는 memory다. [Atlas] 자신의 ablation이 760M에서 Muon 제거가 오히려 perplexity를 개선했다고 보고한 것(→ 14장)은 이 예측과 나란히 읽힌다 — 가장 matmul-무거운 축의 품질 가치조차 미결인데, 그 축은 커널 부담까지 가장 크다.

> **[해설]** 이 재현은 라인의 hybrid 설계 선택에서도 관측된다. Titans의 세 합성 방식(MAC/MAG/MAL) 중 MAL(Memory as Layer)이 throughput 비교에서 유리하게 나오는 국면은, MAL이 이미 성숙한 FlashAttention 커널 위에 memory를 얹기 때문이지 memory 자체가 더 우수해서가 아니다(dossier §5 A5). 커널 성숙도가 아키텍처 선택의 저울을 기울인 것 — 라인 **내부**에서 로또가 한 번 더 돈 사례다. 채택을 읽을 때 "어느 memory가 더 나은가"와 "어느 memory에 커널이 있는가"를 분리하지 않으면 이 교란에 걸린다.

**예측 2 — moment의 형태는 fused deep-memory chunk kernel이다.** 21.4.1의 해부는 그 커널이 구체적으로 무엇을 융합해야 하는지까지 못박는다. 네 가지다.

1. **anchor에서의 deep-MLP forward+backward를 batched GEMM으로.** stale-snapshot(M4)이 chunk 내 모든 gradient를 $W_{\xi(t,C)}$에서 평가하므로, chunk 전체의 forward·backward는 중간 activation을 SRAM에 유지하는 IO-aware tiling으로 처리할 수 있다 — FlashAttention이 softmax에 쓴 그 수를 MLP의 backward로 옮긴 것.
2. **momentum associative scan을 sequence×sequence materialization 없이 융합.** 장애 2의 companion-block combine을 tile 단위로 online 누적한다 — FlashAttention의 running-softmax에 대응하는 running-momentum. `fla/ops/titans`가 materialize하는 $L\times L$ 행렬을 tile-local 재귀로 대체하는 것이 커널화의 핵심 난관이다.
3. **periodic reset 경계를 커널 안에서 처리.** local memory의 $W_{\mathrm{init}}$ reset(→ 15장)이 커널 tiling 경계와 정렬돼야 context-parallel shard가 합성된다.
4. **결정적 비대칭 — bit-exact가 아니라 semantic이다.** FlashAttention은 attention의 수학을 한 글자도 바꾸지 않는 bit-exact tiling이었다(→ Rosetta). 그러나 이 커널은 이미 **semantic 근사** 위에 올라탄다 — chunk $C$가 계산되는 함수 자체를 바꾸는 semantic hyperparameter이기 때문이다(M4, → 9장). 그래서 "무엇을 커널화하느냐"가 "알고리즘이 무엇을 계산하느냐"와 분리되지 않는다. 이것이 이 가족의 FlashAttention-moment이 원본보다 구조적으로 어려운 이유다: 커널 설계와 semantic 선택이 한 문제로 얽혀 있다.

그 커널이 나오기 전까지 quality-optimal한 작은 chunk는 deep memory에서 FLOPs utilization 5–10% 미만으로 돈다([Atlas] 보고, → 14장·15장) — wall-clock 경쟁이 불가능한 영역이다. 누가 그 커널을 쓰든(Google 내부, FLA 커뮤니티, 혹은 NVIDIA) 그 순간이 이 라인을 잠금 해제한다.

**예측 3 — 그 moment은 pair thesis의 절반만 푼다(비대칭).** 이것이 이 장이 이 책의 척추에 기여하는 지점이다. FlashAttention은 prefill과 decode를 **둘 다** 도왔다(FlashDecoding). 그러나 이 가족의 kernel moment은 **prefill/training 절반만** 도울 수 있다. decode는 $C=1$에 갇혀 있어 GEMM density를 제조할 chunk 축이 없다 — arithmetic intensity 0.59 FLOP/byte, 결정적으로 memory-bound이고, state 트래픽이 GEMV 연산을 약 394× 압도한다(claim 1, → 20장). 아무리 영리한 GEMM 커널도 움직여야 할 바이트 수 자체($m\,d^2\,L_{\mathrm{layer}}$의 RMW)를 줄이지 못한다.

이 불가능은 기계적으로 airtight하다. FlashAttention이 준 것은 **연산 재조직**(tiling·IO 재배치)이었지 바이트 수 감축이 아니었다 — 그럼에도 attention이 도움받은 것은 그 IO 재조직이 prefill과 decode **양쪽**에서 실효를 냈기 때문이다(FlashDecoding). 이 가족의 decode는 다르다: 매 token이 전체 state를 read-modify-write하고 그 트래픽 $m\,d^2\,L_{\mathrm{layer}}$은 문맥 길이 $S$에 **무관**하게 고정이며(claim 1·2), chunk 축이 $C=1$로 붕괴해 있어 21.3의 ~150× 제조 손잡이가 **아예 없다**. 커널이 최적화할 대상(GEMM 재배치)이 decode에는 존재하지 않는 것이다. 병목은 순수하게 memory 계층의 RMW 대역폭이고, 이는 커널 문제가 아니라 소자·배치 문제다(→ 20장·24장). KV cache와의 crossover $S^*$(anchor에서 read-half 기준 ≈65k token, claim 2)가 정확히 이 경계를 표시한다: $S^*$ 아래에서는 append-once KV가 싸고 위에서는 고정 RMW인 이 라인이 싸지지만 — 어느 쪽이든 decode가 움직이는 바이트 수는 커널이 아니라 memory 시스템이 정한다.

> **[평가]** 따라서 hardware lottery는 이 라인에서 **비대칭으로만** 작동한다. accelerator가 이길 수 있는 곳과 memory-centric 접근이 필요한 곳의 경계는 정확히 "chunk 축이 있는가"의 경계이고, 그 경계는 pair thesis의 split(→ 18장)과 **정확히 겹친다.** training/prefill에는 축이 있어 kernel 승리가 가능하고 — 그래서 accelerator 영역 — decode에는 축이 없어 kernel이 못 풀고 memory 계층·RMW 대역폭이 병목으로 남는다 — 그래서 memory-centric 영역(→ 20장·24장). FlashAttention이 attention에게 준 양면 승리를, 이 가족은 구조적으로 **한 면**밖에 받을 수 없다. 이 비대칭이 24장이 제안을 **쌍**으로 내는 이유의 하드웨어-역사적 근거다.

**예측 4 — 누가 그 커널을 쓸 동기가 있는가는 player마다 다르다.** Google Research는 자사 long-context 제품과 TPU pod 때문에, NVIDIA는 GDN 계보의 연장선 때문에, FLA 커뮤니티는 오픈소스 커버리지 완성 때문에 각각 다른 강도의 동기를 갖는다. 이 player별 유인 구조가 다음 장의 주제다.

## 21.6 이 독법의 한계 (정직성 각주)

> **[평가]** hardware-lottery 독법은 dossier §5가 A5로 분류한 대로 **framing으로 강하고 standalone 기여로는 약하다.** 이 장은 예측을 내지만 그 예측들은 아직 검증되지 않았다 — fused deep-memory 커널이 나올지, 나오면 채택이 실제로 따라올지는 미래의 사실이다. 또한 이 독법은 "수학이 중요하지 않다"는 강한 주장이 **아니다.** in-context retrieval 격차(attention 53.55 vs 43.70 [Atlas Table 5], FDA 67.3 vs 41.9 [NL], → 14장·16장)는 어떤 커널도 닫지 못한다 — 커널은 wall-clock을 풀지 품질을 풀지 않는다. capacity 이론이 그 격차의 이유($\phi^*$의 무한 capacity)까지 말해 주므로, 완성형은 여전히 attention과의 hybrid일 수 있다. hardware lottery는 **어느 아이디어가 배포되는가**를 설명하는 렌즈이지, **어느 아이디어가 옳은가**를 판정하는 저울이 아니다. 이 구분을 지키는 한에서 이 장의 예측은 유효하다.

## 요약

- **hardware lottery**(Hooker 2020): 아이디어는 품질이 아니라 당대 하드웨어와의 정렬로 이긴다. attention이 scaling 시대를 이긴 것은 표현력이 아니라 두 큰 matmul($QK^\top$, $PV$)이 만드는 **GEMM density** 덕이고, KV cache가 문맥에 비례해 자라는 것은 압축을 포기하고 완벽한 matmul을 유지한 그 거래의 명세서다.
- GEMM density가 승리 조건이 된 것 자체가 역사적 우연이다: Dennard scaling 이후 지수적으로 는 것은 clock이 아니라 systolic array(Kung–Leiserson 1978 → TPU MXU / tensor core)가 값싸게 뽑는 dense-matmul throughput이었고, 이 구조는 sequential·irregular access에 페널티를 매긴다. RNN/LSTM(2014–2017 SOTA)이 밀린 것은 표현력이 아니라 넓어지는 throughput을 굶겨서였다(교과서적 hardware-lottery 패배). attention(2017)과 tensor core(Volta 2017)는 같은 창에서 만나 공진화(lock-in)했다.
- 이 neural-memory 라인은 본질상 recurrent인데도 로또의 세를 **선불**했다: chunkwise stale-snapshot(→ 9장), Newton–Schulz/Muon("tensorize and maximize matmuls", → 14장), periodic reset(context parallelism, → 15장), featurized keys — 네 수 모두 arithmetic intensity를 제조한다. claim 7의 load-bearing 결론은 "chunk $C$로 GEMM density를 제조할 수 있다(곡선 모양과 crossover의 존재)"이며, $C^*$의 절대 위치는 ridge 의존적이라 이전하지 않는다. plain-JAX TNT가 32K에서 step당 FlashAttention을 이긴 것(→ 15장)이 그 제조력의 증거다.
- 그러나 알고리즘 승리 뒤에 필요한 **kernel 승리**가 아직 없다. FLA(5,325★)는 matmul-clean한 계보(DeltaNet/GDN/GLA/RWKV-7/Mamba-2)를 Triton으로 전부 커버하지만, deep-memory+momentum Titans는 `fla/ops/titans`의 naive PyTorch에 머문다(TTT는 Triton화, Titans는 #214에서 정체). 원인은 기계적이다: GDN은 chunk 내 $\prod(I-\eta_tk_tk_t^\top)$가 WY/UT 표현으로 하나의 matmul로 접히지만, Titans는 (i) deep memory의 gradient가 rank-1이 아니고 (ii) momentum이 scan combine을 companion 행렬로 만들어 `fla/ops/titans`가 $L\times L$ 하삼각 행렬을 materialize하게 강제한다 — [TNT]의 존재 이유가 생태계에 남긴 실물 흔적. 단 MesaNet의 `chunk_cg_solver`(arXiv:2506.05233)가 2차 방향의 커널화가 원리적으로 가능함을 보인다.
- 예측: (1) 채택은 kernel 가용성이 정한다(GDN·Mamba-2는 갔고, deep-memory는 대기) — 라인 내부의 hardware lottery 재현; (2) 이 가족의 FlashAttention-moment = momentum·deep MLP·reset을 융합한 fused chunk kernel(그 전까지 deep memory는 <5–10% FLOPs util, [Atlas]); (3) **비대칭** — 그 moment은 chunk 축이 있는 prefill/training 절반만 풀고, $C=1$·memory-bound·394× state-지배인 decode(claim 1)는 못 푼다; 이 경계가 pair thesis split과 정확히 겹친다(→ 18장·24장); (4) player별 유인은 다르다(→ 22장).
- 이 독법은 framing으로 강하고 falsifiable한 기여로는 약하다. 커널은 wall-clock을 풀지 retrieval 격차(53.55 vs 43.70)를 풀지 못한다 — hardware lottery는 무엇이 배포되는가의 렌즈이지 무엇이 옳은가의 저울이 아니다.

## 자가 점검 체크리스트

- [ ] attention의 승리를 표현력이 아니라 GEMM density로 설명하고, KV cache를 그 거래의 부산물로 재서술할 수 있다.
- [ ] GEMM density가 왜 승리 조건이 됐는지를 systolic array/tensor core 공진화와 RNN의 hardware-lottery 패배로 역사화할 수 있다.
- [ ] GDN이 커널화되는 이유(WY/UT 접힘)와 Titans가 안 되는 이유(rank-1 아님 + companion scan → $L\times L$ materialize)를 기계적으로 대비할 수 있다.
- [ ] 이 라인이 matmul 안에 설계된 네 가지 수(chunkwise·NS·reset·featurized keys)를 각각 어느 장이 소유하는지와 함께 열거할 수 있다.
- [ ] claim 7에서 무엇이 load-bearing이고(곡선 모양·crossover 존재) 무엇이 이전 불가인지($C^*$ 절대 위치)를 구분할 수 있다.
- [ ] FlashAttention-moment이 무엇이었는지, 이 가족에 왜 아직 없는지를 FLA/Titans-naive 증거로 설명할 수 있다.
- [ ] 이 가족의 kernel moment이 pair thesis의 어느 절반만 푸는지, 그 비대칭이 왜 pair thesis split과 겹치는지 논증할 수 있다.
- [ ] hardware-lottery 독법이 판정하지 **못하는** 것(retrieval 격차, 아이디어의 옳음)을 짚을 수 있다.
- [ ] 이 장의 논지를 inference 어휘로 옮길 수 있다: "attention은 GEMM 로또를 이겼고 FlashAttention으로 kernel 로또까지 이겼다; 이 라인은 GEMM 로또의 세를 선불했으나 kernel 로또는 절반만 이길 수 있다."

## 다음 장으로

이 장은 kernel 승리가 아직 비어 있다고 진단하고, 그 승리를 쓸 동기가 player마다 다르다고 예고했다. 22장은 그 유인 구조를 편다 — 왜 Google Research가 이 라인의 자연스러운 저자인지(TPU pod·JAX·long-context 제품), NVIDIA가 GDN 계보로 어디에 서 있는지, flash-linear-attention 생태계가 오픈소스 커널의 무게 중심을 어떻게 쥐고 있는지, 그리고 각 player가 이 라인의 FlashAttention-moment 앞에서 합리적으로 두는 다음 수는 무엇인지.
