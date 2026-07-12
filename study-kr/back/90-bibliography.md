# 참고문헌 (Bibliography)

> **범위** — 25개 장(Part I ch01–11, Part II ch12–17, Part III ch18–25) 전체를 grep해 수집·dedupe한 인용 목록이다. 각 항목 끝의 `[인용:]`은 그 문헌을 인용하는 장이다.
> **표기 규약** (STYLE §2.5) — 6편 주 논문은 공식 축약 **[Titans]·[Miras]·[Atlas]·[TNT]·[NL]·[Sleep]**로, 외부 문헌은 저자-연도 + arXiv ID로 인용한다. arXiv 번호는 본문에 등장한 값을 그대로 싣고, 본문에 저자-연도만 있던 항목은 이 라운드에서 web으로 확인해 arXiv ID·저자·연도를 채웠다(확인 실패분은 §D 미해결).
> **검증 상태** — 6편 메타데이터는 `notes/{id}.json`(web-verified)에서, 외부 arXiv ID는 각 장 본문에서, 저자-연도-only 항목(§C 일부)과 P3 이월 항목은 2026-07-12 WebSearch로 교차확인했다.
> **2026-07-12 마감 라운드** — 본문 25장의 arXiv ID·GitHub 별 수·venue·issue #번호를 `notes/impl-availability.md` 및 web으로 spot-verify(in-text arXiv 토큰 71종이 모두 §A/B에 존재함을 역방향 확인). 마지막 미해결 "Kim et al. 2026" 해소(→ §B.9, §D). 그리고 본문에 inline으로만 인용돼(arXiv ID 없이) grep 수집에서 누락됐던 고전 15편 — Polyak 1964, Zinkevich 2003, Shalev-Shwartz 2011, McMahan 2011, Anderson 1972, Kohonen 1972, Hopfield 1982, Blelloch 1990, French 1999, McClelland et al. 1995, Williams 1992, Schmidhuber 1987/1992, Flash-Decoding 2023, Kung–Leiserson 1978 — 을 역방향 확인해 §B에 추가했다.

---

## A. 주 논문 — Google Titans/neural-memory 6부작

이 스터디의 척추. 모두 Ali Behrouz(제1저자 라인) + Vahab Mirrokni(Google) 계열. 저자·날짜는 `notes/*.json` 실측값.

- **[Titans]** — Behrouz, A., Zhong, P., & Mirrokni, V. (2025). *Titans: Learning to Memorize at Test Time.* arXiv:2501.00663 (v1: 2024-12-31). Google Research.
  [인용: ch01–ch14, ch16, ch17 — 라인 전체의 기준점]
- **[Miras]** — Behrouz, A., Razaviyayn, M., Zhong, P., & Mirrokni, V. (2025). *It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization.* arXiv:2504.13173. Google Research. (framework 명칭 **Miras**로 통칭 — STYLE §2.5)
  [인용: ch01–ch09, ch12–ch17]
- **[Atlas]** — Behrouz, A., Li, Z., Kacham, P., Daliri, M., Deng, Y., Zhong, P., Razaviyayn, M., & Mirrokni, V. (2025). *Atlas: Learning to Optimally Memorize the Context at Test Time.* arXiv:2505.23735. Google.
  [인용: ch01–ch10, ch13–ch17]
- **[TNT]** — Li, Z., Behrouz, A., Deng, Y., Zhong, P., Kacham, P., Karami, M., Razaviyayn, M., & Mirrokni, V. (2025). *TNT: Improving Chunkwise Training for Test-Time Memorization.* arXiv:2511.07343. USC / Google Research.
  [인용: ch01, ch02, ch04–ch10, ch14–ch17]
- **[NL]** — Behrouz, A., Razaviyayn, M., Zhong, P., & Mirrokni, V. (2025). *Nested Learning: The Illusion of Deep Learning Architecture.* arXiv:2512.24695. Google (Zhong: Columbia 겸직). NeurIPS 2025 버전 존재.
  [인용: ch01–ch11, ch14–ch17]
- **[Sleep]** — Behrouz, A., Hashemi, F., & Mirrokni, V. (2026). *Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories.* arXiv:2606.03979 (v1: 2026-06; OpenReview 2025-09 공개). Google / Cornell.
  [인용: ch01, ch02, ch04, ch08–ch11, ch14, ch16, ch17]
  ⚠ 혼동 주의: 별개 계열 논문 arXiv:2605.26099(Lee et al., *Do Language Models Need Sleep?*, offline recurrence)와 다름 — `notes/impl-availability.md` §2.6.

---

## B. 외부 문헌 — 계보별

### B.1 Linear attention · Fast Weight Programming (FWP)

- Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* arXiv:2006.16236. ICML 2020. [인용: ch06]
- Schlag, I., Irie, K., & Schmidhuber, J. (2021). *Linear Transformers Are Secretly Fast Weight Programmers.* arXiv:2102.11174. ICML 2021. [인용: ch06]
- Irie, K., Schlag, I., Csordás, R., & Schmidhuber, J. (2021). *Going Beyond Linear Transformers with Recurrent Fast Weight Programmers.* arXiv:2106.06295. NeurIPS 2021. (본문 "Irie & Schmidhuber 2021"(ch06)도 이 FWP 계보를 가리킴) [인용: ch06]
- Irie, K., Schlag, I., Csordás, R., & Schmidhuber, J. (2022). *A Modern Self-Referential Weight Matrix That Learns to Modify Itself.* arXiv:2202.05780. ICML 2022. (self-modifying Titans의 SRWM 계보) [인용: ch16]
- Irie, K., et al. (2023). *Practical Computational Power of Linear Transformers and Their Recurrent and Self-Referential Extensions.* arXiv:2310.16076. EMNLP 2023. (formal-language 인식 구성 — [NL Table 5]) [인용: ch16]
- Hua, W., Dai, Z., Liu, H., & Le, Q. (2022). *Transformer Quality in Linear Time* (FLASH). arXiv:2202.10447. ICML 2022. (mixed-chunk 기법 — chunkwise training의 기원) [인용: ch09]

### B.2 DeltaNet · Gated DeltaNet · state-tracking

- Yang, S., Wang, B., Zhang, Y., Shen, Y., & Kim, Y. (2024). *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* arXiv:2406.06484. NeurIPS 2024. (DeltaNet 병렬화; UT-transform 기반) [인용: ch06, ch09]
- Yang, S., Kautz, J., & Hatamizadeh, A. (2025). *Gated Delta Networks: Improving Mamba2 with Delta Rule* (Gated DeltaNet, 이하 GDN). arXiv:2412.06464. ICLR 2025. 공식 구현 NVlabs/GatedDeltaNet. **Kautz·Hatamizadeh = NVIDIA**, Yang = MIT. (P3 이월 "GDN NVIDIA affiliation" 확인 완료) [인용: ch06, ch21]
- Siems, J., et al. (2025). *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products.* arXiv:2502.10297. [인용: ch06]
- Grazzi, R., et al. (2025). *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues.* arXiv:2411.12537. [인용: ch06]
- Merrill, W., Petty, J., & Sabharwal, A. (2024). *The Illusion of State in State-Space Models.* arXiv:2404.08819. ICML 2024. [인용: ch06]
- Peng, B., et al. (2025). *RWKV-7 "Goose" with Expressive Dynamic State Evolution.* arXiv:2503.14456. [인용: ch06]

### B.3 SSM · Mamba 계보

- Gu, A., Dao, T., Ermon, S., Rudra, A., & Ré, C. (2020). *HiPPO: Recurrent Memory with Optimal Polynomial Projections.* arXiv:2008.07669. NeurIPS 2020. [인용: ch07]
- Gu, A., Goel, K., & Ré, C. (2022). *Efficiently Modeling Long Sequences with Structured State Spaces* (S4). arXiv:2111.00396. ICLR 2022. [인용: ch07]
- Smith, J. T. H., Warrington, A., & Linderman, S. W. (2023). *Simplified State Space Layers for Sequence Modeling* (S5). arXiv:2208.04933. ICLR 2023. [인용: ch07, ch09]
- Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752. [인용: ch07]
- Dao, T., & Gu, A. (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2, SSD). arXiv:2405.21060. ICML 2024. [인용: ch07]
- Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y. (2024). *Gated Linear Attention Transformers with Hardware-Efficient Training* (GLA). arXiv:2312.06635. ICML 2024. [인용: ch06, ch09]
- Sun, Y., Dong, L., Huang, S., et al. (2023). *Retentive Network: A Successor to Transformer for Large Language Models* (RetNet). arXiv:2307.08621. [인용: ch06, ch09]
- Martin, E., & Cundy, C. (2018). *Parallelizing Linear Recurrent Neural Nets Over Sequence Length.* arXiv:1709.04057. ICLR 2018. [인용: ch07]
- Blelloch, G. E. (1990). *Prefix Sums and Their Applications.* Technical Report CMU-CS-90-190, Carnegie Mellon University. (associative scan 알고리즘의 표준 참조 — S5의 parallel scan) [인용: ch07]

### B.4 Test-Time Training (TTT) · online-learning 관점의 sequence model

- Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A. A., & Hardt, M. (2020). *Test-Time Training with Self-Supervision for Generalization under Distribution Shifts.* arXiv:1909.13231. ICML 2020. (TTT 명칭의 직계 조상) [인용: ch08]
- Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021). *Tent: Fully Test-Time Adaptation by Entropy Minimization.* arXiv:2006.10726. ICLR 2021. [인용: ch08]
- Sun, Y., Li, X., Dalal, K., et al. (2024). *Learning to (Learn at Test Time): RNNs with Expressive Hidden States* (TTT-Linear / TTT-MLP). arXiv:2407.04620. (dual form의 원형; 라인의 직계 원형) [인용: ch01, ch06, ch08, ch09, ch15]
- Liu, B., Wang, R., Wu, L., et al. (2025). *Longhorn: State Space Models are Amortized Online Learners.* arXiv:2407.14207. (implicit-GD inner optimizer) [인용: ch03, ch06]
- Dalal, K., et al. (2025). *One-Minute Video Generation with Test-Time Training.* arXiv:2504.05298. (TTT layer의 대규모 실용성) [인용: ch08]
- Wang, K. A., Shi, J., & Fox, E. B. (2025). *Test-time regression: a unifying framework for designing sequence models with associative memory.* arXiv:2501.12352. (동시기 통합 framework; 본문 "Wang et al. 2025") [인용: ch13]
- Zhang, T., Bi, S., Hong, Y., et al. (2025). *Test-Time Training Done Right* (LaCT, Large-Chunk TTT). arXiv:2505.23884. (본문 "Zhang, Bi et al. 2025") [인용: ch15]
- von Oswald, J., Scherrer, N., Kobayashi, S., et al. (2025). *MesaNet: Sequence Modeling by Locally Optimal Test-Time Training.* arXiv:2506.05233. (chunk_cg_solver — 2차 방향 커널화 가능성의 실물) [인용: ch21]
- Guo, H., Yang, S., Goel, T., Xing, E. P., Dao, T., & Kim, Y. (2025). *Log-Linear Attention.* arXiv:2506.04761. [인용: ch15]

### B.5 병렬화 · optimizer · in-context learning as gradient descent

- Andrychowicz, M., et al. (2016). *Learning to learn by gradient descent by gradient descent.* arXiv:1606.04474. NeurIPS 2016. (learned optimizer) [인용: ch04]
- Lim, Y. H., et al. (2024). *Parallelizing non-linear sequential models over the sequence length.* arXiv:2309.12252. ICLR 2024. (Newton 고정점 병렬화) [인용: ch15]
- Gonzalez, X., Warrington, A., Smith, J. T. H., & Linderman, S. W. (2024). *Towards Scalable and Stable Parallelization of Nonlinear RNNs.* arXiv:2407.19115. NeurIPS 2024. [인용: ch15]
- von Oswald, J., et al. (2023). *Transformers learn in-context by gradient descent.* arXiv:2212.07677. ICML 2023. [인용: ch04]
- Akyürek, E., Schuurmans, D., Andreas, J., Ma, T., & Zhou, D. (2023). *What learning algorithm is in-context learning? Investigations with linear models.* arXiv:2211.15661. ICLR 2023. [인용: ch04]
- von Oswald, J., et al. (2023). *Uncovering mesa-optimization algorithms in Transformers.* arXiv:2309.05858. [인용: ch04]
- Bietti, A., Cabannes, V., Bouchacourt, D., Jégou, H., & Bottou, L. (2023). *Birth of a Transformer: A Memory Viewpoint.* arXiv:2306.00802. NeurIPS 2023. [인용: ch05]

### B.6 Optimizer · 훈련 기초 (Part I 배경)

- Robbins, H., & Monro, S. (1951). *A Stochastic Approximation Method.* Annals of Mathematical Statistics 22(3). (SGD의 기원) [인용: ch02]
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning representations by back-propagating errors.* Nature 323. [인용: ch02]
- Polyak, B. T. (1964). *Some methods of speeding up the convergence of iteration methods.* USSR Computational Mathematics and Mathematical Physics 4(5). (heavy-ball momentum의 기원) [인용: ch02]
- Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). *On the importance of initialization and momentum in deep learning.* ICML 2013. [인용: ch02]
- Duchi, J., Hazan, E., & Singer, Y. (2011). *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization* (AdaGrad). JMLR 12. [인용: ch02]
- Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* arXiv:1412.6980. ICLR 2015. [인용: ch02]
- Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization* (AdamW). arXiv:1711.05101. ICLR 2019. [인용: ch02]
- Gupta, V., Koren, T., & Singer, Y. (2018). *Shampoo: Preconditioned Stochastic Tensor Optimization.* arXiv:1802.09568. ICML 2018. [인용: ch02]
- Jordan, K., et al. (2024). *Muon: An optimizer for the hidden layers of neural networks.* Blog: kellerjordan.github.io/posts/muon. (NS-5, $(a,b,c)=(3.4445,-4.7750,2.0315)$) [인용: ch02, ch14]
- Liu, J., et al. (2025). *Muon is Scalable for LLM Training.* arXiv:2502.16982. [인용: ch02]
- Joffrain, T., Low, T. M., Quintana-Ortí, E. S., van de Geijn, R., & Van Zee, F. G. (2006). *Accumulating Householder transformations, revisited* (UT transform). ACM TOMS 32(2). [인용: ch09]
- Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). *Visualizing the Loss Landscape of Neural Nets.* arXiv:1712.09913. NeurIPS 2018. [인용: ch03]

### B.7 Online learning · meta-learning · bilevel

- Zinkevich, M. (2003). *Online Convex Programming and Generalized Infinitesimal Gradient Ascent.* ICML 2003. (OGD와 $O(\sqrt{L})$ regret 정리) [인용: ch03]
- Shalev-Shwartz, S. (2011). *Online Learning and Online Convex Optimization.* Foundations and Trends in Machine Learning 4(2). (OCO 교과서적 정리; 본문의 "Shalev-Shwartz 2011"과 "Shalev-Shwartz 2012, §2.6"[ch03]은 발행연도 표기만 다른 동일 monograph) [인용: ch03]
- McMahan, H. B. (2011). *Follow-the-Regularized-Leader and Mirror Descent: Equivalence Theorems and L1 Regularization.* AISTATS 2011. (FTRL↔mirror descent) [인용: ch03]
- Hazan, E. (2019). *Introduction to Online Convex Optimization* (2nd ed.). arXiv:1909.05207. [인용: ch03]
- Orabona, F. (2019). *A Modern Introduction to Online Learning.* arXiv:1912.13213. [인용: ch03]
- Finn, C., Abbeel, P., & Levine, S. (2017). *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks* (MAML). arXiv:1703.03400. ICML 2017. [인용: ch04]
- Franceschi, L., Frasconi, P., Salzo, S., Grazzi, R., & Pontil, M. (2018). *Bilevel Programming for Hyperparameter Optimization and Meta-Learning.* arXiv:1806.04910. ICML 2018. [인용: ch04]
- Hospedales, T., Antoniou, A., Micaelli, P., & Storkey, A. (2021). *Meta-Learning in Neural Networks: A Survey.* arXiv:2004.05439. IEEE TPAMI. [인용: ch04]

### B.8 Associative memory · Hopfield

- Widrow, B., & Hoff, M. E. (1960). *Adaptive switching circuits* (LMS / delta rule). IRE WESCON Convention Record. [인용: ch05, ch06, ch08]
- Anderson, J. A. (1972). *A simple neural network generating an interactive memory.* Mathematical Biosciences 14(3–4). (correlation matrix memory의 공동 기원) [인용: ch05]
- Kohonen, T. (1972). *Correlation Matrix Memories.* IEEE Transactions on Computers C-21(4). (outer-product 저장의 원형) [인용: ch05]
- Hopfield, J. J. (1982). *Neural networks and physical systems with emergent collective computational abilities.* Proceedings of the National Academy of Sciences (PNAS) 79(8). (energy 기반 auto-associative memory; Atlas/TNT가 capacity 정식화의 뿌리로 인용) [인용: ch05]
- Krotov, D., & Hopfield, J. J. (2016). *Dense Associative Memory for Pattern Recognition.* arXiv:1606.01164. NeurIPS 2016. [인용: ch05]
- Ramsauer, H., et al. (2021). *Hopfield Networks is All You Need.* arXiv:2008.02217. ICLR 2021. [인용: ch05]
- Sukhbaatar, S., Grave, E., Bojanowski, P., & Joulin, A. (2019). *Augmenting Self-attention with Persistent Memory.* arXiv:1907.01470. [인용: ch12]

### B.9 Continual learning · distillation · self-improvement

- McCloskey, M., & Cohen, N. J. (1989). *Catastrophic Interference in Connectionist Networks.* Psychology of Learning and Motivation 24. [인용: ch11]
- French, R. M. (1999). *Catastrophic forgetting in connectionist networks.* Trends in Cognitive Sciences 3(4). (CF·pseudo-rehearsal 계보 정리) [인용: ch11]
- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). *Why there are complementary learning systems in the hippocampus and neocortex.* Psychological Review 102(3). (CLS 이론 — Sleep의 wake/sleep 분업의 신경과학 근거) [인용: ch11]
- Williams, R. J. (1992). *Simple statistical gradient-following algorithms for connectionist reinforcement learning* (REINFORCE). Machine Learning 8(3–4). (policy-gradient 항등식) [인용: ch11]
- Kirkpatrick, J., et al. (2017). *Overcoming catastrophic forgetting in neural networks* (EWC). arXiv:1612.00796. PNAS 114(13). [인용: ch11]
- Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531. [인용: ch11]
- Kim, Y., & Rush, A. M. (2016). *Sequence-Level Knowledge Distillation.* arXiv:1606.07947. EMNLP 2016. [인용: ch17]
- Agarwal, R., et al. (2024). *On-Policy Distillation of Language Models* (GKD). arXiv:2306.13649. ICLR 2024. [인용: ch11]
- Kim, J., Luo, X., Kim, M., Lee, S., Kim, D., Jeon, J., Li, D., & Yang, Y. (2026). *Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?* arXiv:2603.24472 (2026-03-25). (본문 "Kim et al. 2026" — [Sleep App. A.4]가 인용한 OPSD 실패 모드: epistemic verbalisation 억제 → 최대 40% OOD 하락) [인용: ch17]
- Singh, A., et al. (2024). *Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models* (ReST^EM). arXiv:2312.06585. TMLR 2024. [인용: ch11]
- Ouyang, L., et al. (2022). *Training language models to follow instructions with human feedback* (InstructGPT). arXiv:2203.02155. NeurIPS 2022. [인용: ch11]
- Zweiger, A., et al. (2025). *Self-Adapting Language Models* (SEAL). arXiv:2506.10943. (Dreaming의 인접 계보) [인용: ch11]
- Hu, E. J., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685. ICLR 2022. [인용: ch11]
- Lin, K., Snell, C., Wang, Y., et al. (2025). *Sleep-time Compute: Beyond Inference Scaling at Test-time.* arXiv:2504.13171. (텍스트-공간 offline compute — Sleep과 대비) [인용: ch17]
- Eyuboglu, S., Ehrlich, R., Arora, S., et al. (2025). *Cartridges: Lightweight and general-purpose long context representations via self-study.* arXiv:2506.06266. (KV-공간 offline 압축 — Sleep과 대비) [인용: ch17]

### B.10 Systems · serving · scaling law

- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* arXiv:2205.14135. NeurIPS 2022. [인용: ch10, ch21]
- Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* arXiv:2307.08691. [인용: ch10]
- Dao, T., Haziza, D., Massa, F., & Sizov, G. (2023). *Flash-Decoding for Long-Context Inference.* PyTorch / Stanford CRFM 블로그. (FlashAttention의 IO 재조직을 decode까지 확장 — ch21의 prefill/decode 대칭 논거) [인용: ch21]
- Kung, H. T., & Leiserson, C. E. (1978). *Systolic Arrays (for VLSI).* Sparse Matrix Proceedings 1978, SIAM. (systolic array = TPU MXU / GPU tensor core의 조상; GEMM density 공진화 서술) [인용: ch21]
- Kwon, W., et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM). arXiv:2309.06180. SOSP 2023. [인용: ch10]
- Hoffmann, J., et al. (2022). *Training Compute-Optimal Large Language Models* (Chinchilla). arXiv:2203.15556. NeurIPS 2022. [인용: ch19]
- Hooker, S. (2020). *The Hardware Lottery.* arXiv:2009.06489. [인용: ch21]
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet Classification with Deep Convolutional Neural Networks* (AlexNet). NeurIPS 2012. [인용: ch21]

### B.11 교과서 · 고전 참조

- Schmidhuber, J. (1987). *Evolutionary Principles in Self-Referential Learning* (diploma thesis). TU München. (학습 절차 자체를 학습 대상으로 삼는 self-referential learning) [인용: ch16]
- Schmidhuber, J. (1992). *Learning to control fast-weight memories: An alternative to dynamic recurrent networks.* Neural Computation 4(1). (fast weight programming의 원형) [인용: ch04, ch06, ch16]
- Schmidhuber, J. (1993). *A "self-referential" weight matrix.* ICANN 1993. (self-modifying/self-referential 계보의 뿌리) [인용: ch04, ch06, ch16]
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press. (backprop·optimizer 교과서 서술) [인용: ch02]

---

## C. 구현·저장소 인용 (implementation citations)

Part III(ch21·ch22·ch24)의 hardware-lottery / player-strategy / proposals 논증은 아래 저장소를 실측 증거로 인용한다. star·issue 번호는 `notes/impl-availability.md`(2026-07-11 GitHub API 실측) 기준.

- **lucidrains/titans-pytorch** (GitHub, 1,966★). 사실상의 reference 구현(MAC variant + 저자 확장). AssocScan 기반 병렬화. [인용: ch21, ch24]
- **fla-org/flash-linear-attention** (FLA, 5,325★). DeltaNet/GDN/GLA/RWKV-7/Mamba-2 등의 production-grade Triton 커널. `fla/ops/ttt`(진짜 Triton) vs `fla/ops/titans`(naive PyTorch만). RFC issue #107, 정체 #214. [인용: ch09, ch21, ch22, ch24]
- **fla-org/flame** (torchtitan 기반 학습 프레임워크). [인용: ch24]
- **NVlabs/GatedDeltaNet** (619★, ICLR 2025 공식 GDN 구현). [인용: ch21]
- **obekt/HOPE-nested-learning** (84★) · **erikl2/nested-learning** (76★). HOPE 비공식 소규모 구현(참조·감사 대상, 신뢰 기반 아님; 둘 다 surrogate loss 혼합). [인용: ch24]

---

## D. 미해결 인용 (unresolved)

- (없음 — 이 라운드에서 마지막 미해결 항목이던 "Kim et al. 2026"이 해소되어 §B.9로 편입됨. 아래 "확인 완료" 참조.)

### 확인 완료(과거 미해결 → 해소)

- **Kim et al. 2026** (직전 미해결) — *해소.* [Sleep App. A.4]가 OPSD 실패 모드로 인용한 "epistemic verbalisation 억제 → 최대 40% OOD 하락"의 원전은 Jeonghye **Kim** et al. (2026), *Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?*, **arXiv:2603.24472** (2026-03-25; Qwen3-1.7B/8B·DeepSeek-Distill-Qwen-7B·Olmo3-7B-Instruct에서 최대 40% 하락, "epistemic verbalization 억제" 기제 명시). 2026-07-12 WebSearch+arXiv 초록 대조로 저자·제목·수치 3중 확인. §B.9에 정식 등재. (Sleep 부록은 arXiv ID 없이 저자-연도만 표기했으나 외부 식별 가능했음.)
- **[GDN NVIDIA affiliation]** (P3 이월) — *해소.* Gated DeltaNet(arXiv:2412.06464)의 Jan Kautz·Ali Hatamizadeh는 NVIDIA, Songlin Yang은 MIT. 공식 구현 NVlabs/GatedDeltaNet, ICLR 2025. (§B.2)
