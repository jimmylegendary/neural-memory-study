# Early Wake--Sleep Precursor Primary-Source Audit

- Audit date: 2026-07-25
- Scope: seven direct algorithmic precursors spanning 2017--2025
- Evidence boundary: latest public paper, supplement, official venue record,
  and official code repository where one could be established
- Chronology rule: earliest verifiable public disclosure, not inferred
  invention date
- Phase notation: `W` wake update, `B` explicit boundary, `S` query-independent
  offline/sleep update, `Q` read-only query, and `P` unimplemented proposal
- Status: pre-ingestion audit. Stable source, evidence, and claim identifiers
  are assigned only after the evidence registry is live.

## Executive finding

The high-level claim that machine sleep, wake--sleep dual memory, parametric
consolidation, sleep-time pruning, NREM/REM-inspired objectives, or learned
sleep control is new to LLM agents is not supportable.

The primary-source chronology already contains:

| First public disclosure | Prior-art boundary |
|---|---|
| 2017-10, DGDMN | STM-capacity-triggered, query-independent generative replay into durable LTM weights |
| 2017-11, FearNet | fixed-cadence generated-old plus recent-real replay followed by STM clearing |
| 2018-05, Progress & Compress | task-boundary distillation from a plastic active model into a stable fixed-size knowledge base |
| 2023-03, SIESTA | explicitly compute-restricted backpropagation-free wake and latent-replay sleep |
| 2023-12, WSCL | distinct NREM and REM batches, albeit with external labeled “dream” data |
| 2024-09, PCMC | sleep-time representation retraining, re-indexing, and raw-patch pruning |
| 2025-09, Spens et al. | PPO control over sleep operators and learned replay valuation in selected toy tasks |

The remaining defensible research space is the end-to-end lifecycle:
versioned, provenance-bearing events; joint routing among raw text, semantic
text/vector/graph, latent/KV, user-local parameters, and shared parameters;
actual idle-window and resource scheduling; snapshot isolation and delta
transport; promotion validation; selective deletion; rollback; and matched
lifetime accounting.

## 1. Deep Generative Dual Memory Network

**Identity.** Nitin Kamra, Umang Gupta, and Yan Liu, *Deep Generative Dual
Memory Network for Continual Learning*, USC Computer Science.
[arXiv:1710.10368v1](https://arxiv.org/abs/1710.10368) was public on
2017-10-28. OpenReview records an ICLR 2018 blind submission, but this audit
did not establish final conference publication. The safest status is therefore
arXiv/CoRR preprint and ICLR submission.

### Actual lifecycle

- Phase: `Q + W + B + S`.
- Wake input is a complete supervised task batch
  `{task descriptor, X, Y}`, not an individual interaction stream.
- A fresh short-term memory DGM is allocated for a new task. If a previously
  consolidated task returns, LTM reconstructs and labels current inputs before
  adding them to the corresponding short-term model.
- Sleep begins when all short-term slots are occupied. Main experiments use
  `nSTM=2`; TDigits uses five.
- Each short-term learned VAE generates samples, its learned classifier labels
  them, and the LTM VAE/classifier is retrained by Deep Generative Replay.
  Short-term slots are then freed.
- The durable destination is LTM generator and classifier weights plus the task
  dictionary and age counter. Querying routes by task identity and does not
  update memory.

The replay allocator starts with new-task and generated-old allocations,
enforces a per-update `Nmax`, and reserves at least a hand-set `κ` fraction for
new tasks. The classifier uses cross-entropy. All models use RMSProp,
learning rate `0.001`, `rho=0.9`, `epsilon=1e-8`, batch 128; classifiers train
for six epochs and VAEs for 25. The paper identifies the generator as a VAE
but does not report the exact ELBO, reconstruction formula, KL weight, or
annealing schedule.

Learned components are the VAE and classifier in every short- and long-term
module. Hand-set components include task dictionaries, sleep trigger, replay
allocation, `κ`, `Nmax`, task routing, and the descriptor-free reconstruction
threshold `gamma_dgm=1.55`.

### Capacity, evidence, and nonclaims

`Nmax` limits examples in one replay update, not model parameters, lifetime
compute, or total information. In four of five datasets it equals the complete
dataset size. Only TDigits limits it to 120,000, half the total. Old tasks
receive a declining replay fraction, and the paper itself observes compounded
generator error and blurred older samples.

| Dataset | DGDMN ACC | DGDMN BWT | DGR ACC | DGR BWT |
|---|---:|---:|---:|---:|
| Digits, 10 tasks | .818 | -.150 | .596 | -.425 |
| Permnist, 6 tasks | .831 | -.075 | .861 | -.068 |
| Shapes, 6 tasks | .722 | -.261 | .661 | -.288 |
| Hindi, 8 tasks | .658 | -.335 | .731 | -.270 |

Dual memory is therefore not uniformly superior to single-memory DGR. The main
system requires task identity; descriptor-free recognition is one Digits
ablation. Streaming and reinforcement-learning extensions are proposals, not
demonstrations. There is no targeted deletion, rollback, version lineage,
per-example provenance, or retention policy. No run count, confidence
interval, significance test, or official code was found; exact short-term
architectures, permutations, seeds, and VAE loss are also missing.

## 2. FearNet

**Identity.** Ronald Kemker and Christopher Kanan, *FearNet: Brain-Inspired
Model for Incremental Learning*, RIT Carlson Center for Imaging Science.
[arXiv:1711.10563v1](https://arxiv.org/abs/1711.10563) was public on
2017-11-28; [OpenReview](https://openreview.net/forum?id=SJ1Xmf-Rb) establishes
ICLR 2018 publication.

### Actual lifecycle

- Phase: `Q + W + B + S`.
- The first session is a large multi-class base batch. Each later study session
  contains all examples from one new class.
- Inputs are fixed pretrained embeddings: ImageNet-pretrained ResNet-50 for
  CIFAR/CUB and a pretrained YouTube-8M audio CNN for AudioSet.
- During wake, the hippocampal component stores every recent feature vector and
  answers with a deterministic probabilistic nearest exemplar.
- Main experiments sleep after every ten new class sessions.
- During sleep, each old class is sampled from a Gaussian fitted in the mPFC
  encoder space; decoded pseudo-examples are mixed with all real recent
  examples, the mPFC autoencoder/classifier is fine-tuned, class statistics are
  recomputed, and all recent exemplars are deleted.
- A learned binary BLA router estimates whether an item belongs in mPFC or HC.
  Querying combines router and subsystem confidence without updating memory.

The mPFC objective is classification plus weighted matched-layer MSE
reconstruction. NAdam uses learning rate `2e-3`; the decoder rate is one
hundredth of the encoder rate. Base training uses 1,000 epochs, each sleep 60,
and BLA training 20. Hidden sizes, batch sizes, and reconstruction weights are
dataset-specific. Learned components are mPFC and BLA; the sleep counter,
Gaussian fit, pseudo-example count, HC rule, HC clearing, and final routing
equation are fixed.

### Capacity, evidence, and nonclaims

HC retains every recent real example until sleep, so “does not store previous
examples” applies only after consolidation. The mPFC network is fixed-size, but
full per-class means and covariance matrices grow with class count. The
reported 74.4 MB at 1,000 classes is extrapolated, not measured. Full covariance
at 100 CIFAR classes uses 10.7 MB and reports
`(Omega_base, Omega_new, Omega_all)=(.942,.805,.959)`; diagonal covariance uses
3.8 MB and reports `(.781,.877,.800)`.

| Dataset | Omega base | Omega new | Omega all |
|---|---:|---:|---:|
| CIFAR-100 | .927 | .824 | .947 |
| CUB-200 | .924 | .598 | .891 |
| AudioSet-100 | .962 | .455 | .932 |

Routing is a material failure mode. Oracle-to-BLA `Omega_new` falls
`.912 -> .824` on CIFAR, `.729 -> .598` on CUB, and `.701 -> .455` on
AudioSet. Frequent global mPFC updates can gradually damage older memories;
the paper explicitly reports the retention/plasticity cadence trade-off.
Repeated appearances of one class remain unresolved.

The evaluation is not raw-input continual learning and starts with half the
classes trained for 1,000 epochs. The current official CUB documentation warns
of overlap with ImageNet, creating a plausible pretrained-feature leakage
confound. This is an audit inference, not a claim in the paper. There is no
selective deletion, rollback, source lineage, or correction mechanism. No
official code, exact AudioSet subset, class order, seed count, or dependency
lock was found.

## 3. Progress & Compress

**Identity.** Jonathan Schwarz et al., *Progress & Compress: A Scalable
Framework for Continual Learning*.
[arXiv:1805.06370](https://arxiv.org/abs/1805.06370) was public on
2018-05-16; the final paper is
[ICML 2018, PMLR 80](https://proceedings.mlr.press/v80/schwarz18a.html).
The work is DeepMind London-led; Jelena Luketina is also affiliated with Oxford.

### Actual lifecycle

- Phase: `W + B/S`; it is boundary consolidation, not autonomous wall-clock
  sleep.
- In the progress phase the knowledge base is frozen while an active column
  and lateral adapter learn the current task.
- A task-distribution change/completion supplied by the environment triggers
  compression.
- The active column becomes teacher and the fixed-size knowledge base becomes
  student. KL/cross-entropy distillation uses current-task inputs or visited
  states, while online EWC protects the previous knowledge base.
- The active column is then reinitialized or reused. Durable state is the
  knowledge-base policy/classifier/value weights and a running Fisher/MAP
  approximation.

The running importance update
`F_star_i = gamma F_star_(i-1) + F_i` makes `gamma<1` an explicit,
hand-set ageing mechanism. Omniglot uses 2,500 SGD steps per task, learning
rate `.1`, distillation rate `.05`, and temperature two. Reinforcement-learning
experiments use distributed actor-critic with RMSProp and 50 million frames per
Atari visit.

### Evidence and nonclaims

- Omniglot over five passes improves from `70.32 +/- 3.3` to
  `82.84 +/- 1.4` with 659K parameters.
- Dedicated models reach 88.34 with 5.68M parameters and Progressive Nets 86.50
  with 108M, so P&C's result is principally a fixed-capacity trade-off.
- Six-game Atari, four seeds, reports forward transfer around 125--131% with
  active reinitialization. Without reinitialization it falls from 131% to 101%
  over visits.
- Omniglot does not show final cross-alphabet generalization improvement, and
  online EWC is competitive. On diverse Atari tasks EWC-family methods can
  incur more than 40% negative transfer.

The method requires known task boundaries and current-task inputs, has no
open-ended stream policy, exact deletion, rollback, or provenance, and does not
bound lifetime update compute. No official implementation linked by the paper
or identifiable in the official DeepMind research repository was found.

## 4. SIESTA

**Identity.** Yousuf Harun et al., *SIESTA: Efficient Online Continual Learning
with Sleep*. [arXiv:2303.10725v1](https://arxiv.org/abs/2303.10725) was public
on 2023-03-19; [OpenReview](https://openreview.net/forum?id=MqDVlBWRRV)
establishes TMLR publication on 2023-11-01. Affiliations span RIT, the US Space
Force, and the University of Rochester.

### Actual lifecycle

- Phase: explicit `W <-> S`.
- Wake receives one labeled image at a time. A frozen lower network produces a
  `14 x 14 x 80` latent; OPQ compresses it into a bounded replay store.
- The output classifier weight is updated to a running class mean without
  backpropagation.
- At capacity, a random exemplar is evicted from the most populated class.
- Main experiments sleep every 120,000 samples, approximately every 100
  classes.
- Sleep uses balanced-uniform latent replay to update the upper network and
  classifier—97.81% of MobileNetV3-L parameters—under supervised
  cross-entropy. The lower encoder remains frozen.
- The no-augmentation setting performs 1.28 million per-example backward
  updates per sleep; the augmentation setting performs 6.4 million and adds
  latent Mixup/CutMix.

Sleep uses SGD with OneCycle, batch 64, momentum `.9`, weight decay `1e-5`,
final-layer initial learning rate `.2`, and layerwise decay `.99`. Learned
components include SwAV initialization, OPQ fitting, sleep-time upper-network
weights, and a temperature. Running means, balanced replay, fixed cadence, and
most-populated-class random eviction are not learned.

### Evidence and nonclaims

| ImageNet-1K setting | Average top-5 | Final top-5 | Memory | Updates |
|---|---:|---:|---:|---:|
| SIESTA, no augmentation | 88.33 | 83.59 | 2.02 GB | 11.53M |
| Offline final comparator | — | 83.31 | — | — |
| SIESTA, augmentation | 90.67 | 87.00 | 2.02 GB | — |
| Offline augmentation final | — | 90.74 | — | — |

The mean immediate sleep gain is `4.25 +/- 1.38` percentage points. At batch
512, processing 900 classes takes 1.89 hours on one RTX A5000. Reducing the
buffer from 2.01 GB to 0.75 GB lowers final top-5 from 83.59 to 80.57.

“Zero forgetting” is not a measured zero-forgetting guarantee: the
no-augmentation final result is merely not statistically different from its
offline comparator (`p=.08`), while the augmentation result differs
significantly (`p<.001`). Sleeping every 50 classes performs worse than every
100, consistent with repeated weight perturbation. The largest buffer contains
1,281,167 compressed instances—nearly the full ImageNet training-set count.
The method assumes a frozen lower representation remains useful for future
domains and lacks selective deletion, rollback, or provenance.

The [official repository](https://github.com/yousuf907/SIESTA) provides
PyTorch code, an MIT license, pretrained weights, and an environment, but not a
versioned release/package.

## 5. Wake-Sleep Consolidated Learning

**Identity.** Stefano Sorrenti et al., *Wake-Sleep Consolidated Learning*,
PeRCeiVe Lab, University of Catania. The first public
[arXiv:2401.08623](https://arxiv.org/abs/2401.08623) version dates to
2023-12-06. The archival result is IEEE TNNLS 36(7):12668--12679, July 2025;
accepted 2024-09-06 and first published online 2024-09-26.

### Actual lifecycle

- Phase: `W + B + S(NREM <-> REM)` with known task boundaries.
- Wake trains and compares allowable cumulative layer-freezing masks for one
  epoch each, using the current task and 10% of long-term memory, then builds a
  5,000-sample short-term memory.
- Sleep runs ten epochs, alternating NREM and REM batches.
- NREM optimizes the plugged-in continual-learning loss over STM plus the
  remaining 90% of LTM. Reservoir sampling moves selected STM samples into LTM.
- REM performs ordinary supervised learning on a separate labeled dataset
  whose classes do not overlap the continual-learning classes.
- Main experiments use ResNet-18 and mini-batch SGD at learning rate `.03`;
  batch is 32 for CIFAR/Tiny-ImageNet and eight for FG-ImageNet.

The model weights and selected freezing mask are learned. The boundary,
prefix-mask search space, 10/90 split, reservoir sampler, ten sleep epochs, and
external dream dataset are fixed.

An important version boundary must be preserved. ArXiv v1 contains an
unquantified statement that replacing the external data with GAN samples made
no significant difference. The final IEEE paper removes that statement and
lists internal latent/generative dreaming as future work. The evaluated REM
data are real, labeled external images—not generated dreams.

### Evidence and nonclaims

All final-paper headline results average five runs.

| ER-ACE, buffer 500 | Baseline | With WSCL |
|---|---:|---:|
| CIFAR-10 | 67.17 +/- 1.54 | 74.18 +/- 1.28 |
| CIFAR-100 | 32.85 +/- 4.02 | 39.78 +/- .36 |
| Tiny-ImageNet half | 32.10 +/- 2.21 | 41.25 +/- 1.75 |
| FG-ImageNet | 11.58 +/- 3.59 | 20.51 +/- .56 |

On the Tiny ablation, Wake produces FAA/FWT `4.70/-0.93`; Wake+REM
`25.68/11.89`; Wake+NREM `27.61/-0.67`; and full WSCL `35.68/8.60`.
Positive transfer is not universal: full WSCL FWT on CORe50 is `-5.62`.
For some DER++/CIFAR-10 settings WSCL increases forgetting, and ER-ACE
CIFAR-10 FWT remains `-1.87`.

The external-data benefit is not separated from adding target-disjoint labels
and compute. Mask selection trains multiple candidates, so its search cost must
be charged. The method has no learned dream generator, autonomous trigger,
selective deletion, rollback, provenance, or external-data lifecycle
accounting. The [official repository](https://github.com/perceivelab/wscl)
uses CC BY-NC 4.0 and has no formal release.

## 6. Patch-Based Contrastive Learning and Memory Consolidation

**Identity.** Cameron Taylor et al., *Patch-Based Contrastive Learning and
Memory Consolidation for Online Unsupervised Continual Learning*.
[arXiv:2409.16391](https://arxiv.org/abs/2409.16391) was public on
2024-09-24. The paper identifies CoLLAs 2024; the archival record is
[PMLR 274:938--958](https://proceedings.mlr.press/v274/taylor25a.html), 2025.
Affiliations include Georgia Tech, CYENS, and the Cyprus Institute.

### Actual lifecycle

- Phase: `W <-> fixed S`.
- Wake splits each unlabeled image into overlapping patches and embeds them.
  A percentile novelty threshold either creates an STM centroid or updates the
  nearest centroid by fixed-rate moving average and stores the raw patch.
- STM is fixed-capacity LRU. After `theta=30` matches, a centroid and its raw
  patches are copied to LTM.
- Main experiments sleep once at the midpoint of each streaming task. A learned
  trigger is future work.
- Sleep retrains the ResNet encoder for 300 epochs over raw STM and LTM patches
  with a SimCLR-style patch contrastive objective.
- SGD uses initial learning rate `.6`, momentum `.9`, weight decay `1e-5`,
  cosine scheduling, and batch 512.
- Centroids are re-embedded with the new encoder. A centroid loses one raw patch
  stochastically according to its excess occupancy:
  `P_prune=((|M_j|-M_min)/(M-M_min))^k`.
- STM patch memory is cleared after sleep; LTM centroid and raw-patch state
  persists.

The encoder is learned. Novelty thresholding, moving averages, LRU,
copy threshold, first-patch re-embedding, cadence, and pruning rule are fixed.

### Evidence and nonclaims

Across three seeded trials:

| Dataset, M=20 | PCMC accuracy | No-consolidation | PCMC max memory | No-consolidation max memory |
|---|---:|---:|---:|---:|
| ImageNet-40 | 59.7 +/- 1.34 | 60.0 +/- 1.21 | 8,819 image-equivalents | 12,715 |
| Places365-40 | 55.7 +/- .98 | 55.8 +/- .97 | 8,660 | 12,800 |

Thus the primary demonstrated benefit is roughly 30% raw-patch reduction with a
small mean accuracy cost. The paper explicitly states that neither total
centroid count nor total raw-patch count has a direct cap. ImageNet-40 adds
about 733 patches per task on average; pruning removes local redundancy rather
than old centroid identities.

Despite the “cannot store observed data” framing, the implementation retains
raw patches. The evaluation classifier uses 100 labeled train and 100 labeled
test examples per class, so only representation construction is unsupervised.
Three-hundred-epoch sleeps, a fixed midpoint, and a 16-task/40-class vision
stream do not establish resource-bounded lifelong behavior. There is no
per-user deletion, rollback, or provenance.

The [official repository](https://github.com/CameronTaylorFL/upl-benchmark)
has only two commits and no formal release or checkpoint. Its README says MIT,
but this audit found no LICENSE file and noted a stale “IJCAI 2021” reference;
redistribution rights therefore require separate review.

## 7. Modelling the Control of Offline Processing with Reinforcement Learning

**Identity.** Ellie Spens, Neil Burgess, and Timothy Behrens, *Modelling the
Control of Offline Processing with Reinforcement Learning*. First public on
OpenReview on 2025-09-18; final
[NeurIPS 2025 Main Track](https://papers.nips.cc/paper_files/paper/2025/hash/d7e5870810331da5a8ac8bd16d42e074-Abstract-Conference.html).
Affiliations are Oxford and University College London.

### Actual lifecycle

- Phase: explicit `W <-> S` inside bounded simulated meta-episodes.
- Image and maze setups perform exactly three sleep actions after every wake
  acquisition. The relational setup permits at most ten actions and stops each
  step with probability `.05`.
- A recurrent PPO/LSTM meta-controller learns which task-specific offline
  operator to execute. Its reward is performance in the next wake: classifier
  accuracy, maze success, or correctly inferred graph edges.

The three action spaces differ materially:

1. **Image:** wake receives 200 Fashion-MNIST examples. Actions update a GMM,
   replay 50 real memories, or replay 50 GMM samples. A linear value estimator
   is trained on approximate Shapley marginal contributions for a candidate
   quarter, after which deterministic MMR with `lambda=.9` combines value and
   diversity.
2. **Maze:** wake receives 50 trajectories in a changing `6 x 6` maze.
   Actions update a transition model, oversample 50 real trajectories to 200,
   or replay 200 simulated trajectories. The lower agent is a DQN with learning
   rate `1e-4` and epsilon `.2`.
3. **Relational:** wake receives 20 graph edges. Six actions include graph
   consolidation, GCN-autoencoder training, and inferred-edge expansion. It
   omits learned data valuation and retrains the world model each episode.

Learned elements are the PPO policy/critic, lower learners, world/generative
models, and—in image and maze only—the Shapley-supervised value estimator.
Shapley label computation, MMR, candidate construction, action budget, and
reset schedule are fixed.

### Evidence and nonclaims

- Image, 25 meta-episodes: controller CL/non-CL `.307/.411`, mean
  `.359 +/- .010 SEM`; memory-only `.171/.411`, mean `.291 +/- .003`;
  generative-only `.307/.355`, mean `.331 +/- .011`.
- Image valuation, ten episodes: memory replay `.438 +/- .007` versus random
  `.307 +/- .017`; generated replay `.373 +/- .007` versus random
  `.290 +/- .013`.
- Maze, ten incremental updates and three seeds: controller
  `.822 +/- .007` versus memory-only `.161 +/- .005`; learned valuation
  `.871 +/- .010` versus random `.822 +/- .007`; longest-path selection
  `.736 +/- .013`.
- Relational results emphasize curricula and action-frequency plots rather than
  a comparable quantitative baseline table.

The image continual learner, generator, and valuation model are reinitialized
after every five episodes in which all ten classes have appeared. Shapley
valuation is costly and the image setup assumes a separate validation set.
Controller training uses random selection for efficiency; valuation is trained
separately. These are short toy simulations, not an LLM/session lifetime.
Relational world-model reset weakens any durable-consolidation interpretation.
There is no lifetime capacity, deletion, rollback, provenance, privacy, or
wall-clock idle-window experiment.

The [official code](https://github.com/ellie-as/rl-with-metacognitive-actions)
contains notebooks, seeded logs, and dataset download, but no license or
versioned release.

## Cross-paper capacity conclusion

These systems defer rather than solve finite lifetime capacity:

- DGDMN caps replay examples per update, not lifetime state.
- FearNet clears recent examples but accumulates per-class Gaussian statistics.
- Progress & Compress weakens old tasks through `gamma` inside fixed weights.
- SIESTA and WSCL bound replay buffers with random/reservoir replacement.
- PCMC prunes redundant patches but explicitly has no global LTM cap.
- Spens et al. optimize short action curricula without a lifetime capacity
  experiment.

Consequently, the research program must distinguish at least five resources:
active serving bytes, durable bytes, parameter/module count, cumulative sleep
compute, and retrieval competition. “Consolidated” does not mean “capacity was
freed” unless raw state is compressed, migrated, or deleted.

## Novelty boundary for the present program

Do not claim:

- first sleep-time compute;
- first wake--sleep dual memory;
- first sleep-time parametric update or generated replay;
- first sleep-time pruning;
- first learned sleep scheduler or replay selector;
- first NREM/REM-inspired machine-learning objective.

The narrow working contribution is instead:

> Learn and evaluate a past-only lifecycle policy that routes the same
> versioned memory objects among reversible raw storage, semantic
> text/vector/graph views, reusable latent/KV artifacts, user-local parameters,
> shared parameters, and deletion under matched lifetime resource and risk
> constraints, while preserving source lineage, selective revocation, and
> rollback.

That hypothesis remains subject to the full evidence-registry novelty audit.
