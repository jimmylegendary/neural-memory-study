# Google meeting research package

This package turns the HOPE / Sleep / Memory Caching / NSTM evidence base into a six-axis meeting flow with stable question identifiers and separate live and post-meeting capture.

## Reading and use order

1. Read the [six-axis evidence pre-read](01-SIX-AXIS-PROBLEM-MAP.md).
2. Facilitate from the [canonical KQ/BQ playbook](02-GOOGLE-KICK-QUESTION-PLAYBOOK.md).
3. Use the [device and system appendix](03-DEVICE-SOLUTION-BRIDGE.md) only for the branch selected by the answer.
4. Record KQ/BQ answers in the [live capture template](04-ANSWER-CAPTURE-LIVE.csv).
5. Normalize each KQ outcome in the [post-meeting template](05-ANSWER-CAPTURE-POSTMEETING.csv).
6. Keep non-repository dependencies in the [external-artifact register](EXTERNAL-ARTIFACTS.md).

Stable inventory: six `KQ` IDs (`KQ-01…06`), eight `BQ` IDs (`BQ-01…08`), and ten `DBQ` IDs (`DBQ-01…10`). Existing question content and these identifiers are canonical.

## Six-axis coverage

| Axis | Primary questions | Supporting questions |
|---|---|---|
| 1. Long-context understanding and test-time scaling | `KQ-01` | `BQ-03`, `BQ-08`, `DBQ-01`, `DBQ-09` |
| 2. Recurrent-state reuse and serving efficiency | `KQ-02`, `KQ-03` | `BQ-02`, `BQ-06`, `DBQ-02`, `DBQ-04`, `DBQ-10` |
| 3. RNN/TTT pre-training, prefill, and decode efficiency | `KQ-02`, `KQ-03` | `BQ-01`, `BQ-05`, `BQ-08`, `DBQ-03`, `DBQ-07`, `DBQ-10` |
| 4. Long-term memory and continual learning | `KQ-01`, `KQ-04`, `KQ-05` | `BQ-03`, `BQ-04`, `BQ-06`, `BQ-07`, `BQ-08`, `DBQ-04`, `DBQ-05`, `DBQ-08`, `DBQ-09` |
| 5. Monotonic memory growth under fixed capacity | `KQ-04`, `KQ-05` | `BQ-01`, `BQ-02`, `BQ-04`, `BQ-06`, `DBQ-01`, `DBQ-05`, `DBQ-06`, `DBQ-08`, `DBQ-09`, `DBQ-10` |
| 6. Sleep-time compute as a scaling axis | `KQ-05`, `KQ-06` | `BQ-01`, `BQ-07`, `BQ-08`, `DBQ-03`, `DBQ-05`, `DBQ-07`, `DBQ-08`, `DBQ-09`, `DBQ-10` |

## Paper-version provenance

| Evidence ID | Repository evidence | Version rule |
|---|---|---|
| HOPE / Nested Learning | [2512.24695 extraction](../../papers/2512.24695.txt) | Repository filename carries no arXiv version suffix; do not infer one. |
| Language Models Need Sleep | [2606.03979v2 extraction](../../papers/2606.03979v2.txt) | Canonical claim and line evidence is arXiv v2 dated 2026-07-10; the v1 Korean translation is reference-only. |
| Memory Caching | [2602.24281 extraction](../../papers/2602.24281.txt) | Repository filename carries no arXiv version suffix; do not infer one. |
| NSTM | [2607.15271 extraction](../../papers/2607.15271.txt) | Repository filename carries no arXiv version suffix; do not infer one. |
| TTT supporting boundary | [2407.04620 extraction](../../papers/external/2407.04620.txt) | Supporting systems boundary, not one of the four primary meeting papers. |

## Quantitative artifacts

- [Native live-Sheet archive](../attn-vs-hope/README.md): the user-reviewed operational model, preserved as one tab.
- [E5 analytical engine](../../experiments/E5-attn-vs-hope/README.md): deterministic equations, scenarios, results, and tests.
- [Generated 11-tab workbook](../../experiments/E5-attn-vs-hope/sheet/Attn-vs-HOPE.xlsx): reproducible analytical companion, distinct from the native archive.
- [Approved analytical design](../../docs/superpowers/specs/2026-07-27-attn-vs-hope-analytical-model-design.md).
