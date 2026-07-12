# Veridraft claim-gating readiness — neural-memory-study

- Bundle: `claims/bundle.json`  (bundle_id `neural-memory-study-2026`)
- Profile: `systems-paper`  (`veridraft.config.json`)
- Warrant-anchor commit: `59fc1ffb381677d04060d008e32b0b55ea995309`
- Generated: 2026-07-12T04:01:14Z by `python3 -m veridraft gate`

## Summary

| claim class | type | count | evidence kind |
|---|---|---|---|
| Faithful-summary of the 6 source papers | P2 | 18 | `source_artifact` `papers/<id>.pdf@<commit>` |
| Original quantitative (cost-model + CPU micro-bench) | P1 | 12 | `source_artifact` `experiments/results.json@<commit>` + `result_refs` verbatim metrics |
| **Total** |  | **30** |  |

**Evidence gate: PASSED (exit 0). blocked: (none). All 30 claims admissible; `require_results` satisfied (every P1 references a result block with verbatim metrics).**

## `veridraft gate neural-memory-study-2026`
```
gate profile: systems-paper  bundle: neural-memory-study-2026
  [PASS] p1_batch_bw_bound (P1)  admissible_evidence=1
  [PASS] p1_chunk_roofline_axis (P1)  admissible_evidence=1
  [PASS] p1_frequency_tier_win (P1)  admissible_evidence=1
  [PASS] p1_kv_ttt_crossover (P1)  admissible_evidence=1
  [PASS] p1_kv_ttt_write_asymmetry (P1)  admissible_evidence=1
  [PASS] p1_kvmanager_gap (P1)  admissible_evidence=1
  [PASS] p1_memory_bound (P1)  admissible_evidence=1
  [PASS] p1_residency_crossover (P1)  admissible_evidence=1
  [PASS] p1_retention_capacity_rank (P1)  admissible_evidence=1
  [PASS] p1_rmw_bandwidth_cliff (P1)  admissible_evidence=1
  [PASS] p1_rmw_per_token (P1)  admissible_evidence=1
  [PASS] p1_scratchpad_resident_win (P1)  admissible_evidence=1
  [PASS] p2_atlas_1 (P2)  admissible_evidence=1
  [PASS] p2_atlas_2 (P2)  admissible_evidence=1
  [PASS] p2_atlas_3 (P2)  admissible_evidence=1
  [PASS] p2_miras_1 (P2)  admissible_evidence=1
  [PASS] p2_miras_2 (P2)  admissible_evidence=1
  [PASS] p2_miras_3 (P2)  admissible_evidence=1
  [PASS] p2_nested_1 (P2)  admissible_evidence=1
  [PASS] p2_nested_2 (P2)  admissible_evidence=1
  [PASS] p2_nested_3 (P2)  admissible_evidence=1
  [PASS] p2_sleep_1 (P2)  admissible_evidence=1
  [PASS] p2_sleep_2 (P2)  admissible_evidence=1
  [PASS] p2_sleep_3 (P2)  admissible_evidence=1
  [PASS] p2_titans_1 (P2)  admissible_evidence=1
  [PASS] p2_titans_2 (P2)  admissible_evidence=1
  [PASS] p2_titans_3 (P2)  admissible_evidence=1
  [PASS] p2_tnt_1 (P2)  admissible_evidence=1
  [PASS] p2_tnt_2 (P2)  admissible_evidence=1
  [PASS] p2_tnt_3 (P2)  admissible_evidence=1
passed: ['p1_batch_bw_bound', 'p1_chunk_roofline_axis', 'p1_frequency_tier_win', 'p1_kv_ttt_crossover', 'p1_kv_ttt_write_asymmetry', 'p1_kvmanager_gap', 'p1_memory_bound', 'p1_residency_crossover', 'p1_retention_capacity_rank', 'p1_rmw_bandwidth_cliff', 'p1_rmw_per_token', 'p1_scratchpad_resident_win', 'p2_atlas_1', 'p2_atlas_2', 'p2_atlas_3', 'p2_miras_1', 'p2_miras_2', 'p2_miras_3', 'p2_nested_1', 'p2_nested_2', 'p2_nested_3', 'p2_sleep_1', 'p2_sleep_2', 'p2_sleep_3', 'p2_titans_1', 'p2_titans_2', 'p2_titans_3', 'p2_tnt_1', 'p2_tnt_2', 'p2_tnt_3']
blocked: (none)
```

## `veridraft readiness --venue mlsys`
```
venue=mlsys  policy=DISCLOSURE  disclosure_required=True
  disclosure (methods):
    During the preparation of this work the authors used an AI writing assistant (PaperOrchestra, wrapped by Veridraft) to draft and refine the manuscript from evidence-gated claims. All content was reviewed and edited by the authors, who take full responsibility for the accuracy and integrity of the work.
  - Disclose LLM use in the methods; add the generated disclosure text.
  - No local detector configured. A low detector score is a RISK proxy, not proof of human authorship; detectors false-flag human (esp. non-native) writing. Run a self-hosted OSS detector (e.g. Binoculars) locally for a RISK band; never send a confidential manuscript to a cloud detector.
```

## `veridraft status`
```
  art-neural-memory-study-2026  state=gated            track=public-source-assisted (b=public,v=team)  output=None
```

## Honesty-contract notes

- Every P1 metric is transcribed verbatim from a git-tracked, clean-at-HEAD `results.json`; caveat tags (REL / NO-WALLCLOCK / NOVEL-SIM-FALSE / CPU-SHAPE / M-ASSUMED / directional-ordinal) are carried inside each claim statement per D4 (exploration-grade: ratios, crossover positions, tier orderings, bound classes are load-bearing — not silicon absolutes).
- P2 faithful-summary claims carry no numbers; the gate checks ref RESOLVABILITY (path@commit), the discipline checks each statement against the single cited PDF.
- The retention rank claim points at `experiments/E4-scaling/results.json` (where the Spearman rho lives); all other P1 refs point at the consolidated `experiments/results.json`.
- `digest_ok=False` at import is expected: the bundle carries no signed digest field; it is not a gate condition.
