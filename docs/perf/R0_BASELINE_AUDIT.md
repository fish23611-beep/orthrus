# R0 baseline audit

Audit date: 2026-10-01 UTC.

## Baseline identity

- Expected commit: `40ee2404a126dee600135e9b49650ed42aee8aa6`
- Authorized source ref: `refs/heads/fix/mstc-timegap-cuda-runtime`
- Source-ref SHA observed before fetch: `40ee2404a126dee600135e9b49650ed42aee8aa6`
- Fetched object type: `commit`
- Isolated worktree: `/tmp/orthrus-mstc-perf-r0r2`
- Local branch: `perf/mstc-e3-e5-runtime-v1`
- Historical branch `freeze/e5-k1-external-validation`: absent on origin; this is a provenance difference, not a content mismatch.

The `/home` checkout was not used as the code baseline. Its branch and pre-existing untracked files were preserved.

## Effective frozen configuration

The effective configuration was audited after merging `config/orthrus.yml` with
`config/experiments/mstc_full.yml`, rather than by reading one YAML fragment in
isolation.

| Field | Effective value |
|---|---|
| `experiment_identity.semantics_version` | `temporal_v2` |
| `model_selection.method` | `min_val_objective` |
| batch size | `512` |
| neighbor budgets | `[8, 8, 8]` |
| shared encoder | `true` |
| fusion | `gated` |
| scale embedding | `true` |
| time-loss coefficient | `0.3` |
| calibration | `hierarchical_relation` |
| node aggregation | `topk_mean`, K=`5`, include destination=`true` |
| threshold | `validation_quantile`, quantile=`0.999` |
| semantic corpus scope | `train_only` |

## Frozen semantic audit

- TimeGap loss is `0.5 * (loss_time_src + loss_time_dst)`.
- Joint anomaly score is `loss_type + lambda_time * loss_time`; effective `lambda_time` is `0.3`.
- Formal model selection minimizes the mean validation `score_raw`; the guarded legacy test-selection path raises instead of selecting by test MCC.
- Canonical checkpoint evaluation remains `load → reset → replay train → validation → test`, with no reset between validation and test.
- Word2Vec training-node selection honors `train_only`.
- Formal node aggregation remains Top-K mean with K=5; no K=1 post-processing setting was folded into this baseline.
- Dataset splits match the checked-in THEIA E3/E5 contracts.

All R0 baseline, configuration, semantic, isolation, and `/home` safety gates passed before R1 edits began.
