# R1 diff audit

Audit date: 2026-10-01 UTC.
Auditor: R1 transfer gate (this document is the source of truth for the staging commit).

## Identity

- `BASE_SHA`: `40ee2404a126dee600135e9b49650ed42aee8aa6`
- `perf_branch`: `perf/mstc-e3-e5-runtime-v1`
- Base ancestry verified: `git merge-base --is-ancestor 40ee2404... HEAD` → PASS.

## Modified files (tracked)

| Path | Classification | Purpose |
|---|---|---|
| `src/config.py` | B. profiler CLI plumbing | Adds opt-in `--perf-profile` (store_true, default off) and `--perf-profile-sample-every-n-batches` (int, default `None`); records them on the yacs cfg as `_perf_profile` (default `False`) and `_perf_profile_sample_every_n_batches` (default `None`). |
| `src/experiments/run_experiment.py` | B. profiler CLI plumbing | Forwards the two new CLI flags through `build_parser` and `build_pipeline_args`. |
| `src/orthrus.py` | A. profiler instrumentation | Calls `configure_profiler(...)` from `run(args)` based on CLI flag and env var, wraps train/test/evaluate stage bodies in `profiler.timer(...)`, and ensures `finish_profiler(...)` runs in `finally`. |
| `src/data_utils.py` | A. profiler instrumentation | Wraps `_prepare_window`, `LazyTemporalWindowCollection._load`, sidecar restore, source-metadata scan, compact-index build, and `BoundedFullData` construction in timer blocks. Adds `loading.windows` counter. No change to load order or RNG. |
| `src/detection/orthrus_gnn_training.py` | A. profiler instrumentation | Uses `profiler.profile_batches(batch_loader)` and per-stage `cuda_timer` + `timer` wrappers around forward / loss / backward / optimizer / logging. Adds `training.batches`, `training.events`, `training.windows` counters. H2D transfer wrapped in `training.T_h2d` + `training.T_h2d_cuda`. |
| `src/detection/orthrus_gnn_testing.py` | A. profiler instrumentation | Wraps `test(...)` batch loop (counters for batches/events/windows), CSV export (`evaluation.T_export`), model construction (`loading.T_model_construction`), reset state (`evaluation.T_reset_state`), replay train (`evaluation.T_replay_train`), per-split timing (`evaluation.T_val` / `evaluation.T_test`). |
| `src/model.py` | A. profiler instrumentation | Adds `training.T_timegap_target` and `training.T_loss` detail-timing records inside `MSTCOrthrus.forward`. |
| `src/mstc/multiscale_encoder.py` | A. profiler instrumentation | Wraps `_historical_node_features`, `_edge_features`, `neighbor_loader` call, and `neighbor_loader.insert` in per-stage detail timers (all parented to `training.T_total_batch`). Edge-feature call is hoisted into a local variable solely to give the timer a stable scope; the same call/arg order is preserved. |
| `src/mstc/multiscale_sampler.py` | A. profiler instrumentation | Wraps `history_store.query(...)` in `training.T_history_query` detail timer. Has a `try/except ModuleNotFoundError` import so tests importing it as `src.mstc.*` still resolve `get_profiler`. |
| `src/mstc/node_evaluation.py` | A. profiler instrumentation | Wraps `aggregator.aggregate_events(...)` (validation and test) in `evaluation.T_node_aggregation`; wraps `compute_node_threshold(...)` in `evaluation.T_threshold`. |
| `src/mstc/evaluation_runner.py` | A. profiler instrumentation | Wraps `calibration_module.run_calibration(...)` in `evaluation.T_calibration`; wraps `node_predictions.csv` write in `evaluation.T_export`. |

## New files

| Path | Classification | Purpose |
|---|---|---|
| `src/runtime/__init__.py` | A. profiler instrumentation | Package marker; no model or experiment semantics. |
| `src/runtime/runtime_profiler.py` | A. profiler instrumentation | The opt-in profiler itself. Disabled by default. Exports `configure_profiler`, `get_profiler`, `finish_profiler`, `profile_requested`, `sample_every_from`. Writes JSON + CSV + summary artifacts only when `enabled=True` and `run_root` is provided; on flush it performs one `torch.cuda.synchronize` to resolve queued CUDA events (only relevant when enabled). |
| `tests/test_runtime_profiler.py` | C. tests | 6 unit tests covering: disabled profiler is artifact-free, enabled profiler writes expected files, totals survive raw-sample truncation, deterministic batch sampling, `profile_batches` times real retrieval and preserves order, env-var controls. |
| `docs/perf/R0_BASELINE_AUDIT.md` | D. docs | (already present) R0 baseline audit. |
| `docs/perf/R2_BASELINE_PROFILE_REPORT.md` | D. docs | (already present) R2 baseline profile report. |
| `docs/perf/R2_SERVER_COMMANDS.md` | D. docs | (already present) R2 server commands. Updated/confirmed this round (see Phase 11 below). |
| `docs/perf/R1_DIFF_AUDIT.md` | D. docs | This document. |

## Files explicitly NOT modified

- `Ground_Truth/` (submodule, untouched)
- `/home/**` (protected)
- Any YAML in `config/`
- Any checkpoint / artifact / cache / `__pycache__` / embedding

## Semantic audit (algorithm/config/loss/temporal/eval)

- `batch_size`: unchanged. No diff in `src/config.py`, `src/data_utils.py`, or any training entrypoint that would alter it.
- `epochs`: unchanged. `num_epochs = cfg.detection.gnn_training.num_epochs` is untouched.
- `optimizer`: unchanged. `optimizer_factory(...)` is called with the same arguments; only wrapped in a profiler timer.
- `learning rate`: unchanged. No optimizer config edits.
- `neighbor budgets`: unchanged. `MultiScaleNeighborLoader(...)` and `_encode_scale(...)` keep their arguments.
- `share_encoder`: unchanged. No edit to `shared_graph_encoder` wiring.
- `fusion`: unchanged. No edit to fusion logic.
- `scale embedding`: unchanged. `self.scale_embedding(scale_ids)` is called in the same place with the same input.
- `lambda_time`: unchanged. The line `score_raw = loss_type + self.lambda_time * loss_time` is preserved verbatim in `src/model.py`.
- `loss_time = 0.5 * (loss_time_src + loss_time_dst)`: unchanged. Verified inside `MSTCOrthrus.forward`; only surrounded by a profiler detail-time record that is no-op when the profiler is disabled or detail sampling is inactive.
- `score_raw = loss_type + 0.3 * loss_time`: unchanged. Same line, same coefficient.
- `event ordering`, `HistoryStore` semantics: unchanged. `history_store.query(...)` and `neighbor_loader.insert(...)` are called with the same arguments in the same order; the only change is wrapping them in detail-timer context managers (no-op when disabled).
- `reset → replay train → validation → no reset → test`: unchanged. The four logical phases are preserved in `orthrus_gnn_testing.main`; only instrumentation is added (timers around `reset_state`, `_replay_train_history`, val/test loops).
- `calibration`: unchanged. `calibration_module.run_calibration(...)` arguments are identical.
- `Top-K`, `threshold`, `model selection`, `train/val/test split`: unchanged. No edit to `compute_node_threshold`, `apply_node_threshold`, `aggregator.aggregate_events`, epoch selection, or split construction.
- Test-based model selection: NOT introduced. `test_epoch_selection.py` and `test_epoch_selection_lightweight.py` (and `test_test_replay_protocol.py`) continue to pass; the guarded legacy test-selection path still raises.

### Summary flags

| Field | Value |
|---|---|
| `ALGORITHM_FILES_CHANGED` | NO (all edits are wrapper/context-manager additions) |
| `CONFIG_CHANGED` | NO (no YAML touched; only the yacs `_perf_profile` and `_perf_profile_sample_every_n_batches` fields are added, both opt-in) |
| `DATA_SPLIT_CHANGED` | NO |
| `LOSS_SEMANTICS_CHANGED` | NO |
| `TEMPORAL_SEMANTICS_CHANGED` | NO |
| `EVALUATION_SEMANTICS_CHANGED` | NO |
| `PROFILER_DEFAULT_DISABLED` | YES (default `_ACTIVE_PROFILER = RuntimeProfiler(enabled=False)`; CLI flag is `store_true`; `profile_requested(False)` is False when env var unset) |
| `R1_RUNTIME_SEMANTICS_CHANGED` | **NO** |

## Profiler default-disabled evidence

- `src/runtime/runtime_profiler.py` defines `_ACTIVE_PROFILER = RuntimeProfiler(enabled=False)` (module-level singleton, line 409).
- `RuntimeProfiler.enabled` is `bool(enabled)`; `add_counter`, `record`, `timer`, `cuda_timer`, `profile_batches`, `flush` all early-return when `self.enabled` is False.
- `flush()` writes no artifacts when `enabled=False` or `run_root is None`.
- CLI: `parser.add_argument('--perf-profile', action='store_true', ...)` and `--perf-profile-sample-every-n-batches` defaults to `None`.
- yacs cfg: `_perf_profile` defaults to `False`; `_perf_profile_sample_every_n_batches` defaults to `None`.
- `configure_profiler(...)` is only called from `run(args)` after `profile_requested(getattr(args, 'perf_profile', False))` returns True.
- Live verification (default state, no CLI flag, env unset):
  - `default_enabled=False`
  - `default_detail_active=False`
  - `default_run_root=None`
  - `profile_requested(False)` returns `False`
- Default execution path:
  - No forced `torch.cuda.synchronize` (synchronize only runs inside `_resolve_cuda()`, which only runs inside `flush()`, which only writes files when `enabled=True`).
  - No large artifact files created (no `profiling/` dir, no JSON/CSV/summary written).
  - Batch order: `profile_batches` yields batches in the same order with a `None` token when disabled, preserving order.
  - RNG order: no `torch.manual_seed`, no generator reorder, no `torch.use_deterministic_algorithms` calls.
  - HistoryStore mutation order: every `history_store.query(...)` and `neighbor_loader.insert(...)` call site is wrapped in a no-op context manager when disabled.
  - Loss computation: unchanged.
  - Optimizer step: unchanged (wrapped in `with profiler.timer(...), profiler.cuda_timer(...)` whose `__enter__` is a no-op when disabled).

## Test results

- Targeted tests (`tests/test_runtime_profiler.py`): 6 passed / 0 failed.
- Related tests (CLI, synthetic E2E, frozen temporal semantics, history state, time-gap, model selection, replay protocol, multiscale encoder/sampler, MSTC canonical metrics, MSTC node evaluation, MSTC evaluation/calibration adapter, MSTC consumer lookup, calibration, node threshold, aggregation, dataset views, metrics, pipeline stages, run metadata): 422 passed / 0 failed.
- Full pytest (`pytest tests/`): **1072 passed, 4 failed**.
- Failed test names (all due to uninitialized `Ground_Truth/darpa` submodule):
  - `tests/test_artifact_root_routing.py::TestDualRootEndToEnd::test_matrix_dual_root_routing_end_to_end`
  - `tests/test_artifact_root_routing.py::TestDualRootEndToEnd::test_standalone_run_experiment_uses_artifact_root_for_both`
  - `tests/test_artifact_root_routing.py::TestDualRootEndToEnd::test_preprocessing_paths_use_shared_root_not_scoped`
  - `tests/test_c8_2_preprocess_correctness.py::test_corpus_scope_changes_only_semantic_task_hash_chain`
- All four failures raise `FileNotFoundError` from `src/config.py:71` complaining that the `Ground_Truth/darpa` submodule is not initialized.
- The required submodule commit `012f321f46137650496e639b0ad7e0a66db07a73` is NOT present in the local git object store (`git cat-file -t 012f321...` returns "could not get object info"), so the submodule cannot be initialized without network access. Per the R1 transfer rules, we keep this as BLOCKED and the server must re-run the full pytest gate after init.
- Skipped: 0.
- `FULL_TEST_GATE = BLOCKED_BY_GROUND_TRUTH_SUBMODULE` (NOT a code failure).

## Conclusion

- `R1_DIFF_AUDIT_GATE = PASS`
- `R1_RUNTIME_SEMANTICS_CHANGED = NO`
- Profiler is observational-only; runtime semantics are preserved.
- Staging commit authorized (see companion output and bundle).
