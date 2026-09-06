# MAGIC M8 Progress: Synthetic End-to-End Smoke Tests

**Status**: M8 local synthetic smoke implementation complete.

**Date**: 2026-09-06

---

## Overview

M8 implements bounded CPU synthetic end-to-end smoke tests for the MAGIC baseline. This is NOT MAGIC end-to-end reproduction, real backend verification, or E3/E5 verification.

---

## What M8 Does

### Synthetic Fixture

- Generates synthetic canonical records for train/validation/test splits
- Train: >= 50 nodes supporting k=10 scoring
- Multiple node types: process/subject, file, netflow/socket, pipe, ipc
- Multiple edge types: EXEC, WRITE, READ, RECV, LOAD, CONNECT, OPEN
- READ/RECV/LOAD edges for causal direction reversal testing
- Same timestamp with different global_event_index
- Validation contains train-unseen types (pipe, ipc)
- Test contains anomalous nodes (far from train cluster)
- Deterministic embeddings based on node_id and seed

### CPU E2E Path

Full synthetic E2E data flow:
```
synthetic canonical records
    ↓
M3 MAGICInputAdapter
    ↓
M4 causal snapshots + validation_quantile(q=0.999)
    ↓
synthetic backend (synthetic_smoke, NOT magic_upstream)
    ↓
M5 MAGICEntityScorer (k=10)
    ↓
validation raw scores
    ↓
M6 validation_quantile(q=0.999) → frozen threshold
    ↓
test raw scores
    ↓
predictions
    ↓
test labels (only here)
    ↓
canonical metrics
    ↓
synthetic canonical artifacts
```

### Leakage Tests

1. **Ground-truth leakage**: Changing labels does not affect scores/predictions
2. **Future leakage**: Snapshot at T1 excludes events at T2 > T1
3. **Max node merge**: final_score(node) = max over all valid snapshot scores

### Seed Reproducibility

- seed=0 produces identical checksums across runs
- seed=1 produces different traces than seed=0
- Dataset/split semantics remain stable across seeds

### Artifact Contract

Outputs to `<output_dir>/SYNTHETIC_MAGIC/runs/magic/smoke_seed<N>/`:
- `environment.json` (is_smoke: true, backend: synthetic_smoke)
- `config_resolved.yml` (is_smoke: true, backend: synthetic_smoke)
- `runtime.json` (is_smoke: true, backend: synthetic_smoke)
- `raw_magic/graph_manifest.json`
- `raw_magic/local_global_node_map.csv`
- `raw_magic/validation_node_window_scores.csv` (NO y_true)
- `raw_magic/test_node_window_scores.csv` (NO y_true)
- `raw_magic/knn_reference_manifest.json`
- `raw_magic/threshold.json` (provenance: validation_only)
- `node_scores/node_predictions.csv`
- `node_scores/metrics.json`

### Cache Identity

Config fingerprint includes:
- seed
- fixture_checksum
- k
- q
- upstream_sha
- backend

Same fingerprint → safe cache reuse. Different fingerprint → must regenerate.

### Collector Exclusion

`collect_results.py` with `include_smoke=False` (default) skips all `is_smoke=true` runs. Explicit `--include-smoke` flag required for debugging.

---

## What M8 Does NOT Do

- ❌ Real THEIA_E3 data
- ❌ Real THEIA_E5 data
- ❌ MAGIC real training
- ❌ Official paper results
- ❌ Downloading datasets
- ❌ Auto-installing dependencies
- ❌ Installing DGL/Torch 1.12
- ❌ Building Docker
- ❌ GAT/GMAE backend
- ❌ M9 pilot
- ❌ Modifying /opt/magic-upstream

---

## Boundaries

### Backend Identity

```
backend = synthetic_smoke
```

NOT:
```
backend = magic_upstream
backend = gnn_real
```

### Dataset Identity

```
dataset = SYNTHETIC_MAGIC
```

NOT:
```
dataset = THEIA_E3
dataset = THEIA_E5
```

---

## Test Coverage (29 tests)

1. ✅ synthetic fixture validity
2. ✅ enough nodes for k=10
3. ✅ full train→val→test synthetic E2E
4. ✅ no ground-truth leakage
5. ✅ no future leakage
6. ✅ max node merge
7. ✅ q=0.999 threshold
8. ✅ all canonical test nodes retained
9. ✅ seed=0 repeated checksum identical
10. ✅ seed=1 stochastic trace differs
11. ✅ artifact schema
12. ✅ raw artifact has no y_true
13. ✅ final prediction may contain y_true
14. ✅ threshold provenance
15. ✅ is_smoke=true
16. ✅ dataset=SYNTHETIC_MAGIC
17. ✅ backend=synthetic_smoke
18. ✅ cache same identity reusable
19. ✅ seed change invalidates cache
20. ✅ fixture change invalidates cache
21. ✅ config change invalidates cache
22. ✅ upstream SHA change invalidates cache
23. ✅ collector default excludes smoke
24. ✅ explicit debug include smoke if implemented
25. ✅ smoke does not require DGL
26. ✅ smoke does not require CUDA
27. ✅ smoke does not require PostgreSQL
28. ✅ smoke does not access network
29. ✅ M3-M7 regressions remain green

---

## Files Added/Modified

### New Files

- `tests/test_magic/fixtures/__init__.py` - Fixture package init
- `tests/test_magic/fixtures/synthetic_fixture.py` - Synthetic fixture generator
- `src/baselines/magic/smoke.py` - Smoke harness
- `src/baselines/magic/run_magic_smoke.py` - CLI entry point
- `config/smoke/magic.yml` - Smoke config
- `docs/MAGIC_M8_PROGRESS.md` - This document

### Modified Files

- None (M8 only adds new files, does not modify existing M3-M7)

---

## Running M8 Tests

### Run smoke tests only
```bash
PYTHONPATH=/home pytest -q tests/test_magic/test_magic_smoke.py
```

### Run all MAGIC tests
```bash
PYTHONPATH=/home pytest -q tests/test_magic
```

### Run CLI smoke
```bash
python src/baselines/magic/run_magic_smoke.py --seed 0 --seed 1
```

---

## Upstream Integrity

SHA256 of /opt/magic-upstream files (unchanged):
- train.py: `9003a2ec19632073d55a1d689b828272bc53496685175fa4c88dbb63fdf028ad`
- eval.py: `299352ea0387237fd0fcefe806024e1ecf18bd0d82ebdbdf12d31b4fbeed2208`
- model/eval.py: `4761d6481bd29ddc2789eebf2d4dbb04a76176f23223e0690a1f0a0f5f2d75dc`
- utils/loaddata.py: `57a84d42d6fa3f13a98e7ef606e9c3087574868bea3d70d62adaeca2bf155924`
- utils/utils.py: `9e350ba709815437434c27aa4983e28785b4ab673ea3bf75dff6094984757fec`

---

## Pending Work

### M9: Full Backend Integration
- Real GAT/GMAE backend integration
- Real THEIA_E3/E5 data
- Full paper matrix

### M10: Docker/CI Integration
- Docker build for reproducible environment
- CI pipeline for automated testing

---

## Accurate Status Statement

> M8 local synthetic smoke implementation complete.

NOT:
- ❌ MAGIC end-to-end reproduction complete
- ❌ MAGIC real backend verified
- ❌ MAGIC E3/E5 verified
- ❌ MAGIC training complete
