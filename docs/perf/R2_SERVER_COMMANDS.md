# R2 server profiling commands

These commands run only the bounded R2 baseline. They do not perform R3+
optimization. Run them from the reviewed perf worktree containing the uncommitted
R1 profiler diff.

## 1. Preflight

```bash
set -euo pipefail

EXPECTED_SHA=40ee2404a126dee600135e9b49650ed42aee8aa6
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
git submodule update --init --recursive Ground_Truth/darpa

export ORTHRUS_SHARED_ARTIFACT_ROOT=/path/to/mstc-preprocessing-artifacts
export ORTHRUS_R2_OUTPUT_ROOT=/path/to/new/r2-profile-output

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
print("compiled_arches", torch.cuda.get_arch_list())
x = torch.ones(1, device="cuda")
print("cuda_kernel_smoke", float((x + 1).cpu()))
PY
```

Do not continue if the CUDA kernel smoke fails or if the shared root does not
contain the exact paths resolved by the checked-in configuration.

Read-only artifact preflight for both datasets:

```bash
python - <<'PY'
import sys
import os
from pathlib import Path
sys.path.insert(0, "src")
from experiments.run_experiment import build_parser, build_pipeline_args
from config import get_yml_cfg

for dataset in ("THEIA_E3", "THEIA_E5"):
    ns = build_parser().parse_args([
        "--dataset", dataset,
        "--config", "config/experiments/mstc_full.yml",
        "--seed", "0",
        "--artifact-root", "/tmp/r2-path-check",
        "--shared-artifact-root", os.environ["ORTHRUS_SHARED_ARTIFACT_ROOT"],
        "--max-windows-per-split", "2",
    ])
    cfg = get_yml_cfg(build_pipeline_args(ns))
    required = {
        "graphs": cfg.graph_construction.build_graphs._graphs_dir,
        "word2vec": cfg.edge_featurization.embed_nodes.feature_word2vec._model_dir,
        "edge_embeddings": cfg.edge_featurization.embed_edges._edge_embeds_dir,
    }
    print(dataset)
    for name, path in required.items():
        p = Path(path)
        print(name, p, "OK" if p.is_dir() and any(p.iterdir()) else "MISSING")
PY
```

## 2. Bounded E3 and E5 profiles

Each window count gets an isolated output root. The shared root is read-only
input; no preprocessing is requested.

```bash
for dataset in THEIA_E3 THEIA_E5; do
  for windows in 2 10 20; do
    python src/experiments/run_experiment.py \
      --dataset "$dataset" \
      --config config/experiments/mstc_full.yml \
      --seed 0 \
      --stages train,test,evaluate \
      --artifact-root "$ORTHRUS_R2_OUTPUT_ROOT/${dataset}/w${windows}" \
      --shared-artifact-root "$ORTHRUS_SHARED_ARTIFACT_ROOT" \
      --max-windows-per-split "$windows" \
      --perf-profile \
      --perf-profile-sample-every-n-batches 10
  done
done
```

For every successful run, expect these files below its run directory:

```text
profiling/perf_profile.json
profiling/perf_profile.csv
profiling/perf_summary.json
```

The JSON/CSV report stage totals, sampled batch hotspots, peak GPU allocation,
peak CPU RSS, process I/O deltas, and event/batch/window throughput.

## 3. Correctness and overhead controls

Run the 2-window case once more with profiling disabled and a fresh output root.
Compare the resolved config, selected epoch, validation objective, raw event
scores, node scores, and metrics against the profiled run. Exact score artifacts
must match; the profiler is observational only.

```bash
python src/experiments/run_experiment.py \
  --dataset THEIA_E3 \
  --config config/experiments/mstc_full.yml \
  --seed 0 \
  --stages train,test,evaluate \
  --artifact-root "$ORTHRUS_R2_OUTPUT_ROOT/THEIA_E3/w2-profiler-off" \
  --shared-artifact-root "$ORTHRUS_SHARED_ARTIFACT_ROOT" \
  --max-windows-per-split 2
```

Measure profiler overhead only from repeated, isolated profiler-on/off runs on
the same idle server and report the median wall time. Do not infer overhead from
this development container or compare runs that use different input hashes.

