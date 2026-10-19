"""Reproduce the bounded Windows acceptance suite with commands and exit codes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "affected": """
        runtime_profiler bounded_full_data_loader loader_telemetry_persistence
        dataset_views compact_history_io sparse_node_sidecar sidecar_cache_identity
        preprocess_memory_streaming environment_dump run_metadata e2e_smoke_synthetic
        val_test_data_flow checkpoint_cuda_lifecycle memory_metrics artifact_names
    """.split(),
    "mstc": """
        numeric_stability checkpoint_policy multiscale_encoder multiscale_sampler
        multiscale_pipeline_integration backbones time_gap time_gap_causal_fit
        time_gap_decoder time_statistics_lifecycle calibration calibration_runner
        aggregation node_threshold mstc_node_evaluation c6_evaluation_integration
        test_replay_protocol no_future_leakage
    """.split(),
    "io": """
        c8_2_preprocess_correctness c8_3b_empty_day_semantics embed_edges_temporal_order
        pytorch26_serialization_compat mstc_evaluation_calibration_adapter
        evaluation_artifacts mstc_canonical_metrics epoch_selection word2vec_runtime_compat
    """.split(),
    "orchestration": """
        runtime_benchmark run_experiment run_from_training_no_nameerror
        final_temporal_semantics global_event_index checkpoint_resume
    """.split(),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("Run this acceptance suite on native Windows")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(WANDB_MODE="disabled", PYTHONUTF8="1")
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["NLTK_DATA"] = str(ROOT / "artifacts/nltk_data") + os.pathsep + env.get("NLTK_DATA", "")
    manifest = {
        "utc": datetime.now(timezone.utc).isoformat(), "cwd": str(ROOT),
        "python": sys.executable,
        "environment": {key: env[key] for key in ["WANDB_MODE", "PYTHONUTF8", "PYTHONPATH", "NLTK_DATA"]},
        "commands": [],
        "excluded_evidence": ["Native Linux execution", "Real THEIA 2-window GPU experiment"],
    }

    def run(name, command):
        log = output / f"{name}.log"
        with log.open("wb") as handle:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        manifest["commands"].append({"name": name, "argv": command,
                                     "exit_code": completed.returncode, "log": str(log)})
        (output / "commands.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"{name}: exit_code={completed.returncode}; {log}", flush=True)

    for name, files in GROUPS.items():
        run(name, [sys.executable, "-m", "pytest", "-q",
                   *[f"tests/test_{file}.py" for file in files],
                   f"--junitxml={output / (name + '.xml')}"])
    run("native-smoke", [sys.executable, "scripts/check_windows_native.py",
                         "--output", str(output / "native-smoke.json")])
    run("preprocess-benchmark", [sys.executable, "scripts/benchmark_preprocess_memory.py",
                                 "--events", "4096", "--fetch-size", "128"])
    run("syntax", [sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"])
    run("pip-check", [sys.executable, "-m", "pip", "check"])
    run("pip-freeze", [sys.executable, "-m", "pip", "freeze"])
    run("diff-check", ["git", "diff", "--check"])
    return 0 if all(row["exit_code"] == 0 for row in manifest["commands"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
