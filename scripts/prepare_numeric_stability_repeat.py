"""Prepare (never launch) two isolated, single-Epoch6, two-window repeats."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--time-statistics", required=True)
    parser.add_argument("--shared-artifact-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--legacy-numerics", action="store_true", help="Disable deterministic mode for a separate control pair")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    if checkpoint.name != "model_epoch_6" or not (checkpoint / "state_dict.pkl").is_file():
        parser.error("Use the read-only model_epoch_6 directory with state_dict.pkl")
    statistics = Path(args.time_statistics).resolve(strict=True)
    shared = Path(args.shared_artifact_root).resolve(strict=True)
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    evaluation = config.setdefault("detection", {}).setdefault("evaluation", {})
    evaluation["checkpoint_policy"] = "all_checkpoints"
    evaluation["deterministic_inference"] = not args.legacy_numerics
    config.setdefault("pipeline", {})["mode"] = "detection_only"
    config.setdefault("logging", {})["wandb_mode"] = "disabled"
    config_path = root / "config.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    repo = Path(__file__).resolve().parents[1]
    commands = []
    for repeat in ("repeat_1", "repeat_2"):
        artifact = root / repeat
        metadata = artifact / "THEIA_E5/runs/orthrus/smoke/seed_0/metadata"
        metadata.mkdir(parents=True)
        shutil.copyfile(statistics, metadata / "time_statistics.json")
        commands.append([sys.executable, str(repo / "src/experiments/run_experiment.py"),
                         "--dataset", "THEIA_E5", "--config", str(config_path),
                         "--seed", "0", "--artifact-root", str(artifact),
                         "--shared-artifact-root", str(shared), "--shared-artifact-readonly",
                         "--checkpoint", str(checkpoint), "--stages", "test,evaluate",
                         "--max-windows-per-split", "2"])
    for split in ("val", "test"):
        suffix = f"THEIA_E5/runs/orthrus/smoke/seed_0/edge_scores/{split}/model_epoch_6"
        commands.append([sys.executable, str(repo / "scripts/check_numeric_stability.py"),
                         "--output", str(root / f"compare_{split}.json"), "compare",
                         "--left", str(root / "repeat_1" / suffix),
                         "--right", str(root / "repeat_2" / suffix)])
    # Set environment in the outer shell, before either child imports torch.
    script = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {shlex.quote(str(repo))}",
              "export CUDA_VISIBLE_DEVICES=1", "export CUBLAS_WORKSPACE_CONFIG=:4096:8",
              'test "$(git branch --show-current)" = fix/p4-numeric-stability',
              "mkdir " + shlex.quote(str(root / ".started")),
              "status_path=" + shlex.quote(str(root / "final_rc")),
              "trap 'rc=$?; printf \"%s\\n\" \"$rc\" > \"$status_path\"' EXIT"]
    script.extend(shlex.join(command) for command in commands)
    (root / "run.sh").write_text("\n".join(script) + "\n")
    (root / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    print(f"Prepared only: {root / 'run.sh'}; no inference launched")


if __name__ == "__main__":
    main()
