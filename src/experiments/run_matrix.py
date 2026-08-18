"""Deterministic scheduler for ORTHRUS dataset × config × seed experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from artifact_paths import resolve_artifact_root
from experiments import run_experiment
from mstc.experiment_utils import _atomic_json
_ORIGINAL_RUN_EXPERIMENT_MAIN = run_experiment.main


_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()



def _read_json(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=SRC_ROOT.parent,
            text=True, capture_output=True, check=False, timeout=10,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _parse_csv(value: str, *, label: str, convert: Callable[[str], object] = str) -> list:
    raw_items = value.split(",")
    if not raw_items or any(not item.strip() for item in raw_items):
        raise ValueError(f"--{label} must be a comma-separated list without empty items")
    items = []
    seen = set()
    for raw in raw_items:
        item = convert(raw.strip())
        if item not in seen:
            seen.add(item)
            items.append(item)
    return items


def parse_datasets(value: str) -> list[str]:
    """Parse ordered datasets, de-duplicating repeated values by first occurrence."""
    return _parse_csv(value, label="datasets")


def parse_seeds(value: str) -> list[int]:
    """Parse ordered integer seeds, de-duplicating repeated values by first occurrence."""
    def as_int(item: str) -> int:
        try:
            return int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid seed {item!r}; seeds must be integers") from exc
    return _parse_csv(value, label="seeds", convert=as_int)


def parse_configs(value: str) -> list[Path]:
    paths = _parse_csv(value, label="configs", convert=lambda item: Path(item).expanduser())
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise ValueError(f"Config file does not exist or is not a regular file: {path}")
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def _config_id(config: Path) -> str:
    safe_stem = _COMPONENT_RE.sub("-", config.stem).strip(".-") or "config"
    digest = hashlib.sha256(str(config).encode("utf-8")).hexdigest()[:12]
    return f"{safe_stem}-{digest}"


def _safe_dataset(dataset: str) -> str:
    return _COMPONENT_RE.sub("-", dataset).strip(".-") or "dataset"


def run_status_path(artifact_root: Path, dataset: str, config: Path, seed: int) -> Path:
    """Return the deterministic scheduler-owned status marker for one identity."""
    return (
        artifact_root / "results" / "run_status"
        / _safe_dataset(dataset) / _config_id(config) / f"seed_{seed}" / "run_status.json"
    )


def run_artifact_root(artifact_root: Path, config: Path) -> Path:
    """Isolate C8-B artifact paths for configurations with the same model name."""
    return artifact_root / "matrix_artifacts" / _config_id(config)


def _status_payload(dataset: str, config: Path, seed: int, status: str, *, scoped_root: Path, **extra) -> dict:
    return {
        "dataset": dataset,
        "config": str(config),
        "seed": seed,
        "status": status,
        "artifact_root": str(scoped_root),
        "timestamp": _utc_now(),
        **extra,
    }


def is_completed(artifact_root: Path, dataset: str, config: Path, seed: int) -> bool:
    """A run is complete only when its valid, matching marker says completed."""
    payload = _read_json(run_status_path(artifact_root, dataset, config, seed))
    return bool(payload and payload.get("status") == "completed" and
                payload.get("dataset") == dataset and
                payload.get("config") == str(config) and
                payload.get("seed") == seed)


@dataclass
class MatrixSummary:
    artifact_root: Path
    runs: list[dict]

    @property
    def completed(self) -> int:
        return sum(run["status"] == "completed" for run in self.runs)

    @property
    def skipped(self) -> int:
        return sum(run["status"] == "skipped" for run in self.runs)

    @property
    def failed(self) -> int:
        return sum(run["status"] == "failed" for run in self.runs)

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0

    def to_dict(self) -> dict:
        return {
            "created_at": _utc_now(),
            "total_runs": len(self.runs),
            "completed": self.completed,
            "skipped": self.skipped,
            "failed": self.failed,
            "runs": self.runs,
        }


def _run_argv(dataset: str, config: Path, seed: int, artifact_root: Path,
              shared_artifact_root: Path | None, stages: str | None) -> list[str]:
    argv = [
        "--dataset", dataset,
        "--config", str(config),
        "--seed", str(seed),
        "--artifact-root", str(artifact_root),
    ]
    if shared_artifact_root is not None:
        argv.extend(["--shared-artifact-root", str(shared_artifact_root)])
    if stages is not None:
        argv.extend(["--stages", stages])
    return argv


class ChildRunError(RuntimeError):
    """A production run_experiment subprocess returned unsuccessfully."""

    def __init__(self, returncode: int):
        self.returncode = int(returncode)
        self.signal = -self.returncode if self.returncode < 0 else None
        if self.signal is not None:
            try:
                signal_name = signal.Signals(self.signal).name
            except ValueError:
                signal_name = f"signal {self.signal}"
            message = f"run_experiment child terminated by {signal_name} ({self.signal})"
        else:
            message = f"run_experiment child exited with return code {self.returncode}"
        super().__init__(message)


def _production_runner(argv: Sequence[str]) -> subprocess.CompletedProcess:
    """Run one matrix identity in a fresh interpreter for OS-level cleanup."""
    command = [sys.executable, str(SRC_ROOT / "experiments" / "run_experiment.py"), *argv]
    return subprocess.run(command, cwd=SRC_ROOT.parent, check=False)


def _record_failure(status_path: Path, dataset: str, config: Path, seed: int, scoped_root: Path,
                    exc: BaseException, *, returncode: int | None = None,
                    status_extra: dict | None = None) -> Path:
    failure_path = status_path.with_name("failure.json")
    failure = {
        "dataset": dataset,
        "config": str(config),
        "seed": seed,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "timestamp": _utc_now(),
        "git_commit": _git_commit(),
    }
    if returncode is not None:
        failure["returncode"] = int(returncode)
        failure["exit_code"] = int(returncode)
        signal_number = -int(returncode) if int(returncode) < 0 else None
        failure["signal"] = signal_number
        if signal_number is not None:
            try:
                failure["signal_name"] = signal.Signals(signal_number).name
            except ValueError:
                failure["signal_name"] = None
    _atomic_json(failure_path, failure)
    _atomic_json(status_path, _status_payload(
        dataset, config, seed, "failed", scoped_root=scoped_root,
        failure_path=str(failure_path), **(status_extra or {}),
    ))
    return failure_path


def _matching_stale_running(payload: dict | None, dataset: str, config: Path, seed: int) -> bool:
    return bool(
        payload
        and payload.get("status") == "running"
        and payload.get("dataset") == dataset
        and payload.get("config") == str(config)
        and payload.get("seed") == seed
    )


def run_matrix(datasets: Sequence[str], configs: Sequence[Path], seeds: Sequence[int], *,
               artifact_root: Path, stages: str | None = None, force: bool = False,
               runner: Callable[[Sequence[str] | None], object] | None = None) -> MatrixSummary:
    """Execute a stable dataset → config → seed matrix with process isolation."""
    # Explicit injection stays in-process.  The identity check also preserves
    # legacy tests/integrations that monkeypatch run_experiment.main.
    if runner is not None:
        invoke = runner
    elif run_experiment.main is not _ORIGINAL_RUN_EXPERIMENT_MAIN:
        invoke = run_experiment.main
    else:
        invoke = _production_runner
    records: list[dict] = []

    for dataset in datasets:
        for config in configs:
            scoped_root = run_artifact_root(artifact_root, config)
            for seed in seeds:
                status_path = run_status_path(artifact_root, dataset, config, seed)
                record = {
                    "dataset": dataset,
                    "config": str(config),
                    "seed": seed,
                    "status": "running",
                    "artifact_root": str(scoped_root),
                    "failure_path": None,
                    "stale_running_recovered": False,
                }
                if not force and is_completed(artifact_root, dataset, config, seed):
                    record["status"] = "skipped"
                    records.append(record)
                    continue

                previous = _read_json(status_path)
                stale_running = _matching_stale_running(previous, dataset, config, seed)
                status_extra = {}
                if stale_running:
                    record["stale_running_recovered"] = True
                    status_extra = {
                        "stale_running_recovered": True,
                        "previous_running_timestamp": previous.get("timestamp"),
                    }
                    _atomic_json(status_path, _status_payload(
                        dataset, config, seed, "interrupted", scoped_root=scoped_root,
                        interruption_reason="stale running marker recovered on scheduler startup",
                        **status_extra,
                    ))

                _atomic_json(status_path, _status_payload(
                    dataset, config, seed, "running", scoped_root=scoped_root, **status_extra,
                ))
                try:
                    result = invoke(_run_argv(
                        dataset, config, seed, scoped_root, artifact_root, stages
                    ))
                    returncode = getattr(result, "returncode", 0)
                    if returncode:
                        raise ChildRunError(returncode)
                except KeyboardInterrupt:
                    _atomic_json(status_path, _status_payload(
                        dataset, config, seed, "interrupted", scoped_root=scoped_root,
                        **status_extra,
                    ))
                    raise
                except SystemExit as exc:
                    code = exc.code if isinstance(exc.code, int) else 1
                    if code == 0:
                        record["status"] = "completed"
                        _atomic_json(status_path, _status_payload(
                            dataset, config, seed, "completed", scoped_root=scoped_root,
                            **status_extra,
                        ))
                    else:
                        failure_path = _record_failure(
                            status_path, dataset, config, seed, scoped_root, exc,
                            returncode=code, status_extra=status_extra,
                        )
                        record["status"] = "failed"
                        record["failure_path"] = str(failure_path)
                except ChildRunError as exc:
                    failure_path = _record_failure(
                        status_path, dataset, config, seed, scoped_root, exc,
                        returncode=exc.returncode, status_extra=status_extra,
                    )
                    record["status"] = "failed"
                    record["failure_path"] = str(failure_path)
                except Exception as exc:
                    failure_path = _record_failure(
                        status_path, dataset, config, seed, scoped_root, exc,
                        status_extra=status_extra,
                    )
                    record["status"] = "failed"
                    record["failure_path"] = str(failure_path)
                else:
                    record["status"] = "completed"
                    _atomic_json(status_path, _status_payload(
                        dataset, config, seed, "completed", scoped_root=scoped_root,
                        **status_extra,
                    ))
                records.append(record)

    summary = MatrixSummary(artifact_root=artifact_root, runs=records)
    _atomic_json(artifact_root / "results" / "matrix_summary.json", summary.to_dict())
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Schedule a deterministic ORTHRUS experiment matrix.")
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset names, in execution order.")
    parser.add_argument("--configs", required=True, help="Comma-separated YAML config paths, in execution order.")
    parser.add_argument("--seeds", required=True, help="Comma-separated integer seeds, in execution order.")
    parser.add_argument("--artifact-root", default=None, metavar="PATH", help="Matrix artifact root.")
    parser.add_argument("--stages", default=None, help="Optional C8-B pipeline stages passed unchanged to every run.")
    parser.add_argument("--force", action="store_true", help="Run completed identities again without deleting old artifacts.")
    return parser


def main(argv: Sequence[str] | None = None) -> MatrixSummary:
    args = build_parser().parse_args(argv)
    datasets = parse_datasets(args.datasets)
    configs = parse_configs(args.configs)
    seeds = parse_seeds(args.seeds)
    artifact_root = resolve_artifact_root(args.artifact_root, os.environ.get("ORTHRUS_ARTIFACT_ROOT"))
    return run_matrix(
        datasets, configs, seeds, artifact_root=artifact_root,
        stages=args.stages, force=args.force,
    )


def cli(argv: Sequence[str] | None = None) -> int:
    summary = main(argv)
    print(
        f"total_runs={len(summary.runs)} completed={summary.completed} "
        f"skipped={summary.skipped} failed={summary.failed}"
    )
    for run in summary.runs:
        if run["status"] == "failed":
            failure = _read_json(Path(run["failure_path"])) if run["failure_path"] else None
            reason = failure.get("exception_message", "unknown error") if failure else "unknown error"
            print(f"FAILED dataset={run['dataset']} config={run['config']} seed={run['seed']}: {reason}")
    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(cli())
