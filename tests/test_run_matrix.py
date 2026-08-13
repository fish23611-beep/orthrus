from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments import run_matrix


def config(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text("pipeline: {mode: full_pipeline}\n", encoding="utf-8")
    return path


def args(root: Path, configs: list[Path], *, datasets="THEIA_E3", seeds="0", extra=()):
    return [
        "--datasets", datasets,
        "--configs", ",".join(map(str, configs)),
        "--seeds", seeds,
        "--artifact-root", str(root),
        *extra,
    ]


def test_matrix_expands_in_stable_dataset_config_seed_order(tmp_path, monkeypatch):
    c1, c2 = config(tmp_path, "c1.yml"), config(tmp_path, "c2.yml")
    calls = []
    monkeypatch.setattr(run_matrix.run_experiment, "main", lambda argv: calls.append(argv))

    summary = run_matrix.main(args(tmp_path / "artifacts", [c1, c2], datasets="THEIA_E3,THEIA_E5", seeds="0,1"))

    assert len(summary.runs) == 8
    assert [(r["dataset"], Path(r["config"]).name, r["seed"]) for r in summary.runs] == [
        ("THEIA_E3", "c1.yml", 0), ("THEIA_E3", "c1.yml", 1),
        ("THEIA_E3", "c2.yml", 0), ("THEIA_E3", "c2.yml", 1),
        ("THEIA_E5", "c1.yml", 0), ("THEIA_E5", "c1.yml", 1),
        ("THEIA_E5", "c2.yml", 0), ("THEIA_E5", "c2.yml", 1),
    ]
    assert len(calls) == 8
    assert calls[0][0:6] == ["--dataset", "THEIA_E3", "--config", str(c1.resolve()), "--seed", "0"]


def test_run_experiment_receives_artifact_root_and_stages(tmp_path, monkeypatch):
    c1 = config(tmp_path, "c1.yml")
    calls = []
    monkeypatch.setattr(run_matrix.run_experiment, "main", lambda argv: calls.append(argv))

    run_matrix.main(args(tmp_path / "artifacts", [c1], seeds="3", extra=("--stages", "test,evaluate")))

    call = calls[0]
    assert call[-2:] == ["--stages", "test,evaluate"]
    assert Path(call[call.index("--artifact-root") + 1]).name == run_matrix._config_id(c1.resolve())


def test_failure_isolated_and_failure_json_contains_traceback(tmp_path, monkeypatch):
    c1 = config(tmp_path, "c1.yml")
    calls = []

    def runner(argv):
        calls.append(argv)
        if "--seed" in argv and argv[argv.index("--seed") + 1] == "1":
            raise RuntimeError("boom")

    monkeypatch.setattr(run_matrix.run_experiment, "main", runner)
    summary = run_matrix.main(args(tmp_path / "artifacts", [c1], seeds="0,1,2"))

    assert len(calls) == 3
    assert (summary.completed, summary.failed, summary.skipped, summary.exit_code) == (2, 1, 0, 1)
    failure_path = Path(next(run["failure_path"] for run in summary.runs if run["status"] == "failed"))
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["exception_type"] == "RuntimeError"
    assert failure["exception_message"] == "boom"
    assert "RuntimeError: boom" in failure["traceback"]
    assert failure["dataset"] == "THEIA_E3" and failure["seed"] == 1


def test_completed_marker_skips_without_calling_runner(tmp_path, monkeypatch):
    c1 = config(tmp_path, "c1.yml").resolve()
    root = tmp_path / "artifacts"
    status = run_matrix.run_status_path(root, "THEIA_E3", c1, 0)
    run_matrix._atomic_json(status, run_matrix._status_payload(
        "THEIA_E3", c1, 0, "completed", scoped_root=run_matrix.run_artifact_root(root, c1),
    ))
    monkeypatch.setattr(run_matrix.run_experiment, "main", lambda argv: pytest.fail("must skip"))

    summary = run_matrix.main(args(root, [c1]))

    assert summary.skipped == 1 and summary.completed == 0


def test_incomplete_marker_does_not_skip(tmp_path, monkeypatch):
    c1 = config(tmp_path, "c1.yml").resolve()
    root = tmp_path / "artifacts"
    status = run_matrix.run_status_path(root, "THEIA_E3", c1, 0)
    status.parent.mkdir(parents=True)
    status.write_text("{not valid json", encoding="utf-8")
    calls = []
    monkeypatch.setattr(run_matrix.run_experiment, "main", lambda argv: calls.append(argv))

    summary = run_matrix.main(args(root, [c1]))

    assert len(calls) == 1 and summary.completed == 1


def test_invalid_config_fails_before_any_run(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run_matrix.run_experiment, "main", lambda argv: calls.append(argv))

    with pytest.raises(ValueError, match="Config file does not exist"):
        run_matrix.main(args(tmp_path / "artifacts", [tmp_path / "missing.yml"]))

    assert calls == []


def test_invalid_seed_fails_clearly():
    with pytest.raises(ValueError, match="Invalid seed 'abc'"):
        run_matrix.parse_seeds("0,abc,2")


def test_duplicate_dataset_and_seed_preserve_first_order():
    assert run_matrix.parse_datasets("THEIA_E3,THEIA_E3,THEIA_E5") == ["THEIA_E3", "THEIA_E5"]
    assert run_matrix.parse_seeds("0,0,1") == [0, 1]


def test_oom_is_recorded_without_retry_or_config_mutation(tmp_path, monkeypatch):
    c1 = config(tmp_path, "c1.yml")
    before = c1.read_text(encoding="utf-8")
    calls = []

    def runner(argv):
        calls.append(argv)
        if argv[argv.index("--seed") + 1] == "0":
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(run_matrix.run_experiment, "main", runner)
    summary = run_matrix.main(args(tmp_path / "artifacts", [c1], seeds="0,1"))

    assert len(calls) == 2 and summary.failed == 1 and summary.completed == 1
    assert c1.read_text(encoding="utf-8") == before
    failure = json.loads(Path(summary.runs[0]["failure_path"]).read_text(encoding="utf-8"))
    assert "CUDA out of memory" in failure["exception_message"]


def test_keyboard_interrupt_is_not_swallowed(tmp_path, monkeypatch):
    c1 = config(tmp_path, "c1.yml")
    monkeypatch.setattr(run_matrix.run_experiment, "main", lambda argv: (_ for _ in ()).throw(KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        run_matrix.main(args(tmp_path / "artifacts", [c1]))


def test_cli_exit_code_reflects_partial_failure(tmp_path, monkeypatch, capsys):
    c1 = config(tmp_path, "c1.yml")
    monkeypatch.setattr(run_matrix.run_experiment, "main", lambda argv: (_ for _ in ()).throw(RuntimeError("bad")))

    assert run_matrix.cli(args(tmp_path / "artifacts", [c1])) == 1
    assert "failed=1" in capsys.readouterr().out


def test_cli_exit_zero_when_all_runs_succeed(tmp_path, monkeypatch):
    c1 = config(tmp_path, "c1.yml")
    monkeypatch.setattr(run_matrix.run_experiment, "main", lambda argv: None)

    assert run_matrix.cli(args(tmp_path / "artifacts", [c1])) == 0


def test_force_reruns_completed_identity(tmp_path, monkeypatch):
    c1 = config(tmp_path, "c1.yml").resolve()
    root = tmp_path / "artifacts"
    status = run_matrix.run_status_path(root, "THEIA_E3", c1, 0)
    run_matrix._atomic_json(status, run_matrix._status_payload(
        "THEIA_E3", c1, 0, "completed", scoped_root=run_matrix.run_artifact_root(root, c1),
    ))
    calls = []
    monkeypatch.setattr(run_matrix.run_experiment, "main", lambda argv: calls.append(argv))

    summary = run_matrix.main(args(root, [c1], extra=("--force",)))

    assert len(calls) == 1 and summary.completed == 1 and summary.skipped == 0


def test_production_default_uses_fresh_python_subprocess(tmp_path, monkeypatch):
    c1 = config(tmp_path, "c1.yml").resolve()
    calls = []

    def fake_subprocess_run(command, **kwargs):
        calls.append((command, kwargs))
        return run_matrix.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_matrix.subprocess, "run", fake_subprocess_run)
    summary = run_matrix.run_matrix(
        ["THEIA_E3"], [c1], [0], artifact_root=tmp_path / "artifacts"
    )

    assert summary.completed == 1
    command, kwargs = calls[0]
    assert command[0] == sys.executable
    assert Path(command[1]).name == "run_experiment.py"
    assert kwargs["check"] is False


def test_child_sigkill_records_returncode_signal_and_failed_status(tmp_path):
    c1 = config(tmp_path, "c1.yml").resolve()
    root = tmp_path / "artifacts"

    def killed(_argv):
        return run_matrix.subprocess.CompletedProcess([], -9)

    summary = run_matrix.run_matrix(
        ["THEIA_E3"], [c1], [0], artifact_root=root, runner=killed
    )

    assert summary.failed == 1
    failure = json.loads(Path(summary.runs[0]["failure_path"]).read_text(encoding="utf-8"))
    assert failure["returncode"] == -9
    assert failure["signal"] == 9
    assert failure["signal_name"] == "SIGKILL"
    status = json.loads(
        run_matrix.run_status_path(root, "THEIA_E3", c1, 0).read_text(encoding="utf-8")
    )
    assert status["status"] == "failed"


def test_matching_stale_running_is_recovered_and_safely_rerun(tmp_path):
    c1 = config(tmp_path, "c1.yml").resolve()
    root = tmp_path / "artifacts"
    status_path = run_matrix.run_status_path(root, "THEIA_E3", c1, 0)
    run_matrix._atomic_json(
        status_path,
        run_matrix._status_payload(
            "THEIA_E3", c1, 0, "running",
            scoped_root=run_matrix.run_artifact_root(root, c1),
        ),
    )
    calls = []

    summary = run_matrix.run_matrix(
        ["THEIA_E3"], [c1], [0], artifact_root=root,
        runner=lambda argv: calls.append(argv),
    )

    assert len(calls) == 1
    assert summary.completed == 1
    assert summary.runs[0]["stale_running_recovered"] is True
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["stale_running_recovered"] is True
    assert status["previous_running_timestamp"] is not None
