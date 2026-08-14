"""Static contracts for the six C8-F Colab notebooks + All-in-One Master Notebook."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

try:
    import nbformat
except ModuleNotFoundError:  # The minimal dev container does not bundle Jupyter.
    nbformat = None


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOK_NAMES = (
    "00_colab_environment.ipynb",
    "01_preprocess_theia.ipynb",
    "02_baseline_smoke_test.ipynb",
    "03_train_main_models.ipynb",
    "04_run_ablations.ipynb",
    "05_collect_results.ipynb",
    "ORTHRUS_MSTC_PIDS_AllInOne_Colab.ipynb",
)
ENTRYPOINTS = (
    "src/experiments/run_experiment.py",
    "src/experiments/run_matrix.py",
    "src/experiments/collect_results.py",
    "src/experiments/export_tables.py",
)
MASTER_NOTEBOOK_NAME = "ORTHRUS_MSTC_PIDS_AllInOne_Colab.ipynb"
REQUIRED_REPOSITORY_REF = "fix/c8-full-data-io"
REQUIRED_EXPECTED_COMMIT = "099604e139f6f4139af793658ee0bbfde931bd9e"
REQUIRED_SUBMODULE_UPDATE = 'git("submodule", "update", "--init", "--recursive")'


def _head_full_sha() -> str | None:
    """Return the full 40-char SHA of HEAD, or None if it cannot be resolved."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _merge_base_with_head(sha: str) -> str | None:
    """Return the merge-base of ``sha`` and HEAD, or None on failure."""
    try:
        return subprocess.check_output(
            ["git", "merge-base", sha, "HEAD"], cwd=ROOT, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _read_notebook(path: Path):
    if nbformat is not None:
        return nbformat.read(path, as_version=4)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _source(notebook) -> str:
    sources = []
    for cell in notebook["cells"]:
        value = cell.get("source", "")
        sources.append("".join(value) if isinstance(value, list) else value)
    return "\n".join(sources)


def _extract_expected_commit(source: str) -> str | None:
    """Pull the EXPECTED_COMMIT assignment from the notebook source."""
    match = re.search(r"EXPECTED_COMMIT\s*=\s*['\"]([0-9a-fA-F]+)['\"]", source)
    return match.group(1) if match else None


@pytest.fixture(scope="module")
def notebooks():
    return {name: _read_notebook(NOTEBOOK_DIR / name) for name in NOTEBOOK_NAMES}


def test_all_six_notebooks_are_valid_v4_documents_with_cells(notebooks):
    assert set(notebooks) == set(NOTEBOOK_NAMES)
    for name, document in notebooks.items():
        assert document["nbformat"] >= 4, name
        assert document["cells"], name
        assert all(cell.get("cell_type") in {"markdown", "code", "raw"} for cell in document["cells"])


def test_every_code_cell_compiles(notebooks):
    for name, document in notebooks.items():
        for index, cell in enumerate(document["cells"]):
            if cell.get("cell_type") == "code":
                value = cell.get("source", "")
                source = "".join(value) if isinstance(value, list) else value
                compile(source, f"{name}:cell-{index}", "exec")


def test_notebooks_have_no_unfinished_placeholders_or_saved_outputs(notebooks):
    unfinished = re.compile(r"\b(?:TODO|FIXME|PLACEHOLDER|implement later)\b", re.IGNORECASE)
    for name, document in notebooks.items():
        assert not unfinished.search(_source(document)), name
        for cell in document["cells"]:
            if cell.get("cell_type") == "code":
                assert cell.get("outputs", []) == [], name
                assert cell.get("execution_count") is None, name


def test_referenced_experiment_entrypoints_exist_and_are_used(notebooks):
    combined = "\n".join(_source(document) for document in notebooks.values())
    for relative in ENTRYPOINTS:
        assert (ROOT / relative).is_file(), relative
        assert relative in combined, relative
    assert "src/experiments/run_experiment.py" in _source(notebooks["02_baseline_smoke_test.ipynb"])
    assert "src/experiments/run_matrix.py" in _source(notebooks["03_train_main_models.ipynb"])
    assert "src/experiments/run_matrix.py" in _source(notebooks["04_run_ablations.ipynb"])
    assert "src/experiments/collect_results.py" in _source(notebooks["05_collect_results.ipynb"])
    assert "src/experiments/export_tables.py" in _source(notebooks["05_collect_results.ipynb"])


def test_notebooks_contain_no_secrets_or_personal_machine_paths(notebooks):
    forbidden = (
        re.compile(r"ghp_[A-Za-z0-9]+"),
        re.compile(r"github_pat_[A-Za-z0-9_]+"),
        re.compile(r"WANDB_API_KEY\s*=\s*['\"][^'$\"]+"),
        re.compile(r"ORTHRUS_DB_PASSWORD\s*=\s*['\"][^'$\"]+"),
        re.compile(r"[A-Za-z]:\\(?:Code|Users)\\", re.IGNORECASE),
        re.compile(r"/Users/[^/\s]+/"),
        re.compile(r"/home/[^/\s]+/"),
    )
    for name, document in notebooks.items():
        source = _source(document)
        assert not any(pattern.search(source) for pattern in forbidden), name


def test_notebooks_are_orchestration_only(notebooks):
    for name, document in notebooks.items():
        source = _source(document)
        assert "training.main" not in source, name
        assert "optimizer.step(" not in source, name
        assert "loss.backward(" not in source, name


# ================================================================================
# Master Notebook Specific Tests
# ================================================================================

def test_master_notebook_exists():
    """Master Notebook must exist."""
    master_path = NOTEBOOK_DIR / MASTER_NOTEBOOK_NAME
    assert master_path.is_file(), f"Master Notebook not found: {master_path}"


def test_master_notebook_is_valid_nbformat():
    """Master Notebook must be a valid nbformat 4 document."""
    master_path = NOTEBOOK_DIR / MASTER_NOTEBOOK_NAME
    notebook = _read_notebook(master_path)
    assert notebook["nbformat"] >= 4, MASTER_NOTEBOOK_NAME
    assert notebook["cells"], MASTER_NOTEBOOK_NAME


def test_master_notebook_has_required_cells():
    """Master Notebook must have cells for all major sections."""
    master_path = NOTEBOOK_DIR / MASTER_NOTEBOOK_NAME
    notebook = _read_notebook(master_path)
    source = _source(notebook)

    # Check for key section markers in markdown cells
    required_sections = [
        "Unified Parameters",
        "Mount Drive",
        "Repository Ref",
        "Resource Preflight",
        "Install Python Dependencies",
        "Environment Record",
        "Persistent Artifacts",
        "PostgreSQL",
        "Restartable Preprocessing",
        "Baseline Smoke",
        "Main Model Matrix",
        "Ablations",
        "Result Collection",
    ]

    for section in required_sections:
        assert section in source, f"Missing section in Master Notebook: {section}"


def test_master_notebook_uses_one_coherent_frozen_ref_mechanism():
    """Branch/tag checkout and strict commit verification stay coherent.

    The Master Notebook must:
      * pin REPOSITORY_REF == "fix/c8-full-data-io"
      * pin EXPECTED_COMMIT to a 40-character hex SHA
      * reject ``git rev-parse HEAD`` mismatches via ``actual_commit``
      * never use ``git describe --tags --exact-match`` (commit-only pin)
      * initialize the Ground Truth submodule idempotently
    """
    master_path = NOTEBOOK_DIR / MASTER_NOTEBOOK_NAME
    notebook = _read_notebook(master_path)
    source = _source(notebook)

    assert REQUIRED_REPOSITORY_REF in source
    assert "EXPECTED_COMMIT" in source
    assert "actual_commit" in source
    assert "Commit mismatch" in source
    assert "git describe --tags --exact-match" not in source
    assert REQUIRED_SUBMODULE_UPDATE in source

    # The notebook's EXPECTED_COMMIT must be a 40-character hex SHA.  A
    # 7-character short SHA fails the post-checkout equality test because
    # `git rev-parse HEAD` always returns the full 40-character form.
    expected_commit = _extract_expected_commit(source)
    assert expected_commit is not None, (
        "Master Notebook does not define EXPECTED_COMMIT"
    )
    assert len(expected_commit) == 40, (
        f"EXPECTED_COMMIT must be a 40-character full SHA, got "
        f"{len(expected_commit)} chars: {expected_commit!r}"
    )
    int(expected_commit, 16)  # raises ValueError on non-hex
    assert re.fullmatch(r"[0-9a-f]{40}", expected_commit), (
        f"EXPECTED_COMMIT must be lowercase hex, got {expected_commit!r}"
    )

    # The committed EXPECTED_COMMIT must reference the production code
    # commit that authored it.  Because the notebook is itself committed
    # in a separate second-stage docs commit, EXPECTED_COMMIT MUST NOT
    # be the current HEAD (which would be the docs commit) — that would
    # be the very bug we are closing.  Instead, EXPECTED_COMMIT must
    # be an ancestor of HEAD: the notebook's frozen pin still matches
    # the code that produced the compact-history sidecar cache and the
    # one-coherent-frozen-ref machinery, not the docs-only commit.
    head = _head_full_sha()
    assert head is not None, "Could not resolve HEAD to a full SHA"
    assert expected_commit != head, (
        f"EXPECTED_COMMIT {expected_commit} is HEAD itself; the notebook "
        "must pin the production code commit, not its own docs commit."
    )
    merge_base = _merge_base_with_head(expected_commit)
    assert merge_base == expected_commit, (
        f"EXPECTED_COMMIT {expected_commit} is not an ancestor of HEAD "
        f"{head}; the notebook must pin a commit that actually exists "
        "in the branch history."
    )


def test_master_notebook_references_all_entrypoints():
    """Master Notebook must reference all experiment entrypoints."""
    master_path = NOTEBOOK_DIR / MASTER_NOTEBOOK_NAME
    notebook = _read_notebook(master_path)
    source = _source(notebook)

    for entrypoint in ENTRYPOINTS:
        assert entrypoint in source, \
            f"Master Notebook must reference {entrypoint}"


def test_master_notebook_supports_both_theia_datasets():
    """Master Notebook must support both THEIA_E3 and THEIA_E5."""
    master_path = NOTEBOOK_DIR / MASTER_NOTEBOOK_NAME
    notebook = _read_notebook(master_path)
    source = _source(notebook)

    assert "THEIA_E3" in source, "Master Notebook must support THEIA_E3"
    assert "THEIA_E5" in source, "Master Notebook must support THEIA_E5"


def test_master_notebook_has_google_drive_paths():
    """Master Notebook must define Google Drive paths."""
    master_path = NOTEBOOK_DIR / MASTER_NOTEBOOK_NAME
    notebook = _read_notebook(master_path)
    source = _source(notebook)

    assert "DRIVE_ROOT" in source, "Master Notebook must define DRIVE_ROOT"
    assert "ARTIFACT_ROOT" in source, "Master Notebook must define ARTIFACT_ROOT"
    assert "/content/drive/MyDrive" in source, \
        "Master Notebook must use Google Drive paths"


def test_master_notebook_has_postgresql_restore():
    """Master Notebook must have PostgreSQL restore capability."""
    master_path = NOTEBOOK_DIR / MASTER_NOTEBOOK_NAME
    notebook = _read_notebook(master_path)
    source = _source(notebook)

    assert "RESTORE_DATABASE" in source, \
        "Master Notebook must have RESTORE_DATABASE flag"
    assert "pg_restore" in source, \
        "Master Notebook must use pg_restore"


def test_master_notebook_has_ablation_groups():
    """Master Notebook must define ablation experiment groups."""
    master_path = NOTEBOOK_DIR / MASTER_NOTEBOOK_NAME
    notebook = _read_notebook(master_path)
    source = _source(notebook)

    # Check for key ablation group names
    assert "ablation" in source, "Master Notebook must support ablation group"
    assert "multiscale" in source, "Master Notebook must support multiscale group"
    assert "time" in source, "Master Notebook must support time group"
    assert "calibration" in source, "Master Notebook must support calibration group"


def test_master_notebook_has_all_switches():
    """Master Notebook must have all experiment switches."""
    master_path = NOTEBOOK_DIR / MASTER_NOTEBOOK_NAME
    notebook = _read_notebook(master_path)
    source = _source(notebook)

    required_switches = [
        "RUN_PREPROCESS",
        "RUN_BASELINE_SMOKE",
        "RUN_MAIN_MATRIX",
        "RUN_ABLATIONS",
        "RUN_COLLECT_EXPORT",
        "RUN_MANUAL_RESUME",
    ]

    for switch in required_switches:
        assert switch in source, f"Master Notebook must have {switch} switch"


def test_master_notebook_no_todo_placeholder():
    """Master Notebook must not contain TODO/FIXME/PLACEHOLDER."""
    master_path = NOTEBOOK_DIR / MASTER_NOTEBOOK_NAME
    notebook = _read_notebook(master_path)

    unfinished = re.compile(
        r"\b(?:TODO|FIXME|PLACEHOLDER|implement later)\b",
        re.IGNORECASE
    )
    source = _source(notebook)
    assert not unfinished.search(source), \
        f"Master Notebook contains unfinished placeholders"

    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("outputs", []) == [], MASTER_NOTEBOOK_NAME
            assert cell.get("execution_count") is None, MASTER_NOTEBOOK_NAME


def test_master_notebook_is_english_only():
    """AllInOne cell source must contain no CJK characters."""
    master_path = NOTEBOOK_DIR / MASTER_NOTEBOOK_NAME
    notebook = _read_notebook(master_path)
    source = _source(notebook)
    cjk_chars = [c for c in source if "\u4e00" <= c <= "\u9fff"]
    assert len(cjk_chars) == 0, \
        f"Master Notebook contains {len(cjk_chars)} CJK characters"


def test_master_notebook_no_secrets_or_paths():
    """Master Notebook must not contain secrets or personal paths."""
    master_path = NOTEBOOK_DIR / MASTER_NOTEBOOK_NAME
    notebook = _read_notebook(master_path)
    source = _source(notebook)

    forbidden = (
        re.compile(r"ghp_[A-Za-z0-9]+"),
        re.compile(r"github_pat_[A-Za-z0-9_]+"),
        re.compile(r"WANDB_API_KEY\s*=\s*['\"][^'$\"]+"),
        re.compile(r"ORTHRUS_DB_PASSWORD\s*=\s*['\"][^'$\"]+"),
        re.compile(r"[A-Za-z]:\\(?:Code|Users)\\", re.IGNORECASE),
        re.compile(r"/Users/[^/\s]+/"),
        re.compile(r"/home/[^/\s]+/"),
    )

    for pattern in forbidden:
        match = pattern.search(source)
        assert not match, \
            f"Master Notebook contains forbidden pattern: {pattern.pattern}"
