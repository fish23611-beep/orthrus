"""Static contracts for the six C8-F Colab notebooks."""
from __future__ import annotations

import json
import re
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
)
ENTRYPOINTS = (
    "src/experiments/run_experiment.py",
    "src/experiments/run_matrix.py",
    "src/experiments/collect_results.py",
    "src/experiments/export_tables.py",
)


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
