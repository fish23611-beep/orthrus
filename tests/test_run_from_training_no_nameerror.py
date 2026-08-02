"""
tests/test_run_from_training_no_nameerror.py

Unit tests for the --run_from_training path in orthrus.py:
- --run_from_training does NOT access uninitialized timing variables
- Preprocess timing fields are 0.0 when preprocess is skipped
- train/test/evaluate timing fields are present when those stages run
- --run_from_training + explicit --stages raises ValueError
- cfg.pipeline.run_tracing is correctly set by --skip-tracing
- Artifact prerequisite checks are correct

Does NOT connect to PostgreSQL or read real THEIA data.
"""
import sys
import os
from pathlib import Path

# Ensure src/ is on the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch


# -------------------------------------------------------------------------- #
# Dependency availability checks
# -------------------------------------------------------------------------- #
torch_available = True
try:
    import torch  # noqa: F401
    import wandb  # noqa: F401
except ImportError:
    torch_available = False

requires_torch = pytest.mark.skipif(
    not torch_available,
    reason="torch/wandb not installed in venv — tests skipped",
)

# config.py imports psycopg2 at module level; guard against that too.
config_available = True
try:
    from config import get_default_cfg  # noqa: F401
except ImportError:
    config_available = False

requires_config = pytest.mark.skipif(
    not config_available,
    reason="config module (psycopg2) not available — tests skipped",
)


# -------------------------------------------------------------------------- #
# Shared fixture — patches all pipeline stage functions so tests never
# touch the filesystem, database, or external services.
# -------------------------------------------------------------------------- #
@pytest.fixture
def mock_pipeline_runtime():
    """Patch all pipeline stage functions and wandb.

    Returns a dict of MagicMock objects keyed by stage name.
    Does NOT patch _check_artifact_prerequisites (that is tested separately
    in TestArtifactPrerequisites).
    """
    with ExitStack() as stack:
        mocks = {
            "build_graphs":  stack.enter_context(patch("orthrus.build_orthrus_graphs.main")),
            "embed_nodes":    stack.enter_context(patch("orthrus.build_feature_word2vec.main")),
            "embed_edges":    stack.enter_context(patch("orthrus.embed_edges_feature_word2vec.main")),
            "train":          stack.enter_context(patch("orthrus.orthrus_gnn_training.main")),
            "test":           stack.enter_context(patch("orthrus.orthrus_gnn_testing.main")),
            "evaluate":       stack.enter_context(patch("orthrus.evaluation.main")),
            "trace":          stack.enter_context(patch("orthrus.tracing.main")),
            "artifact_check": stack.enter_context(patch("orthrus._check_artifact_prerequisites")),
            "wandb":         stack.enter_context(patch("orthrus.wandb")),
        }
        # Simulate no active wandb run so wandb.log calls are no-ops
        mocks["wandb"].run = None
        yield mocks


# -------------------------------------------------------------------------- #
# Test --run_from_training timing
# -------------------------------------------------------------------------- #
class TestRunFromTrainingTiming:

    @requires_torch
    def test_run_from_training_no_nameerror(self, mock_pipeline_runtime):
        """--run_from_training must not raise NameError for any timing variable.

        Verifies: train/test/evaluate called, trace not called,
        returned timing dict contains all expected keys, preprocess timing is 0.0.
        """
        import orthrus

        args = MagicMock()
        args.run_from_training = True
        args.stages = None
        args.skip_tracing = False

        cfg = MagicMock()
        cfg.pipeline = MagicMock()
        cfg.pipeline.run_tracing = True
        cfg.detection.gnn_training.use_seed = False

        result = orthrus.main(cfg, args)

        assert isinstance(result, dict)

        m = mock_pipeline_runtime
        m["train"].assert_called_once_with(cfg)
        m["test"].assert_called_once_with(cfg)
        m["evaluate"].assert_called_once_with(cfg)
        m["trace"].assert_not_called()

        assert result['time_build_graphs'] == 0.0
        assert result['time_embed_nodes'] == 0.0
        assert result['time_embed_edges'] == 0.0
        assert 'time_gnn_training' in result
        assert 'time_gnn_testing' in result
        assert 'time_evaluation' in result
        assert 'time_total' in result
        assert 'time_tracing' in result

    @requires_torch
    def test_preprocess_timing_is_zero_when_skipped(self, mock_pipeline_runtime):
        """When preprocess is skipped, time_build_graphs / embed_nodes / embed_edges are 0.0."""
        import orthrus

        args = MagicMock()
        args.run_from_training = True
        args.stages = None
        args.skip_tracing = True

        cfg = MagicMock()
        cfg.pipeline = MagicMock()
        cfg.pipeline.run_tracing = False
        cfg.detection.gnn_training.use_seed = False

        result = orthrus.main(cfg, args)

        for key in ['time_build_graphs', 'time_embed_nodes', 'time_embed_edges']:
            assert key in result, f"{key} missing from time_consumption"
            assert result[key] == 0.0, f"{key} should be 0.0 when preprocess skipped, got {result[key]}"

    @requires_torch
    def test_train_test_evaluate_timing_fields_present(self, mock_pipeline_runtime):
        """train/test/evaluate timing fields are present and numeric when stages run."""
        import orthrus

        args = MagicMock()
        args.run_from_training = True
        args.stages = None
        args.skip_tracing = True

        cfg = MagicMock()
        cfg.pipeline = MagicMock()
        cfg.pipeline.run_tracing = False
        cfg.detection.gnn_training.use_seed = False

        result = orthrus.main(cfg, args)

        for key in ['time_gnn_training', 'time_gnn_testing', 'time_evaluation']:
            assert key in result, f"{key} missing from time_consumption"
            assert isinstance(result[key], (int, float)), f"{key} should be numeric"
            assert result[key] >= 0.0, f"{key} should be >= 0.0"

    @requires_torch
    def test_time_total_always_present(self, mock_pipeline_runtime):
        """time_total is always present regardless of which stages run."""
        import orthrus

        args = MagicMock()
        args.run_from_training = True
        args.stages = None
        args.skip_tracing = True

        cfg = MagicMock()
        cfg.pipeline = MagicMock()
        cfg.pipeline.run_tracing = False
        cfg.detection.gnn_training.use_seed = False

        result = orthrus.main(cfg, args)

        assert 'time_total' in result
        assert isinstance(result['time_total'], (int, float))
        assert result['time_total'] >= 0.0

    @requires_torch
    def test_time_tracing_is_zero_when_skipped(self, mock_pipeline_runtime):
        """time_tracing is 0.0 when tracing is skipped (trace not in stages, run_tracing=False)."""
        import orthrus

        args = MagicMock()
        args.run_from_training = False
        args.stages = "train,test,evaluate"
        args.skip_tracing = True

        cfg = MagicMock()
        cfg.pipeline = MagicMock()
        cfg.pipeline.run_tracing = False
        cfg.detection.gnn_training.use_seed = False

        result = orthrus.main(cfg, args)

        assert 'time_tracing' in result
        assert result['time_tracing'] == 0.0, f"time_tracing should be 0.0 when skipped, got {result['time_tracing']}"


# -------------------------------------------------------------------------- #
# --run_from_training + --stages conflict
# -------------------------------------------------------------------------- #
class TestRunFromTrainingStageConflict:

    @requires_torch
    def test_both_set_raises_valueerror(self):
        """Providing both --run_from_training and --stages raises ValueError."""
        import orthrus

        args = MagicMock()
        args.run_from_training = True
        args.stages = "train,test"
        args.skip_tracing = False

        cfg = MagicMock()
        cfg.pipeline = MagicMock()
        cfg.pipeline.run_tracing = True
        cfg.detection.gnn_training.use_seed = False

        with pytest.raises(ValueError) as exc_info:
            orthrus.main(cfg, args)
        assert "conflict" in str(exc_info.value).lower()

    @requires_torch
    def test_stages_alone_no_conflict(self, mock_pipeline_runtime):
        """--stages without --run_from_training does not raise."""
        import orthrus

        args = MagicMock()
        args.run_from_training = False
        args.stages = "preprocess,train"
        args.skip_tracing = False

        cfg = MagicMock()
        cfg.pipeline = MagicMock()
        cfg.pipeline.run_tracing = True
        cfg.detection.gnn_training.use_seed = False

        result = orthrus.main(cfg, args)
        assert isinstance(result, dict)

    @requires_torch
    def test_run_from_training_alone_no_conflict(self, mock_pipeline_runtime):
        """--run_from_training without --stages does not raise."""
        import orthrus

        args = MagicMock()
        args.run_from_training = True
        args.stages = None
        args.skip_tracing = True

        cfg = MagicMock()
        cfg.pipeline = MagicMock()
        cfg.pipeline.run_tracing = False
        cfg.detection.gnn_training.use_seed = False

        result = orthrus.main(cfg, args)
        assert isinstance(result, dict)


# -------------------------------------------------------------------------- #
# --skip-tracing sets cfg.pipeline.run_tracing
# -------------------------------------------------------------------------- #
class TestSkipTracingConfig:

    @requires_config
    def test_skip_tracing_sets_run_tracing_false(self):
        """When args.skip_tracing=True, cfg.pipeline.run_tracing should be False."""
        from config import get_default_cfg

        args = MagicMock()
        args.dataset = "THEIA_E3"
        args.model = "orthrus"
        args.wandb = False
        args.exp = ""
        args.tags = ""
        args.cpu = False
        args.from_weights = False
        args.seed = 0
        args.run_from_training = False
        args.show_attack = 0
        args.gt_type = "orthrus"
        args.plot_gt = False
        args.stages = None
        args.skip_tracing = True  # explicitly True

        cfg = get_default_cfg(args)
        assert cfg.pipeline.run_tracing is False, \
            "skip_tracing=True should set pipeline.run_tracing=False"

    @requires_config
    def test_no_skip_tracing_keeps_run_tracing_true(self):
        """When args.skip_tracing=False (default), cfg.pipeline.run_tracing stays True."""
        from config import get_default_cfg

        args = MagicMock()
        args.dataset = "THEIA_E3"
        args.model = "orthrus"
        args.wandb = False
        args.exp = ""
        args.tags = ""
        args.cpu = False
        args.from_weights = False
        args.seed = 0
        args.run_from_training = False
        args.show_attack = 0
        args.gt_type = "orthrus"
        args.plot_gt = False
        args.stages = None
        args.skip_tracing = False  # explicitly False

        cfg = get_default_cfg(args)
        assert cfg.pipeline.run_tracing is True, \
            "skip_tracing=False should keep pipeline.run_tracing=True"


# -------------------------------------------------------------------------- #
# Artifact prerequisite checks — real filesystem via tmp_path.
# These tests do NOT patch _check_artifact_prerequisites so they verify the
# real logic against real (tmp) directories.
# -------------------------------------------------------------------------- #
class TestArtifactPrerequisites:

    @requires_torch
    def test_test_stage_without_train_missing_checkpoint_dir(self, tmp_path):
        """test without train: missing checkpoint dir raises FileNotFoundError."""
        import orthrus

        cfg = MagicMock()
        cfg.detection.gnn_training._trained_models_dir = str(tmp_path / "nonexistent")
        cfg.pipeline = MagicMock()
        cfg.pipeline.run_tracing = False
        cfg.detection.gnn_training.use_seed = False

        args = MagicMock()
        args.run_from_training = False
        args.stages = "test"
        args.skip_tracing = False

        with pytest.raises(FileNotFoundError) as exc_info:
            orthrus.main(cfg, args)
        assert "test" in str(exc_info.value).lower()
        assert "train" in str(exc_info.value).lower()

    @requires_torch
    def test_test_stage_without_train_empty_checkpoint_dir(self, tmp_path):
        """test without train: empty checkpoint dir raises FileNotFoundError."""
        import orthrus

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        cfg = MagicMock()
        cfg.detection.gnn_training._trained_models_dir = str(checkpoint_dir)
        cfg.pipeline = MagicMock()
        cfg.pipeline.run_tracing = False
        cfg.detection.gnn_training.use_seed = False

        args = MagicMock()
        args.run_from_training = False
        args.stages = "test"
        args.skip_tracing = False

        with pytest.raises(FileNotFoundError) as exc_info:
            orthrus.main(cfg, args)
        assert "empty" in str(exc_info.value).lower()

    @requires_torch
    def test_test_with_train_skips_checkpoint_check(self, tmp_path):
        """test WITH train: _check_artifact_prerequisites is called but
        no FileNotFoundError should be raised even though the checkpoint dir
        does not exist — the check is skipped when train is in stages."""
        import orthrus

        cfg = MagicMock()
        # Point to a directory that does NOT exist
        cfg.detection.gnn_training._trained_models_dir = str(tmp_path / "nonexistent")
        cfg.pipeline = MagicMock()
        cfg.pipeline.run_tracing = False
        cfg.detection.gnn_training.use_seed = False

        args = MagicMock()
        args.run_from_training = False
        args.stages = "train,test"
        args.skip_tracing = False

        # Patch stage functions to prevent real training
        with patch("orthrus.orthrus_gnn_training.main") as m_train, \
             patch("orthrus.orthrus_gnn_testing.main") as m_test, \
             patch("orthrus.wandb") as m_wandb:
            m_wandb.run = None
            m_train.return_value = None
            m_test.return_value = None
            result = orthrus.main(cfg, args)

        m_train.assert_called_once()
        m_test.assert_called_once()
        assert isinstance(result, dict)

    @requires_torch
    def test_evaluate_stage_without_test_missing_edge_losses_dir(self, tmp_path):
        """evaluate without test: missing edge-loss dir raises FileNotFoundError."""
        import orthrus

        cfg = MagicMock()
        cfg.detection.gnn_testing._edge_losses_dir = str(tmp_path / "nonexistent")
        cfg.pipeline = MagicMock()
        cfg.pipeline.run_tracing = False
        cfg.detection.gnn_training.use_seed = False

        args = MagicMock()
        args.run_from_training = False
        args.stages = "evaluate"
        args.skip_tracing = False

        with pytest.raises(FileNotFoundError) as exc_info:
            orthrus.main(cfg, args)
        assert "evaluate" in str(exc_info.value).lower()

    @requires_torch
    def test_evaluate_stage_without_test_no_test_split(self, tmp_path):
        """evaluate without test: dir exists but no 'test' split raises FileNotFoundError."""
        import orthrus

        edge_dir = tmp_path / "edge_losses"
        edge_dir.mkdir()
        (edge_dir / "train").mkdir()  # wrong split — no "test" subdir

        cfg = MagicMock()
        cfg.detection.gnn_testing._edge_losses_dir = str(edge_dir)
        cfg.pipeline = MagicMock()
        cfg.pipeline.run_tracing = False
        cfg.detection.gnn_training.use_seed = False

        args = MagicMock()
        args.run_from_training = False
        args.stages = "evaluate"
        args.skip_tracing = False

        with pytest.raises(FileNotFoundError) as exc_info:
            orthrus.main(cfg, args)
        assert "test" in str(exc_info.value).lower()

    @requires_torch
    def test_evaluate_with_test_skips_test_output_check_but_requires_model(self, tmp_path):
        """test,evaluate run together: evaluate does not check for prior test output,
        but if train is NOT in stages the checkpoint dir must exist and be non-empty.

        Setup:
        - train NOT in stages → artifact check for model weights MUST pass
        - test IS in stages    → no artifact check for edge-loss output
        - evaluate IS in stages → no artifact check for edge-loss output
        """
        import orthrus

        # Real checkpoint dir with a fake weight file — satisfies the model check
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "model_epoch_1.pt").touch()

        # Edge-loss dir does NOT exist — but this should NOT be checked
        # because test IS in the current stages
        edge_losses_dir = tmp_path / "edge_losses"

        cfg = MagicMock()
        cfg.detection.gnn_training._trained_models_dir = str(checkpoint_dir)
        cfg.detection.gnn_testing._edge_losses_dir = str(edge_losses_dir)
        cfg.pipeline = MagicMock()
        cfg.pipeline.run_tracing = False
        cfg.detection.gnn_training.use_seed = False

        args = MagicMock()
        args.run_from_training = False
        args.stages = "test,evaluate"
        args.skip_tracing = False

        with patch("orthrus.orthrus_gnn_testing.main") as m_test, \
             patch("orthrus.evaluation.main") as m_eval, \
             patch("orthrus.wandb") as m_wandb:
            m_wandb.run = None
            m_test.return_value = None
            m_eval.return_value = None
            result = orthrus.main(cfg, args)

        # test and evaluate both ran; no FileNotFoundError raised
        m_test.assert_called_once()
        m_eval.assert_called_once()
        assert isinstance(result, dict)


# -------------------------------------------------------------------------- #
# All timing fields present for every stage combination
# -------------------------------------------------------------------------- #
class TestTimingCompleteness:

    @requires_torch
    @pytest.mark.parametrize("stages,expected_skipped", [
        # (stages string, list of timing keys that should be 0.0 when those stages are absent)
        ("preprocess",
         ["time_gnn_training", "time_gnn_testing", "time_evaluation", "time_tracing"]),
        ("preprocess,train",
         ["time_gnn_testing", "time_evaluation", "time_tracing"]),
        ("train",
         ["time_build_graphs", "time_embed_nodes", "time_embed_edges",
          "time_gnn_testing", "time_evaluation", "time_tracing"]),
        ("test,evaluate",
         ["time_build_graphs", "time_embed_nodes", "time_embed_edges",
          "time_gnn_training", "time_tracing"]),
        ("evaluate",
         ["time_build_graphs", "time_embed_nodes", "time_embed_edges",
          "time_gnn_training", "time_gnn_testing", "time_tracing"]),
    ])
    def test_skipped_stages_have_zero_timing(self, stages, expected_skipped, mock_pipeline_runtime):
        """Timing for stages not in the pipeline is 0.0 (not missing, not None)."""
        import orthrus

        all_timing_keys = [
            'time_build_graphs', 'time_embed_nodes', 'time_embed_edges',
            'time_gnn_training', 'time_gnn_testing', 'time_evaluation', 'time_tracing',
        ]

        args = MagicMock()
        args.run_from_training = False
        args.stages = stages
        args.skip_tracing = False

        cfg = MagicMock()
        cfg.pipeline = MagicMock()
        cfg.pipeline.run_tracing = ("trace" in stages)
        cfg.detection.gnn_training.use_seed = False

        result = orthrus.main(cfg, args)

        # All keys must be present
        for k in all_timing_keys:
            assert k in result, f"{k} missing for stages={stages}"

        # Skipped stages have 0.0 timing
        for k in expected_skipped:
            assert result[k] == 0.0, f"{k} should be 0.0 for stages={stages}, got {result[k]}"

        # time_total is always present
        assert 'time_total' in result
