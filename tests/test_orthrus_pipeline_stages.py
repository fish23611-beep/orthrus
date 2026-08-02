"""
tests/test_orthrus_pipeline_stages.py

Unit tests for the orthrus pipeline stage control mechanism:
- --stages argument parsing
- stage ordering
- trace gating (run_tracing / --skip-tracing)
- conflict detection (--stages + --run_from_training)

Routing / integration tests (TestTraceGating) mock all stage functions and
artifact checks so they never touch the filesystem or database.

Does NOT connect to PostgreSQL or read real THEIA data.
"""
import sys
import os

# Ensure src/ is on the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import MagicMock, patch


# -------------------------------------------------------------------------- #
# torch availability check
# -------------------------------------------------------------------------- #
torch_available = True
try:
    import torch  # noqa: F401
    import wandb  # noqa: F401
except ImportError:
    torch_available = False

requires_torch = pytest.mark.skipif(
    not torch_available,
    reason="torch/wandb not installed in venv — integration tests skipped",
)


# -------------------------------------------------------------------------- #
# Import stage-parsing helpers (no torch/wandb deps)
# -------------------------------------------------------------------------- #
from pipeline_stages import (
    parse_stages as _parse_stages,
    check_conflict as _check_conflict,
    STANDARD_STAGES,
    VALID_STAGES,
)


# -------------------------------------------------------------------------- #
# Fixtures
# -------------------------------------------------------------------------- #
@pytest.fixture
def mock_cfg():
    cfg = MagicMock()
    cfg.pipeline = MagicMock()
    cfg.pipeline.run_tracing = True
    return cfg


@pytest.fixture
def mock_args():
    args = MagicMock()
    args.run_from_training = False
    args.stages = None
    args.skip_tracing = False
    return args


# -------------------------------------------------------------------------- #
# _parse_stages
# -------------------------------------------------------------------------- #
class TestParseStages:

    def test_none_no_run_from_training_full(self):
        """No --stages and no --run_from_training → full pipeline."""
        result = _parse_stages(None, run_from_training=False)
        assert result == ["preprocess", "train", "test", "evaluate", "trace"]

    def test_none_run_from_training(self):
        """--run_from_training → skip preprocess, run train+test+evaluate."""
        result = _parse_stages(None, run_from_training=True)
        assert result == ["train", "test", "evaluate"]

    def test_all(self):
        """--stages all → full pipeline."""
        result = _parse_stages("all", run_from_training=False)
        assert result == ["preprocess", "train", "test", "evaluate", "trace"]

    def test_preprocess_only(self):
        """--stages preprocess → only preprocess."""
        result = _parse_stages("preprocess", run_from_training=False)
        assert result == ["preprocess"]

    def test_train_only(self):
        """--stages train → only train."""
        result = _parse_stages("train", run_from_training=False)
        assert result == ["train"]

    def test_train_test_evaluate(self):
        """--stages train,test,evaluate → correct order, no duplicates."""
        result = _parse_stages("train,test,evaluate", run_from_training=False)
        assert result == ["train", "test", "evaluate"]

    def test_train_test_evaluate_trace(self):
        """--stages train,test,evaluate,trace → includes trace."""
        result = _parse_stages("train,test,evaluate,trace", run_from_training=False)
        assert result == ["train", "test", "evaluate", "trace"]

    def test_no_duplicate_order(self):
        """Duplicate stage names are collapsed in first-seen order."""
        result = _parse_stages("train,train,test", run_from_training=False)
        assert result == ["train", "test"]

    def test_empty_string_raises(self):
        """Empty --stages string raises ValueError."""
        with pytest.raises(ValueError):
            _parse_stages("", run_from_training=False)

    def test_whitespace_only_raises(self):
        """Whitespace-only --stages raises ValueError."""
        with pytest.raises(ValueError):
            _parse_stages("   ,  ", run_from_training=False)

    def test_invalid_stage_raises(self):
        """Unknown stage name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _parse_stages("preprocess,frobnicate", run_from_training=False)
        assert "frobnicate" in str(exc_info.value)
        assert "Invalid stage" in str(exc_info.value)

    def test_preprocess_traces_still_added(self):
        """preprocess alone does NOT auto-add train/test/evaluate/trace."""
        result = _parse_stages("preprocess", run_from_training=False)
        assert result == ["preprocess"]
        assert "train" not in result

    def test_no_stages_auto_adds_trace_by_default(self):
        """Default (no --stages, no --run_from_training) includes trace."""
        result = _parse_stages(None, run_from_training=False)
        assert "trace" in result

    def test_explicit_without_trace(self):
        """Explicit stages without 'trace' omits it."""
        result = _parse_stages("preprocess,train", run_from_training=False)
        assert "trace" not in result

    def test_case_sensitive(self):
        """Stage names are case-sensitive."""
        with pytest.raises(ValueError):
            _parse_stages("Preprocess", run_from_training=False)

    def test_all_uppercase_accepted(self):
        """'ALL' (any case) is accepted as alias for 'all'."""
        result = _parse_stages("ALL", run_from_training=False)
        assert result == ["preprocess", "train", "test", "evaluate", "trace"]

    def test_stages_evaluate_alone(self):
        """--stages evaluate → only evaluate (no dependency enforced)."""
        result = _parse_stages("evaluate", run_from_training=False)
        assert result == ["evaluate"]

    def test_stages_test_evaluate(self):
        """--stages test,evaluate → test+evaluate (no dependency enforced)."""
        result = _parse_stages("test,evaluate", run_from_training=False)
        assert result == ["test", "evaluate"]

    def test_stages_train_test_evaluate(self):
        """--stages train,test,evaluate → all three in order."""
        result = _parse_stages("train,test,evaluate", run_from_training=False)
        assert result == ["train", "test", "evaluate"]

    def test_stages_train_only(self):
        """--stages train → only train (no dependency enforced)."""
        result = _parse_stages("train", run_from_training=False)
        assert result == ["train"]


# -------------------------------------------------------------------------- #
# _check_conflict
# -------------------------------------------------------------------------- #
class TestStageConflict:

    def test_no_conflict_no_stages(self):
        """No --stages, no --run_from_training → no conflict."""
        _check_conflict(None, run_from_training=False)  # should not raise

    def test_no_conflict_stages_only(self):
        """--stages without --run_from_training → no conflict."""
        _check_conflict("train,test", run_from_training=False)  # should not raise

    def test_no_conflict_run_from_training_only(self):
        """--run_from_training without --stages → no conflict."""
        _check_conflict(None, run_from_training=True)  # should not raise

    def test_conflict_raises(self):
        """Both --stages and --run_from_training → ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _check_conflict("train,test", run_from_training=True)
        assert "conflict" in str(exc_info.value).lower()
        assert "--stages" in str(exc_info.value)
        assert "--run_from_training" in str(exc_info.value)


# -------------------------------------------------------------------------- #
# Trace gating — integration tests with all stage functions mocked
# -------------------------------------------------------------------------- #
class TestTraceGating:

    @requires_torch
    @patch('orthrus._check_artifact_prerequisites')
    @patch('orthrus.wandb')
    @patch('orthrus.orthrus_gnn_training')
    @patch('orthrus.orthrus_gnn_testing')
    @patch('orthrus.evaluation')
    @patch('orthrus.tracing')
    @patch('orthrus.build_orthrus_graphs')
    @patch('orthrus.build_feature_word2vec')
    @patch('orthrus.embed_edges_feature_word2vec')
    def test_trace_called_when_in_stages_and_run_tracing_true(
            self, m_embed, m_w2v, m_graphs,
            m_tracing, m_eval, m_test, m_train, m_wandb,
            m_check, mock_cfg, mock_args):
        """When trace in stages and run_tracing=True, tracing.main is called."""
        import orthrus

        mock_args.stages = "preprocess,train,test,evaluate,trace"
        mock_cfg.pipeline.run_tracing = True
        m_wandb.run = MagicMock()  # active wandb run

        result = orthrus.main(mock_cfg, mock_args)

        m_tracing.main.assert_called_once_with(mock_cfg)
        assert "time_total" in result

    @requires_torch
    @patch('orthrus._check_artifact_prerequisites')
    @patch('orthrus.wandb')
    @patch('orthrus.orthrus_gnn_training')
    @patch('orthrus.orthrus_gnn_testing')
    @patch('orthrus.evaluation')
    @patch('orthrus.tracing')
    @patch('orthrus.build_orthrus_graphs')
    @patch('orthrus.build_feature_word2vec')
    @patch('orthrus.embed_edges_feature_word2vec')
    def test_trace_not_called_when_stages_omits_trace(
            self, m_embed, m_w2v, m_graphs,
            m_tracing, m_eval, m_test, m_train, m_wandb,
            m_check, mock_cfg, mock_args):
        """When trace not in stages, tracing.main is NOT called."""
        import orthrus

        mock_args.stages = "preprocess,train,test,evaluate"
        mock_cfg.pipeline.run_tracing = True
        m_wandb.run = MagicMock()

        orthrus.main(mock_cfg, mock_args)

        m_tracing.main.assert_not_called()

    @requires_torch
    @patch('orthrus._check_artifact_prerequisites')
    @patch('orthrus.wandb')
    @patch('orthrus.orthrus_gnn_training')
    @patch('orthrus.orthrus_gnn_testing')
    @patch('orthrus.evaluation')
    @patch('orthrus.tracing')
    @patch('orthrus.build_orthrus_graphs')
    @patch('orthrus.build_feature_word2vec')
    @patch('orthrus.embed_edges_feature_word2vec')
    def test_trace_not_called_when_run_tracing_false(
            self, m_embed, m_w2v, m_graphs,
            m_tracing, m_eval, m_test, m_train, m_wandb,
            m_check, mock_cfg, mock_args):
        """When run_tracing=False (pipeline default or --skip-tracing), tracing.main NOT called."""
        import orthrus

        mock_args.stages = "preprocess,train,test,evaluate,trace"
        mock_cfg.pipeline.run_tracing = False
        m_wandb.run = MagicMock()

        orthrus.main(mock_cfg, mock_args)

        m_tracing.main.assert_not_called()

    @requires_torch
    @patch('orthrus._check_artifact_prerequisites')
    @patch('orthrus.wandb')
    @patch('orthrus.orthrus_gnn_training')
    @patch('orthrus.orthrus_gnn_testing')
    @patch('orthrus.evaluation')
    @patch('orthrus.tracing')
    @patch('orthrus.build_orthrus_graphs')
    @patch('orthrus.build_feature_word2vec')
    @patch('orthrus.embed_edges_feature_word2vec')
    def test_preprocess_calls_three_substages_in_order(
            self, m_embed, m_w2v, m_graphs,
            m_tracing, m_eval, m_test, m_train, m_wandb,
            m_check, mock_cfg, mock_args):
        """--stages preprocess calls build_graphs, embed_nodes, embed_edges in order."""
        import orthrus

        mock_args.stages = "preprocess"
        mock_cfg.pipeline.run_tracing = True
        m_wandb.run = MagicMock()

        orthrus.main(mock_cfg, mock_args)

        m_graphs.main.assert_called_once_with(mock_cfg)
        m_w2v.main.assert_called_once_with(mock_cfg)
        m_embed.main.assert_called_once_with(mock_cfg)
        # train/test/eval/trace should NOT be called
        m_train.main.assert_not_called()
        m_test.main.assert_not_called()
        m_eval.main.assert_not_called()
        m_tracing.main.assert_not_called()

    @requires_torch
    @patch('orthrus._check_artifact_prerequisites')
    @patch('orthrus.wandb')
    @patch('orthrus.orthrus_gnn_training')
    @patch('orthrus.orthrus_gnn_testing')
    @patch('orthrus.evaluation')
    @patch('orthrus.tracing')
    @patch('orthrus.build_orthrus_graphs')
    @patch('orthrus.build_feature_word2vec')
    @patch('orthrus.embed_edges_feature_word2vec')
    def test_train_only_calls_training(
            self, m_embed, m_w2v, m_graphs,
            m_tracing, m_eval, m_test, m_train, m_wandb,
            m_check, mock_cfg, mock_args):
        """--stages preprocess,train,test,evaluate calls each detection stage once."""
        import orthrus

        mock_args.stages = "preprocess,train,test,evaluate"
        mock_cfg.pipeline.run_tracing = True
        m_wandb.run = MagicMock()

        orthrus.main(mock_cfg, mock_args)

        m_train.main.assert_called_once_with(mock_cfg)
        m_test.main.assert_called_once_with(mock_cfg)
        m_eval.main.assert_called_once_with(mock_cfg)

    @requires_torch
    @patch('orthrus._check_artifact_prerequisites')
    @patch('orthrus.wandb')
    @patch('orthrus.orthrus_gnn_training')
    @patch('orthrus.orthrus_gnn_testing')
    @patch('orthrus.evaluation')
    @patch('orthrus.tracing')
    @patch('orthrus.build_orthrus_graphs')
    @patch('orthrus.build_feature_word2vec')
    @patch('orthrus.embed_edges_feature_word2vec')
    def test_stages_test_evaluate_calls_test_and_eval_not_train(
            self, m_embed, m_w2v, m_graphs,
            m_tracing, m_eval, m_test, m_train, m_wandb,
            m_check, mock_cfg, mock_args):
        """--stages test,evaluate (after train) calls test+eval but not train or preprocess."""
        import orthrus

        mock_args.stages = "test,evaluate"
        mock_cfg.pipeline.run_tracing = True
        m_wandb.run = MagicMock()

        orthrus.main(mock_cfg, mock_args)

        m_train.main.assert_not_called()
        m_test.main.assert_called_once_with(mock_cfg)
        m_eval.main.assert_called_once_with(mock_cfg)

    @requires_torch
    @patch('orthrus._check_artifact_prerequisites')
    @patch('orthrus.wandb')
    @patch('orthrus.orthrus_gnn_training')
    @patch('orthrus.orthrus_gnn_testing')
    @patch('orthrus.evaluation')
    @patch('orthrus.tracing')
    @patch('orthrus.build_orthrus_graphs')
    @patch('orthrus.build_feature_word2vec')
    @patch('orthrus.embed_edges_feature_word2vec')
    def test_default_no_stages_no_run_from_training_includes_trace(
            self, m_embed, m_w2v, m_graphs,
            m_tracing, m_eval, m_test, m_train, m_wandb,
            m_check, mock_cfg, mock_args):
        """Default (no --stages, no --run_from_training) runs full pipeline with trace."""
        import orthrus

        mock_args.stages = None
        mock_args.run_from_training = False
        mock_cfg.pipeline.run_tracing = True
        m_wandb.run = MagicMock()

        orthrus.main(mock_cfg, mock_args)

        m_graphs.main.assert_called()
        m_train.main.assert_called()
        m_test.main.assert_called()
        m_eval.main.assert_called()
        m_tracing.main.assert_called()  # trace included by default


# -------------------------------------------------------------------------- #
# CONSTANTS
# -------------------------------------------------------------------------- #
def test_standard_stages_contains_five():
    assert set(STANDARD_STAGES) == {"preprocess", "train", "test", "evaluate", "trace"}


def test_valid_stages_matches_standard():
    assert VALID_STAGES == set(STANDARD_STAGES)
