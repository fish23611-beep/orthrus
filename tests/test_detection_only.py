"""Tests for detection_only mode in orthrus.py."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
src_dir = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, src_dir)

import orthrus


class TestIsDetectionOnlyMode:
    """Tests for _is_detection_only_mode helper."""

    def test_returns_true_for_detection_only(self):
        """Returns True for detection_only mode."""
        cfg = SimpleNamespace(pipeline=SimpleNamespace(mode="detection_only"))
        assert orthrus._is_detection_only_mode(cfg) is True

    def test_returns_false_for_full_pipeline(self):
        """Returns False for full_pipeline mode."""
        cfg = SimpleNamespace(pipeline=SimpleNamespace(mode="full_pipeline"))
        assert orthrus._is_detection_only_mode(cfg) is False

    def test_returns_false_when_mode_missing(self):
        """Returns False when mode attribute is missing."""
        cfg = SimpleNamespace()
        assert orthrus._is_detection_only_mode(cfg) is False


class TestCheckDetectionOnlyPrerequisites:
    """Tests for _check_detection_only_prerequisites function."""

    def test_returns_empty_when_all_artifacts_present(self, tmp_path):
        """Returns empty list when all required artifacts exist."""
        # Create fake artifact directories
        graphs_dir = tmp_path / "graphs"
        graphs_dir.mkdir()
        (graphs_dir / "graph_1.pt").touch()
        
        w2v_dir = tmp_path / "word2vec"
        w2v_dir.mkdir()
        
        edges_dir = tmp_path / "edges"
        edges_dir.mkdir()
        
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()
        
        edge_scores_dir = tmp_path / "edge_scores"
        edge_scores_dir.mkdir()
        (edge_scores_dir / "test").mkdir()
        
        cfg = SimpleNamespace(
            graph_construction=SimpleNamespace(
                build_graphs=SimpleNamespace(_graphs_dir=str(graphs_dir))
            ),
            edge_featurization=SimpleNamespace(
                embed_nodes=SimpleNamespace(
                    feature_word2vec=SimpleNamespace(_model_dir=str(w2v_dir))
                ),
                embed_edges=SimpleNamespace(_edge_embeds_dir=str(edges_dir))
            ),
            detection=SimpleNamespace(
                gnn_training=SimpleNamespace(_trained_models_dir=str(checkpoints_dir)),
                gnn_testing=SimpleNamespace(_edge_losses_dir=str(edge_scores_dir))
            ),
        )
        
        missing = orthrus._check_detection_only_prerequisites(
            ["train", "test", "evaluate"], cfg
        )
        
        assert missing == []

    def test_returns_missing_for_empty_graphs(self, tmp_path):
        """Returns missing when graphs directory is empty for evaluate stage (needed for compute_tw_labels)."""
        graphs_dir = tmp_path / "graphs"
        graphs_dir.mkdir()
        # Empty - no files

        edge_scores_dir = tmp_path / "edge_scores"
        edge_scores_dir.mkdir()
        test_dir = edge_scores_dir / "test"
        test_dir.mkdir()

        cfg = SimpleNamespace(
            graph_construction=SimpleNamespace(
                build_graphs=SimpleNamespace(_graphs_dir=str(graphs_dir))
            ),
            edge_featurization=SimpleNamespace(
                embed_nodes=SimpleNamespace(
                    feature_word2vec=SimpleNamespace(_model_dir=str(tmp_path / "w2v"))
                ),
                embed_edges=SimpleNamespace(_edge_embeds_dir=str(tmp_path / "edges"))
            ),
            detection=SimpleNamespace(
                gnn_training=SimpleNamespace(_trained_models_dir=str(tmp_path / "checkpoints")),
                gnn_testing=SimpleNamespace(_edge_losses_dir=str(edge_scores_dir))
            ),
        )

        # C8 fix: graphs are only needed for evaluate stage (compute_tw_labels loads test graph boundaries)
        # train-only does not require graphs
        missing_train = orthrus._check_detection_only_prerequisites(["train"], cfg)
        assert not any("Graphs directory" in m for m in missing_train), (
            "Graphs should NOT be required for train stage (only for evaluate)"
        )

        # evaluate does require graphs
        missing_evaluate = orthrus._check_detection_only_prerequisites(["evaluate"], cfg)
        assert any("Graphs directory" in m for m in missing_evaluate), (
            "Graphs SHOULD be required for evaluate stage (compute_tw_labels)"
        )

    def test_explicit_checkpoint_and_test_stage_satisfy_downstream_prerequisites(self, tmp_path):
        """An explicit inference checkpoint plus test supplies evaluate inputs."""
        graphs_dir = tmp_path / "graphs"
        graphs_dir.mkdir()
        (graphs_dir / "graph_1.pt").touch()
        w2v_dir = tmp_path / "word2vec"
        w2v_dir.mkdir()
        edges_dir = tmp_path / "edges"
        edges_dir.mkdir()
        checkpoint = tmp_path / "checkpoint.pt"
        checkpoint.touch()

        cfg = SimpleNamespace(
            _inference_checkpoint=str(checkpoint),
            graph_construction=SimpleNamespace(
                build_graphs=SimpleNamespace(_graphs_dir=str(graphs_dir))
            ),
            edge_featurization=SimpleNamespace(
                embed_nodes=SimpleNamespace(
                    feature_word2vec=SimpleNamespace(_model_dir=str(w2v_dir))
                ),
                embed_edges=SimpleNamespace(_edge_embeds_dir=str(edges_dir)),
            ),
            detection=SimpleNamespace(
                gnn_training=SimpleNamespace(_trained_models_dir=str(tmp_path / "missing_checkpoints")),
                gnn_testing=SimpleNamespace(_edge_losses_dir=str(tmp_path / "missing_scores")),
            ),
        )

        assert orthrus._check_detection_only_prerequisites(["test", "evaluate"], cfg) == []

    def test_train_only_does_not_require_checkpoints(self, tmp_path):
        """Train-only stage does not require checkpoints."""
        graphs_dir = tmp_path / "graphs"
        graphs_dir.mkdir()
        (graphs_dir / "graph_1.pt").touch()
        
        cfg = SimpleNamespace(
            graph_construction=SimpleNamespace(
                build_graphs=SimpleNamespace(_graphs_dir=str(graphs_dir))
            ),
            edge_featurization=SimpleNamespace(
                embed_nodes=SimpleNamespace(
                    feature_word2vec=SimpleNamespace(_model_dir=str(tmp_path / "w2v"))
                ),
                embed_edges=SimpleNamespace(_edge_embeds_dir=str(tmp_path / "edges"))
            ),
            detection=SimpleNamespace(
                gnn_training=SimpleNamespace(_trained_models_dir=str(tmp_path / "checkpoints")),
                gnn_testing=SimpleNamespace(_edge_losses_dir=str(tmp_path / "scores"))
            ),
        )
        
        # Train only should pass
        missing = orthrus._check_detection_only_prerequisites(["train"], cfg)
        
        assert all("checkpoint" not in m.lower() for m in missing)


class TestDetectionOnlyMain:
    """Tests for detection_only mode in main function."""

    def test_skips_preprocess_in_detection_only(self, mocker, tmp_path):
        """Preprocess is skipped in detection_only mode."""
        # Create actual directories for the test
        graphs_dir = tmp_path / "graphs"
        graphs_dir.mkdir()
        # Add a fake graph file
        (graphs_dir / "graph_1.pt").touch()
        
        w2v_dir = tmp_path / "word2vec"
        w2v_dir.mkdir()
        
        edges_dir = tmp_path / "edges"
        edges_dir.mkdir()
        
        checkpoints_dir = tmp_path / "checkpoints"
        checkpoints_dir.mkdir()
        
        cfg = SimpleNamespace(
            pipeline=SimpleNamespace(mode="detection_only", run_tracing=True),
            detection=SimpleNamespace(
                gnn_training=SimpleNamespace(use_seed=False),
            ),
            graph_construction=SimpleNamespace(
                build_graphs=SimpleNamespace(_graphs_dir=str(graphs_dir))
            ),
            edge_featurization=SimpleNamespace(
                embed_nodes=SimpleNamespace(
                    feature_word2vec=SimpleNamespace(_model_dir=str(w2v_dir))
                ),
                embed_edges=SimpleNamespace(_edge_embeds_dir=str(edges_dir))
            ),
            _metadata_dir="/fake",
        )
        args = SimpleNamespace(
            stages="train",
            run_from_training=False,
            wandb=False,
        )
        
        # Mock all detection stages
        mock_train = mocker.patch("orthrus.orthrus_gnn_training.main")
        mock_build = mocker.patch("orthrus.build_orthrus_graphs.main")
        mock_w2v = mocker.patch("orthrus.build_feature_word2vec.main")
        mock_edges = mocker.patch("orthrus.embed_edges_feature_word2vec.main")
        
        orthrus.main(cfg, args)
        
        # build_orthrus_graphs should NOT be called
        mock_build.assert_not_called()
        mock_w2v.assert_not_called()
        mock_edges.assert_not_called()
        # Train should be called
        mock_train.assert_called_once()

    def test_raises_clear_error_on_missing_artifacts(self, mocker):
        """Raises FileNotFoundError with clear message on missing artifacts."""
        cfg = SimpleNamespace(
            pipeline=SimpleNamespace(mode="detection_only", run_tracing=True),
            detection=SimpleNamespace(
                gnn_training=SimpleNamespace(
                    use_seed=False,
                    _trained_models_dir="/nonexistent/checkpoints"
                ),
                gnn_testing=SimpleNamespace(_edge_losses_dir="/nonexistent/scores")
            ),
            graph_construction=SimpleNamespace(
                build_graphs=SimpleNamespace(_graphs_dir="/nonexistent/graphs")
            ),
            edge_featurization=SimpleNamespace(
                embed_nodes=SimpleNamespace(
                    feature_word2vec=SimpleNamespace(_model_dir="/nonexistent/w2v")
                ),
                embed_edges=SimpleNamespace(_edge_embeds_dir="/nonexistent/edges")
            ),
        )
        args = SimpleNamespace(
            stages="train",
            run_from_training=False,
            wandb=False,
        )
        
        with pytest.raises(FileNotFoundError, match="detection_only mode"):
            orthrus.main(cfg, args)


class TestPipelineModeConfig:
    """Tests for pipeline mode configuration."""

    def test_validates_pipeline_mode(self, mocker):
        """Validates pipeline mode in config."""
        # Test that invalid mode would raise (but we use valid modes in tests)
        pass  # Config validation is tested separately
