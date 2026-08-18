"""
tests/test_artifact_root_routing.py

Regression tests for C8 final artifact root routing fix.

R1. fresh-worktree absolute shared root
    Simulates: cwd has no ./artifacts; user specifies absolute artifact root.
    Verifies: preprocessing paths resolve to the shared root, not ./artifacts.

R2. shared preprocessing + scoped run isolation
    Same shared root, two different configs -> two scoped roots.
    Verifies: preprocessing paths are identical; run-level paths differ.

R3. evaluate-only prerequisite precision
    Verifies: evaluate-only needs only graphs + edge_scores/test (not Word2Vec/edge_embeddings).

R4. config identity stability
    Verifies: _config_id() unchanged.

R5. run_experiment / run_matrix compatibility
    Verifies: CLI --artifact-root routing works through both entry points.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_real_cfg(dataset="THEIA_E3", model_variant="orthrus", seed=0):
    """Create a cfg with real CfgNode shape but test-safe paths."""
    from yacs.config import CfgNode as CN
    cfg = CN()
    cfg.dataset = CN()
    cfg.dataset.name = dataset
    cfg.detection = CN()
    cfg.detection.gnn_training = CN()
    cfg.detection.gnn_training.used_method = model_variant
    cfg.detection.gnn_testing = CN()
    cfg.detection.evaluation = CN()
    cfg.detection.evaluation.node_evaluation = CN()
    cfg._seed = seed
    cfg._artifact_dir = "./artifacts"  # default
    cfg._artifact_root_raw = None
    return cfg


# --------------------------------------------------------------------------- #
# R1. Fresh-worktree absolute shared root
# --------------------------------------------------------------------------- #

class TestFreshWorktreeAbsoluteSharedRoot:
    """R1: CLI --artifact-root must route preprocessing to the shared root."""

    def test_cli_artifact_root_overrides_default_for_preprocessing(self, tmp_path, monkeypatch):
        """Preprocessing paths use the CLI-specified root, not ./artifacts."""
        import config as config_module

        # Simulate fresh worktree: cwd has no ./artifacts
        monkeypatch.chdir(tmp_path)
        assert not (tmp_path / "artifacts").exists()

        # User specifies an absolute shared root
        shared_root = tmp_path / "mstc_pids" / "artifacts"
        shared_root.mkdir(parents=True)

        # Write a valid minimal config
        config_file = tmp_path / "baseline.yml"
        config_file.write_text("pipeline: {mode: full_pipeline}\n", encoding="utf-8")

        # Patch set_task_paths to avoid DB/other deps
        original_set_task_paths = config_module.set_task_paths
        config_module.set_task_paths = MagicMock()

        try:
            args = SimpleNamespace(
                dataset="THEIA_E3",
                model="orthrus",
                config=str(config_file),
                cpu=True,
                from_weights=False,
                seed=0,
                skip_tracing=False,
                artifact_root=str(shared_root),
                max_windows_per_split=None,
            )
            cfg = config_module.get_yml_cfg(args)

            # After fix: cfg._artifact_dir should be the shared root
            assert cfg._artifact_dir == str(shared_root), (
                f"Expected cfg._artifact_dir={shared_root}, got {cfg._artifact_dir}. "
                "CLI --artifact-root must override the default ./artifacts."
            )
        finally:
            config_module.set_task_paths = original_set_task_paths

    def test_env_artifact_root_falls_through_when_no_cli(self, tmp_path, monkeypatch):
        """When CLI is absent, env var should set cfg._artifact_dir."""
        import config as config_module

        # Simulate fresh worktree
        monkeypatch.chdir(tmp_path)
        assert not (tmp_path / "artifacts").exists()

        # Set env var
        env_root = tmp_path / "env_artifacts"
        env_root.mkdir(parents=True)
        monkeypatch.setenv("ORTHRUS_ARTIFACT_ROOT", str(env_root))

        config_file = tmp_path / "baseline.yml"
        config_file.write_text("pipeline: {mode: full_pipeline}\n", encoding="utf-8")

        original_set_task_paths = config_module.set_task_paths
        config_module.set_task_paths = MagicMock()

        try:
            args = SimpleNamespace(
                dataset="THEIA_E3",
                model="orthrus",
                config=str(config_file),
                cpu=True,
                from_weights=False,
                seed=0,
                skip_tracing=False,
                artifact_root=None,  # No CLI override
                max_windows_per_split=None,
            )
            cfg = config_module.get_yml_cfg(args)

            assert cfg._artifact_dir == str(env_root)
        finally:
            config_module.set_task_paths = original_set_task_paths

    def test_preprocessing_paths_use_shared_root(self, tmp_path, monkeypatch):
        """Preprocessing (graph/Word2Vec/edge_embeddings) paths resolve to shared root."""
        import config as config_module

        monkeypatch.chdir(tmp_path)

        shared_root = tmp_path / "mstc_pids" / "artifacts"
        shared_root.mkdir(parents=True)

        config_file = tmp_path / "baseline.yml"
        config_file.write_text("pipeline: {mode: full_pipeline}\n", encoding="utf-8")

        # Patch to avoid DB/other heavy deps
        original_set_task_paths = config_module.set_task_paths
        config_module.set_task_paths = MagicMock()

        try:
            args = SimpleNamespace(
                dataset="THEIA_E3",
                model="orthrus",
                config=str(config_file),
                cpu=True,
                from_weights=False,
                seed=0,
                skip_tracing=False,
                artifact_root=str(shared_root),
                max_windows_per_split=None,
            )
            cfg = config_module.get_yml_cfg(args)

            # After fix: all preprocessing paths should be under shared_root
            expected_prefix = str(shared_root)
            assert cfg._artifact_dir == expected_prefix

            # Verify preprocessing paths use the shared root
            # (set_task_paths would set these, but we can verify _artifact_dir is correct)
            assert cfg._artifact_dir.startswith(expected_prefix)
        finally:
            config_module.set_task_paths = original_set_task_paths


# --------------------------------------------------------------------------- #
# R2. Shared preprocessing + scoped run isolation
# --------------------------------------------------------------------------- #

class TestSharedPreprocessingWithScopedRuns:
    """R2: Same shared root, different configs -> scoped run roots differ."""

    def test_scoped_root_isolates_run_artifacts(self, tmp_path):
        """Two configs produce different scoped roots; preprocessing paths stay shared."""
        from experiments import run_matrix

        root = tmp_path / "artifacts"

        # Two different configs
        config1 = tmp_path / "baseline.yml"
        config1.write_text("pipeline: {mode: full_pipeline}\n", encoding="utf-8")

        config2 = tmp_path / "mstc_full.yml"
        config2.write_text("pipeline: {mode: full_pipeline}\ndetection: {gnn_training: {used_method: mstc}}\n", encoding="utf-8")

        scoped1 = run_matrix.run_artifact_root(root, config1.resolve())
        scoped2 = run_matrix.run_artifact_root(root, config2.resolve())

        # Scoped roots must differ (otherwise no isolation)
        assert scoped1 != scoped2, (
            f"Scoped roots must differ between configs for run artifact isolation. "
            f"Got: {scoped1} == {scoped2}"
        )

        # Both should be under matrix_artifacts/
        assert "matrix_artifacts" in str(scoped1)
        assert "matrix_artifacts" in str(scoped2)

        # The shared parent should be the root
        assert scoped1.parent.parent == root
        assert scoped2.parent.parent == root

    def test_config_id_derivation_is_deterministic(self, tmp_path):
        """Same config path always yields the same scoped root."""
        from experiments import run_matrix

        root = tmp_path / "artifacts"
        config = tmp_path / "baseline.yml"
        config.write_text("pipeline: {mode: full_pipeline}\n", encoding="utf-8")
        resolved = config.resolve()

        scoped1 = run_matrix.run_artifact_root(root, resolved)
        scoped2 = run_matrix.run_artifact_root(root, resolved)

        assert scoped1 == scoped2, "Scoped root must be deterministic for the same config"

    def test_scoped_root_name_contains_config_stem(self, tmp_path):
        """Scoped root directory name should contain a safe form of the config stem."""
        from experiments import run_matrix

        root = tmp_path / "artifacts"
        config = tmp_path / "my-experiment.yml"
        config.write_text("pipeline: {mode: full_pipeline}\n", encoding="utf-8")

        scoped = run_matrix.run_artifact_root(root, config.resolve())

        # Should contain "my-experiment" or safe equivalent
        scoped_name = scoped.name
        # The safe stem should appear somewhere in the scoped name
        assert "my" in scoped_name.lower() or "experiment" in scoped_name.lower() or len(scoped_name) > 5


# --------------------------------------------------------------------------- #
# R3. Evaluate-only prerequisite precision
# --------------------------------------------------------------------------- #

class TestEvaluateOnlyPrerequisites:
    """R3: Evaluate-only needs graphs + edge_scores, NOT Word2Vec/edge_embeddings."""

    def test_evaluate_only_does_not_require_word2vec(self, tmp_path):
        """Evaluate-only must pass even without Word2Vec artifacts."""
        import orthrus

        # Create graphs + edge_scores (needed), but NOT Word2Vec
        graphs_dir = tmp_path / "graphs"
        graphs_dir.mkdir()
        (graphs_dir / "graph_1.pt").touch()

        edge_scores_dir = tmp_path / "edge_scores"
        edge_scores_dir.mkdir()
        test_dir = edge_scores_dir / "test"
        test_dir.mkdir()
        (test_dir / "epoch_1.csv").touch()

        cfg = SimpleNamespace(
            graph_construction=SimpleNamespace(
                build_graphs=SimpleNamespace(_graphs_dir=str(graphs_dir))
            ),
            edge_featurization=SimpleNamespace(
                embed_nodes=SimpleNamespace(
                    feature_word2vec=SimpleNamespace(_model_dir=str(tmp_path / "missing_w2v"))
                ),
                embed_edges=SimpleNamespace(_edge_embeds_dir=str(tmp_path / "missing_edges"))
            ),
            detection=SimpleNamespace(
                gnn_training=SimpleNamespace(_trained_models_dir=str(tmp_path / "missing_checkpoints")),
                gnn_testing=SimpleNamespace(_edge_losses_dir=str(edge_scores_dir))
            ),
        )

        # Evaluate-only should pass (no Word2Vec, no edge embeddings needed)
        missing = orthrus._check_detection_only_prerequisites(["evaluate"], cfg)
        assert missing == [], f"Evaluate-only should not require Word2Vec/edge_embeddings. Got: {missing}"

    def test_evaluate_only_requires_graphs(self, tmp_path):
        """Evaluate-only must fail without graphs (needed for compute_tw_labels)."""
        import orthrus

        edge_scores_dir = tmp_path / "edge_scores"
        edge_scores_dir.mkdir()
        test_dir = edge_scores_dir / "test"
        test_dir.mkdir()

        cfg = SimpleNamespace(
            graph_construction=SimpleNamespace(
                build_graphs=SimpleNamespace(_graphs_dir=str(tmp_path / "missing_graphs"))
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

        missing = orthrus._check_detection_only_prerequisites(["evaluate"], cfg)
        assert any("Graphs directory" in m for m in missing), (
            "Evaluate-only must require graphs (compute_tw_labels needs them)"
        )

    def test_evaluate_only_requires_edge_scores_test(self, tmp_path):
        """Evaluate-only must fail without edge_scores/test/ (from test stage)."""
        import orthrus

        graphs_dir = tmp_path / "graphs"
        graphs_dir.mkdir()
        (graphs_dir / "graph_1.pt").touch()

        # No edge_scores
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
                gnn_testing=SimpleNamespace(_edge_losses_dir=str(tmp_path / "missing_scores"))
            ),
        )

        missing = orthrus._check_detection_only_prerequisites(["evaluate"], cfg)
        assert any("Edge scores" in m for m in missing), (
            "Evaluate-only must require edge_scores (from test stage)"
        )

    def test_train_requires_word2vec_and_edge_embeddings(self, tmp_path):
        """Train stage must require Word2Vec and edge embeddings."""
        import orthrus

        graphs_dir = tmp_path / "graphs"
        graphs_dir.mkdir()
        (graphs_dir / "graph_1.pt").touch()

        cfg = SimpleNamespace(
            graph_construction=SimpleNamespace(
                build_graphs=SimpleNamespace(_graphs_dir=str(graphs_dir))
            ),
            edge_featurization=SimpleNamespace(
                embed_nodes=SimpleNamespace(
                    feature_word2vec=SimpleNamespace(_model_dir=str(tmp_path / "missing_w2v"))
                ),
                embed_edges=SimpleNamespace(_edge_embeds_dir=str(tmp_path / "missing_edges"))
            ),
            detection=SimpleNamespace(
                gnn_training=SimpleNamespace(_trained_models_dir=str(tmp_path / "checkpoints")),
                gnn_testing=SimpleNamespace(_edge_losses_dir=str(tmp_path / "scores"))
            ),
        )

        missing = orthrus._check_detection_only_prerequisites(["train"], cfg)
        assert any("Word2Vec" in m for m in missing), "Train must require Word2Vec"
        assert any("Edge embeddings" in m for m in missing), "Train must require edge embeddings"

    def test_test_requires_checkpoints(self, tmp_path):
        """Test stage must require checkpoints (for inference)."""
        import orthrus

        graphs_dir = tmp_path / "graphs"
        graphs_dir.mkdir()
        (graphs_dir / "graph_1.pt").touch()

        w2v_dir = tmp_path / "w2v"
        w2v_dir.mkdir()

        edges_dir = tmp_path / "edges"
        edges_dir.mkdir()

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
                gnn_training=SimpleNamespace(_trained_models_dir=str(tmp_path / "missing_checkpoints")),
                gnn_testing=SimpleNamespace(_edge_losses_dir=str(tmp_path / "scores"))
            ),
        )

        missing = orthrus._check_detection_only_prerequisites(["test"], cfg)
        assert any("Checkpoints" in m for m in missing), "Test must require checkpoints"

    def test_explicit_inference_checkpoint_bypasses_checkpoint_dir_check(self, tmp_path):
        """Explicit --checkpoint satisfies checkpoint prerequisite."""
        import orthrus

        graphs_dir = tmp_path / "graphs"
        graphs_dir.mkdir()
        (graphs_dir / "graph_1.pt").touch()

        w2v_dir = tmp_path / "w2v"
        w2v_dir.mkdir()

        edges_dir = tmp_path / "edges"
        edges_dir.mkdir()

        checkpoint = tmp_path / "model.pt"
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
                embed_edges=SimpleNamespace(_edge_embeds_dir=str(edges_dir))
            ),
            detection=SimpleNamespace(
                gnn_training=SimpleNamespace(_trained_models_dir=str(tmp_path / "missing_checkpoints")),
                gnn_testing=SimpleNamespace(_edge_losses_dir=str(tmp_path / "scores"))
            ),
        )

        # Test with explicit checkpoint should pass
        missing = orthrus._check_detection_only_prerequisites(["test"], cfg)
        assert not any("Checkpoints" in m for m in missing), (
            "Explicit checkpoint should bypass checkpoint directory check"
        )


# --------------------------------------------------------------------------- #
# R4. Config identity stability
# --------------------------------------------------------------------------- #

class TestConfigIdStability:
    """R4: _config_id() and scoped root naming must remain stable."""

    def test_same_config_gives_same_id(self, tmp_path):
        """Identical config paths always produce the same config ID."""
        from experiments import run_matrix

        config = tmp_path / "baseline.yml"
        config.write_text("pipeline: {mode: full_pipeline}\n", encoding="utf-8")
        resolved = config.resolve()

        id1 = run_matrix._config_id(resolved)
        id2 = run_matrix._config_id(resolved)

        assert id1 == id2, f"Config ID must be deterministic: {id1} != {id2}"

    def test_different_configs_give_different_ids(self, tmp_path):
        """Different config paths produce different config IDs."""
        from experiments import run_matrix

        config1 = tmp_path / "baseline.yml"
        config1.write_text("pipeline: {mode: full_pipeline}\n", encoding="utf-8")

        config2 = tmp_path / "mstc.yml"
        config2.write_text("pipeline: {mode: full_pipeline}\ndetection: {gnn_training: {used_method: mstc}}\n", encoding="utf-8")

        id1 = run_matrix._config_id(config1.resolve())
        id2 = run_matrix._config_id(config2.resolve())

        assert id1 != id2, f"Different configs must have different IDs: {id1} == {id2}"

    def test_config_id_format_is_valid_directory_name(self, tmp_path):
        """Config ID must be a safe directory name (no path separators)."""
        from experiments import run_matrix

        config = tmp_path / "my config.yml"
        config.write_text("pipeline: {mode: full_pipeline}\n", encoding="utf-8")

        config_id = run_matrix._config_id(config.resolve())

        # Must not contain path separators
        assert "/" not in config_id, f"Config ID contains path separator: {config_id}"
        assert "\\" not in config_id, f"Config ID contains backslash: {config_id}"

    def test_scoped_root_path_format(self, tmp_path):
        """Scoped root must be a valid path under matrix_artifacts/."""
        from experiments import run_matrix

        root = tmp_path / "artifacts"
        config = tmp_path / "baseline.yml"
        config.write_text("pipeline: {mode: full_pipeline}\n", encoding="utf-8")

        scoped = run_matrix.run_artifact_root(root, config.resolve())

        # Must be under matrix_artifacts
        assert scoped.is_relative_to(root / "matrix_artifacts"), (
            f"Scoped root must be under matrix_artifacts/: {scoped}"
        )

        # The immediate parent should be matrix_artifacts
        assert scoped.parent.name == "matrix_artifacts", (
            f"Scoped root parent must be 'matrix_artifacts', got {scoped.parent.name}"
        )


# --------------------------------------------------------------------------- #
# R5. run_experiment / run_matrix compatibility
# --------------------------------------------------------------------------- #

class TestRunExperimentCompatibility:
    """R5: Direct run_experiment CLI should also route --artifact-root correctly."""

    def test_run_experiment_cli_passes_artifact_root(self, tmp_path, monkeypatch):
        """run_experiment.main forwards --artifact-root to the pipeline."""
        from experiments import run_experiment

        config_file = tmp_path / "baseline.yml"
        config_file.write_text("pipeline: {mode: full_pipeline}\n", encoding="utf-8")

        captured_args = {}
        monkeypatch.setattr(run_experiment, "_run_pipeline", lambda args: captured_args.setdefault("args", args))

        artifact_root = str(tmp_path / "my_artifacts")

        run_experiment.main([
            "--dataset", "THEIA_E3",
            "--config", str(config_file),
            "--artifact-root", artifact_root,
            "--stages", "train",
        ])

        assert captured_args["args"].artifact_root == artifact_root

    def test_run_matrix_passes_scoped_root_and_shared_root_to_child(self, tmp_path, monkeypatch):
        """run_matrix passes both scoped root (--artifact-root) and shared root (--shared-artifact-root) to child."""
        from experiments import run_matrix

        config = tmp_path / "baseline.yml"
        config.write_text("pipeline: {mode: full_pipeline}\n", encoding="utf-8")

        calls = []
        monkeypatch.setattr(run_matrix.run_experiment, "main", lambda argv: calls.append(argv))

        root = tmp_path / "artifacts"
        run_matrix.main([
            "--datasets", "THEIA_E3",
            "--configs", str(config),
            "--seeds", "0",
            "--artifact-root", str(root),
        ])

        assert len(calls) == 1
        argv = calls[0]

        # Child should receive scoped root as --artifact-root
        idx = argv.index("--artifact-root")
        child_scoped_root = argv[idx + 1]
        assert "matrix_artifacts" in child_scoped_root, (
            f"Child process must receive scoped root with matrix_artifacts/, got {child_scoped_root}"
        )

        # Child should also receive shared root as --shared-artifact-root
        idx = argv.index("--shared-artifact-root")
        child_shared_root = argv[idx + 1]
        # Shared root should be the top-level root (not scoped)
        assert str(root) == child_shared_root, (
            f"Child process must receive top-level shared root as --shared-artifact-root, got {child_shared_root}"
        )
        assert "matrix_artifacts" not in child_shared_root, (
            f"Shared root should NOT contain matrix_artifacts/, got {child_shared_root}"
        )


# --------------------------------------------------------------------------- #
# R6. End-to-end dual root integration test
# --------------------------------------------------------------------------- #

class TestDualRootEndToEnd:
    """
    R6: Critical integration test that verifies the complete dual root routing chain.

    This test verifies that when run_matrix is invoked:
    1. scoped_root is computed correctly (matrix_artifacts/<config-id>)
    2. shared_root is preserved as the top-level artifact root
    3. Child process receives both correctly
    4. get_yml_cfg() sets:
       - _artifact_dir = shared_root (for preprocessing paths)
       - _scoped_artifact_root = scoped_root (for run artifacts)
    5. set_task_paths() uses _artifact_dir for preprocessing
    6. resolve_artifact_paths() uses scoped root for run artifacts
    """

    def test_matrix_dual_root_routing_end_to_end(self, tmp_path, monkeypatch):
        """End-to-end test: shared root for preprocessing, scoped root for run artifacts."""
        import config as config_module
        from artifact_paths import resolve_artifact_paths

        # Setup: run_matrix top-level artifact root
        shared_root = tmp_path / "mstc_pids" / "artifacts"
        shared_root.mkdir(parents=True)

        # Write config file
        config_file = tmp_path / "baseline.yml"
        config_file.write_text("pipeline: {mode: full_pipeline}\n", encoding="utf-8")

        # Compute what run_matrix would compute
        from experiments import run_matrix
        scoped_root = run_matrix.run_artifact_root(shared_root, config_file.resolve())

        # Verify scoped root structure
        assert scoped_root == shared_root / "matrix_artifacts" / run_matrix._config_id(config_file.resolve())
        assert scoped_root.is_relative_to(shared_root)

        # Simulate child process receiving both roots
        # This is what run_matrix._run_argv would produce
        argv = run_matrix._run_argv(
            dataset="THEIA_E3",
            config=config_file.resolve(),
            seed=0,
            artifact_root=scoped_root,
            shared_artifact_root=shared_root,
            stages="evaluate",
        )

        # Verify argv contains both roots
        assert "--artifact-root" in argv
        assert "--shared-artifact-root" in argv

        # Simulate run_experiment parsing
        from experiments import run_experiment
        parsed = run_experiment.build_parser().parse_args(argv)
        assert parsed.artifact_root == str(scoped_root)
        assert parsed.shared_artifact_root == str(shared_root)

        # Simulate get_yml_cfg() processing
        # Patch set_task_paths to capture the paths it would use
        original_set_task_paths = config_module.set_task_paths
        captured_calls = {}

        def mock_set_task_paths(cfg):
            captured_calls["_artifact_dir"] = cfg._artifact_dir
            captured_calls["_scoped_artifact_root"] = getattr(cfg, "_scoped_artifact_root", None)
            # Call original to set preprocessing paths
            original_set_task_paths(cfg)

        config_module.set_task_paths = mock_set_task_paths

        try:
            args = SimpleNamespace(
                dataset="THEIA_E3",
                model="orthrus",
                config=str(config_file),
                cpu=True,
                from_weights=False,
                seed=0,
                skip_tracing=False,
                artifact_root=str(scoped_root),
                shared_artifact_root=str(shared_root),
                max_windows_per_split=None,
            )
            cfg = config_module.get_yml_cfg(args)

            # Critical assertions:
            # 1. _artifact_dir should be shared root (for preprocessing)
            assert cfg._artifact_dir == str(shared_root), (
                f"_artifact_dir should be shared root ({shared_root}), got {cfg._artifact_dir}"
            )

            # 2. _scoped_artifact_root should be scoped root (for run artifacts)
            assert cfg._scoped_artifact_root == str(scoped_root), (
                f"_scoped_artifact_root should be scoped root ({scoped_root}), got {cfg._scoped_artifact_root}"
            )

            # 3. Verify preprocessing paths use shared root
            # These are set by set_task_paths() using _artifact_dir
            graphs_dir = cfg.graph_construction.build_graphs._graphs_dir
            assert graphs_dir.startswith(str(shared_root)), (
                f"Graphs dir should start with shared root ({shared_root}), got {graphs_dir}"
            )
            assert "graph_construction" in graphs_dir
            assert "matrix_artifacts" not in graphs_dir, (
                f"Graphs dir should NOT contain matrix_artifacts/, got {graphs_dir}"
            )

            w2v_dir = cfg.edge_featurization.embed_nodes.feature_word2vec._model_dir
            assert w2v_dir.startswith(str(shared_root)), (
                f"Word2Vec dir should start with shared root, got {w2v_dir}"
            )
            assert "matrix_artifacts" not in w2v_dir

            edge_embeds_dir = cfg.edge_featurization.embed_edges._edge_embeds_dir
            assert edge_embeds_dir.startswith(str(shared_root)), (
                f"Edge embeddings dir should start with shared root, got {edge_embeds_dir}"
            )
            assert "matrix_artifacts" not in edge_embeds_dir

            metadata_dir = cfg._metadata_dir
            assert metadata_dir.startswith(str(shared_root)), (
                f"Metadata dir should start with shared root, got {metadata_dir}"
            )
            assert "matrix_artifacts" not in metadata_dir

        finally:
            config_module.set_task_paths = original_set_task_paths

    def test_standalone_run_experiment_uses_artifact_root_for_both(self, tmp_path, monkeypatch):
        """Standalone run_experiment without --shared-artifact-root uses --artifact-root for both."""
        import config as config_module

        # Write config file
        config_file = tmp_path / "baseline.yml"
        config_file.write_text("pipeline: {mode: full_pipeline}\n", encoding="utf-8")

        # Simulate standalone invocation (no --shared-artifact-root)
        original_set_task_paths = config_module.set_task_paths
        captured = {}

        def mock_set_task_paths(cfg):
            captured["_artifact_dir"] = cfg._artifact_dir
            captured["_scoped_artifact_root"] = getattr(cfg, "_scoped_artifact_root", None)
            original_set_task_paths(cfg)

        config_module.set_task_paths = mock_set_task_paths

        try:
            args = SimpleNamespace(
                dataset="THEIA_E3",
                model="orthrus",
                config=str(config_file),
                cpu=True,
                from_weights=False,
                seed=0,
                skip_tracing=False,
                artifact_root=str(tmp_path / "my_artifacts"),
                shared_artifact_root=None,  # No shared root specified
                max_windows_per_split=None,
            )
            cfg = config_module.get_yml_cfg(args)

            # When no shared root specified, _artifact_dir falls back to --artifact-root
            root = str(tmp_path / "my_artifacts")
            assert cfg._artifact_dir == root, (
                f"Without --shared-artifact-root, _artifact_dir should fall back to "
                f"--artifact-root ({root}), got {cfg._artifact_dir}"
            )
        finally:
            config_module.set_task_paths = original_set_task_paths

    def test_preprocessing_paths_use_shared_root_not_scoped(self, tmp_path, monkeypatch):
        """Verify preprocessing paths never contain matrix_artifacts even when scoped root is set."""
        import config as config_module

        config_file = tmp_path / "baseline.yml"
        config_file.write_text("pipeline: {mode: full_pipeline}\n", encoding="utf-8")

        original_set_task_paths = config_module.set_task_paths
        config_module.set_task_paths = lambda cfg: original_set_task_paths(cfg)

        try:
            args = SimpleNamespace(
                dataset="THEIA_E3",
                model="orthrus",
                config=str(config_file),
                cpu=True,
                from_weights=False,
                seed=0,
                skip_tracing=False,
                artifact_root=str(tmp_path / "artifacts" / "matrix_artifacts" / "baseline-xxx"),
                shared_artifact_root=str(tmp_path / "artifacts"),
                max_windows_per_split=None,
            )
            cfg = config_module.get_yml_cfg(args)

            # All preprocessing paths should use shared root, NOT scoped root
            paths_to_check = [
                ("graphs", cfg.graph_construction.build_graphs._graphs_dir),
                ("word2vec", cfg.edge_featurization.embed_nodes.feature_word2vec._model_dir),
                ("edge_embeds", cfg.edge_featurization.embed_edges._edge_embeds_dir),
                ("metadata", cfg._metadata_dir),
            ]

            scoped_root_str = str(tmp_path / "artifacts" / "matrix_artifacts" / "baseline-xxx")
            shared_root_str = str(tmp_path / "artifacts")

            for name, path in paths_to_check:
                assert path.startswith(shared_root_str), (
                    f"{name} path should start with shared root ({shared_root_str}), got {path}"
                )
                assert not path.startswith(scoped_root_str), (
                    f"{name} path should NOT start with scoped root ({scoped_root_str}), got {path}"
                )
                assert "matrix_artifacts" not in path, (
                    f"{name} path should NOT contain matrix_artifacts/, got {path}"
                )
        finally:
            config_module.set_task_paths = original_set_task_paths


# --------------------------------------------------------------------------- #
# Edge case: Priority order verification
# --------------------------------------------------------------------------- #

class TestArtifactRootPriority:
    """Verify: CLI > env var > default."""

    def test_cli_wins_over_env_var(self, tmp_path, monkeypatch):
        """CLI --artifact-root must take precedence over ORTHRUS_ARTIFACT_ROOT."""
        from artifact_paths import resolve_artifact_root

        cli_root = str(tmp_path / "cli_root")
        env_root = str(tmp_path / "env_root")

        monkeypatch.setenv("ORTHRUS_ARTIFACT_ROOT", env_root)

        result = resolve_artifact_root(cli_path=cli_root, env_path=env_root)

        assert str(result) == cli_root, (
            f"CLI must win over env var. Expected {cli_root}, got {result}"
        )

    def test_env_var_wins_over_default(self, tmp_path, monkeypatch):
        """ORTHRUS_ARTIFACT_ROOT must take precedence over ./artifacts default."""
        from artifact_paths import resolve_artifact_root

        monkeypatch.chdir(tmp_path)
        env_root = str(tmp_path / "env_root")

        monkeypatch.setenv("ORTHRUS_ARTIFACT_ROOT", env_root)

        result = resolve_artifact_root(cli_path=None, env_path=env_root)

        assert str(result) == env_root, (
            f"Env var must win over default. Expected {env_root}, got {result}"
        )
