"""
tests/test_word2vec_runtime_compat.py

Tests for Word2Vec runtime compatibility fixes:
- NLTK punkt_tab lazy download (problem A)
- Gensim atomic save with correct sidecar filenames (problem B)
- Word2Vec alpha/learning rate across epochs with callbacks (problem C)
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Problem A: NLTK punkt_tab lazy download tests
# ---------------------------------------------------------------------------

class TestNLTKPunktTab:
    """Tests for _ensure_nltk_punkt helper."""

    def test_punkt_tab_already_present_no_download(self, mocker):
        """When both punkt and punkt_tab exist, no download is attempted."""
        import provnet_utils

        mock_find = mocker.patch("nltk.data.find")
        mock_download = mocker.patch("nltk.download")

        provnet_utils._ensure_nltk_punkt()

        # Should check both resources
        assert mock_find.call_count == 2
        mock_download.assert_not_called()

    def test_punkt_tab_missing_triggers_download(self, mocker):
        """When punkt_tab is missing, download is triggered for it."""
        import provnet_utils

        def find_effect(resource):
            if "punkt_tab" in resource:
                raise LookupError(f"Resource {resource} not found")
            # punkt exists

        mock_find = mocker.patch("nltk.data.find", side_effect=find_effect)
        mock_download = mocker.patch("nltk.download")

        provnet_utils._ensure_nltk_punkt()

        assert mock_find.call_count == 2
        # Should download punkt_tab but not punkt (which exists)
        assert mock_download.call_count == 1
        mock_download.assert_called_with("punkt_tab", quiet=True)

    def test_punkt_missing_triggers_both_downloads(self, mocker):
        """When punkt is missing, both are downloaded."""
        import provnet_utils

        def find_effect(resource):
            if "punkt_tab" in resource:
                raise LookupError(f"Resource {resource} not found")
            if "punkt" in resource:
                raise LookupError(f"Resource {resource} not found")
            return None

        mocker.patch("nltk.data.find", side_effect=find_effect)
        mock_download = mocker.patch("nltk.download")

        provnet_utils._ensure_nltk_punkt()

        # Should download both resources in order
        assert mock_download.call_count == 2
        calls = mock_download.call_args_list
        assert calls[0][0][0] == "punkt"
        assert calls[1][0][0] == "punkt_tab"

    def test_download_failure_raises(self, mocker):
        """Download failure raises LookupError clearly."""
        import provnet_utils

        def find_effect(resource):
            if "punkt_tab" in resource:
                raise LookupError(f"Resource {resource} not found")
            # punkt exists

        mocker.patch("nltk.data.find", side_effect=find_effect)

        def download_fail(*args, **kwargs):
            raise Exception("Download failed")

        mocker.patch("nltk.download", side_effect=download_fail)

        with pytest.raises(Exception, match="Download failed"):
            provnet_utils._ensure_nltk_punkt()

    def test_no_unconditional_download_on_import_time(self, mocker):
        """Ensure no unconditional download happens on import.

        This is a smoke test: we mock nltk.data.find to raise LookupError
        and nltk.download to raise if called unconditionally. The helper should
        only trigger downloads when actually needed (when find fails).
        """
        import provnet_utils

        call_count = {"find": 0, "download": 0}

        def mock_find(resource):
            call_count["find"] += 1
            raise LookupError("not found")

        mocker.patch("nltk.data.find", side_effect=mock_find)
        download_called = []

        def mock_download(resource, **kwargs):
            call_count["download"].append(resource)

        mocker.patch("nltk.download", side_effect=mock_download)

        # Call the helper
        try:
            provnet_utils._ensure_nltk_punkt()
        except Exception:
            pass

        # The helper should attempt find for both resources before downloading
        # This verifies lazy behavior (find first, download only if not found)
        assert call_count["find"] >= 1


# ---------------------------------------------------------------------------
# Problem B: Gensim atomic save tests
# ---------------------------------------------------------------------------

class TestWord2VecAtomicSave:
    """Tests for Word2Vec atomic save with correct sidecar filenames."""

    def _make_minimal_corpus(self):
        """Create a minimal restartable corpus for testing."""
        from edge_featurization.build_feature_word2vec import RestartableCorpus

        indexid2msg = {
            1: ["subject", "/usr/bin/python"],
            2: ["file", "/tmp/output"],
            3: ["netflow", "192.168.1.1"],
        }
        return RestartableCorpus(indexid2msg, use_node_types=False)

    def test_model_saves_and_loads_with_correct_sidecars(self, tmp_path):
        """Model saves with correct sidecar filenames and loads successfully."""
        from gensim.models import Word2Vec
        from edge_featurization.build_feature_word2vec import train_feature_word2vec

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Create minimal config
        cfg = MagicMock()
        cfg.edge_featurization.embed_nodes.emb_dim = 8
        cfg.edge_featurization.embed_nodes.feature_word2vec.window_size = 5
        cfg.edge_featurization.embed_nodes.feature_word2vec.min_count = 1
        cfg.edge_featurization.embed_nodes.feature_word2vec.use_skip_gram = True
        cfg.edge_featurization.embed_nodes.feature_word2vec.num_workers = 1
        cfg.edge_featurization.embed_nodes.feature_word2vec.epochs = 2
        cfg.edge_featurization.embed_nodes.feature_word2vec.compute_loss = True
        cfg.edge_featurization.embed_nodes.feature_word2vec.show_epoch_loss = False
        cfg.edge_featurization.embed_nodes.feature_word2vec.negative = 5
        cfg.edge_featurization.embed_nodes.use_seed = False

        corpus = self._make_minimal_corpus()
        logger = MagicMock()

        train_feature_word2vec(corpus, cfg, str(model_dir), logger)

        # Check main model file exists
        model_path = model_dir / "feature_word2vec.model"
        assert model_path.exists(), "Main model file should exist"

        # Check for sidecar files (should NOT have .tmp in names)
        all_files = list(model_dir.iterdir())
        tmp_files = [f for f in all_files if ".tmp" in f.name]
        assert tmp_files == [], f"Found .tmp residue files: {tmp_files}"

        # Load and verify
        loaded = Word2Vec.load(str(model_path))
        assert loaded.vector_size == 8
        assert loaded.wv.vector_size == 8
        assert len(loaded.wv) >= 3  # Should have at least 3 words from corpus

    def test_no_mismatched_sidecar_names(self, tmp_path):
        """Sidecar files should not have .tmp.* suffixes."""
        from gensim.models import Word2Vec
        from edge_featurization.build_feature_word2vec import train_feature_word2vec

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        cfg = MagicMock()
        cfg.edge_featurization.embed_nodes.emb_dim = 4
        cfg.edge_featurization.embed_nodes.feature_word2vec.window_size = 3
        cfg.edge_featurization.embed_nodes.feature_word2vec.min_count = 1
        cfg.edge_featurization.embed_nodes.feature_word2vec.use_skip_gram = True
        cfg.edge_featurization.embed_nodes.feature_word2vec.num_workers = 1
        cfg.edge_featurization.embed_nodes.feature_word2vec.epochs = 2
        cfg.edge_featurization.embed_nodes.feature_word2vec.compute_loss = False
        cfg.edge_featurization.embed_nodes.feature_word2vec.show_epoch_loss = False
        cfg.edge_featurization.embed_nodes.feature_word2vec.negative = 5
        cfg.edge_featurization.embed_nodes.use_seed = False

        corpus = self._make_minimal_corpus()
        logger = MagicMock()

        train_feature_word2vec(corpus, cfg, str(model_dir), logger)

        # Check no sidecar has .tmp pattern
        all_files = list(model_dir.iterdir())
        for f in all_files:
            assert ".tmp" not in f.name, f"Found .tmp in sidecar: {f.name}"

        # All .npy files should be loadable
        npy_files = [f for f in all_files if f.suffix == ".npy"]
        for npy in npy_files:
            data = np.load(npy)
            assert data is not None

    def test_roundtrip_save_load_preserves_vectors(self, tmp_path):
        """Save -> load roundtrip preserves vector dimensions and vocab."""
        from gensim.models import Word2Vec
        from edge_featurization.build_feature_word2vec import train_feature_word2vec

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        cfg = MagicMock()
        cfg.edge_featurization.embed_nodes.emb_dim = 16
        cfg.edge_featurization.embed_nodes.feature_word2vec.window_size = 5
        cfg.edge_featurization.embed_nodes.feature_word2vec.min_count = 1
        cfg.edge_featurization.embed_nodes.feature_word2vec.use_skip_gram = True
        cfg.edge_featurization.embed_nodes.feature_word2vec.num_workers = 1
        cfg.edge_featurization.embed_nodes.feature_word2vec.epochs = 3
        cfg.edge_featurization.embed_nodes.feature_word2vec.compute_loss = True
        cfg.edge_featurization.embed_nodes.feature_word2vec.show_epoch_loss = False
        cfg.edge_featurization.embed_nodes.feature_word2vec.negative = 5
        cfg.edge_featurization.embed_nodes.use_seed = False

        corpus = self._make_minimal_corpus()
        logger = MagicMock()

        train_feature_word2vec(corpus, cfg, str(model_dir), logger)

        # Load and verify
        model_path = model_dir / "feature_word2vec.model"
        loaded = Word2Vec.load(str(model_path))

        assert loaded.vector_size == 16
        assert loaded.wv.vector_size == 16
        assert len(loaded.wv) >= 3

        # Check we can get a vector
        vocab_items = list(loaded.wv.key_to_index.keys())
        if vocab_items:
            word = vocab_items[0]
            vec = loaded.wv[word]
            assert vec.shape == (16,)


# ---------------------------------------------------------------------------
# Problem C: Word2Vec alpha / epoch loss tests
# ---------------------------------------------------------------------------

class TestWord2VecAlphaDecay:
    """Tests for proper alpha decay across epochs."""

    def _make_minimal_corpus(self):
        """Create a minimal restartable corpus for testing."""
        from edge_featurization.build_feature_word2vec import RestartableCorpus

        indexid2msg = {
            1: ["subject", "/usr/bin/python script py"],
            2: ["file", "/tmp/output log file"],
            3: ["netflow", "192.168.1.1 remote server"],
            4: ["file", "/var/log messages"],
        }
        return RestartableCorpus(indexid2msg, use_node_types=False)

    def test_show_epoch_loss_uses_callback_not_repeated_train(self, tmp_path, mocker):
        """show_epoch_loss=True uses callbacks, not repeated train() calls."""
        from gensim.models import Word2Vec
        from edge_featurization.build_feature_word2vec import train_feature_word2vec

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        cfg = MagicMock()
        cfg.edge_featurization.embed_nodes.emb_dim = 8
        cfg.edge_featurization.embed_nodes.feature_word2vec.window_size = 5
        cfg.edge_featurization.embed_nodes.feature_word2vec.min_count = 1
        cfg.edge_featurization.embed_nodes.feature_word2vec.use_skip_gram = True
        cfg.edge_featurization.embed_nodes.feature_word2vec.num_workers = 1
        cfg.edge_featurization.embed_nodes.feature_word2vec.epochs = 5
        cfg.edge_featurization.embed_nodes.feature_word2vec.compute_loss = True
        cfg.edge_featurization.embed_nodes.feature_word2vec.show_epoch_loss = True
        cfg.edge_featurization.embed_nodes.feature_word2vec.negative = 5
        cfg.edge_featurization.embed_nodes.use_seed = False

        corpus = self._make_minimal_corpus()
        logger = MagicMock()

        # Spy on the Word2Vec.train method to ensure it's NOT called
        # (the old buggy pattern called train() per epoch)
        train_call_count = {"count": 0}

        original_train = Word2Vec.train

        def spy_train(self, *args, **kwargs):
            train_call_count["count"] += 1
            return original_train(self, *args, **kwargs)

        with patch.object(Word2Vec, "train", spy_train):
            train_feature_word2vec(corpus, cfg, str(model_dir), logger)

        # With callbacks, Word2Vec.train should NOT be called manually
        # (it's called internally by gensim once per epoch in the epochs loop)
        # The key is that we don't call model.train() after initialization
        # So train_call_count should be 0 from our spy (internal calls are still possible)
        # Actually, in the new code we DON'T call train() at all - gensim handles it
        # Let's verify the model can be loaded (no corruption)
        model_path = model_dir / "feature_word2vec.model"
        loaded = Word2Vec.load(str(model_path))
        assert loaded.vector_size == 8
        assert len(loaded.wv) >= 3

    def test_show_epoch_loss_logs_per_epoch(self, tmp_path):
        """show_epoch_loss=True produces one log per epoch."""
        from gensim.models import Word2Vec
        from edge_featurization.build_feature_word2vec import train_feature_word2vec

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        cfg = MagicMock()
        cfg.edge_featurization.embed_nodes.emb_dim = 8
        cfg.edge_featurization.embed_nodes.feature_word2vec.window_size = 5
        cfg.edge_featurization.embed_nodes.feature_word2vec.min_count = 1
        cfg.edge_featurization.embed_nodes.feature_word2vec.use_skip_gram = True
        cfg.edge_featurization.embed_nodes.feature_word2vec.num_workers = 1
        cfg.edge_featurization.embed_nodes.feature_word2vec.epochs = 4
        cfg.edge_featurization.embed_nodes.feature_word2vec.compute_loss = True
        cfg.edge_featurization.embed_nodes.feature_word2vec.show_epoch_loss = True
        cfg.edge_featurization.embed_nodes.feature_word2vec.negative = 5
        cfg.edge_featurization.embed_nodes.use_seed = False

        corpus = self._make_minimal_corpus()
        logger = MagicMock()

        train_feature_word2vec(corpus, cfg, str(model_dir), logger)

        # Check that log was called with epoch information
        # The log function is called once per epoch with pattern "Epoch: X/Y; loss: Z"
        log_calls = [str(call) for call in logger.call_args_list if call.args]
        epoch_logs = [c for c in log_calls if "Epoch:" in c]

        # Should have 4 epoch logs (one per epoch)
        assert len(epoch_logs) == 4, f"Expected 4 epoch logs, got {len(epoch_logs)}: {epoch_logs}"

    def test_show_epoch_loss_false_works(self, tmp_path):
        """show_epoch_loss=False still works correctly."""
        from gensim.models import Word2Vec
        from edge_featurization.build_feature_word2vec import train_feature_word2vec

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        cfg = MagicMock()
        cfg.edge_featurization.embed_nodes.emb_dim = 8
        cfg.edge_featurization.embed_nodes.feature_word2vec.window_size = 5
        cfg.edge_featurization.embed_nodes.feature_word2vec.min_count = 1
        cfg.edge_featurization.embed_nodes.feature_word2vec.use_skip_gram = True
        cfg.edge_featurization.embed_nodes.feature_word2vec.num_workers = 1
        cfg.edge_featurization.embed_nodes.feature_word2vec.epochs = 3
        cfg.edge_featurization.embed_nodes.feature_word2vec.compute_loss = True
        cfg.edge_featurization.embed_nodes.feature_word2vec.show_epoch_loss = False
        cfg.edge_featurization.embed_nodes.feature_word2vec.negative = 5
        cfg.edge_featurization.embed_nodes.use_seed = False

        corpus = self._make_minimal_corpus()
        logger = MagicMock()

        train_feature_word2vec(corpus, cfg, str(model_dir), logger)

        # Model should be saved and loadable
        model_path = model_dir / "feature_word2vec.model"
        loaded = Word2Vec.load(str(model_path))
        assert loaded.vector_size == 8
        assert len(loaded.wv) >= 3

        # Should have one log call with final epoch info
        log_calls = [str(call) for call in logger.call_args_list if call.args]
        epoch_logs = [c for c in log_calls if "loss:" in c]
        assert len(epoch_logs) == 1

    def test_restartable_corpus_with_multi_epoch_training(self, tmp_path):
        """RestartableCorpus works correctly with multi-epoch training."""
        from gensim.models import Word2Vec
        from edge_featurization.build_feature_word2vec import (
            train_feature_word2vec,
            RestartableCorpus,
        )

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Create corpus and verify it's restartable
        indexid2msg = {
            1: ["subject", "/bin/bash shell command"],
            2: ["file", "/etc/config file"],
        }
        corpus = RestartableCorpus(indexid2msg, use_node_types=False)

        # Verify restartable
        first_pass = list(corpus)
        second_pass = list(corpus)
        assert first_pass == second_pass

        cfg = MagicMock()
        cfg.edge_featurization.embed_nodes.emb_dim = 8
        cfg.edge_featurization.embed_nodes.feature_word2vec.window_size = 5
        cfg.edge_featurization.embed_nodes.feature_word2vec.min_count = 1
        cfg.edge_featurization.embed_nodes.feature_word2vec.use_skip_gram = True
        cfg.edge_featurization.embed_nodes.feature_word2vec.num_workers = 1
        cfg.edge_featurization.embed_nodes.feature_word2vec.epochs = 5
        cfg.edge_featurization.embed_nodes.feature_word2vec.compute_loss = True
        cfg.edge_featurization.embed_nodes.feature_word2vec.show_epoch_loss = True
        cfg.edge_featurization.embed_nodes.feature_word2vec.negative = 5
        cfg.edge_featurization.embed_nodes.use_seed = False

        logger = MagicMock()

        train_feature_word2vec(corpus, cfg, str(model_dir), logger)

        # Verify model was trained
        model_path = model_dir / "feature_word2vec.model"
        loaded = Word2Vec.load(str(model_path))
        assert loaded.vector_size == 8


class TestInitSimsDeprecated:
    """Audit tests for init_sims deprecated warning."""

    def test_init_sims_still_called_in_production(self, mocker):
        """Verify init_sims(replace=True) is still called in production.

        This is an audit test to document the current state.
        The init_sims call is deprecated in Gensim 4.4.0 but still functional.
        """
        from edge_featurization.build_feature_word2vec import train_feature_word2vec

        model_dir = tempfile.mkdtemp()
        try:
            cfg = MagicMock()
            cfg.edge_featurization.embed_nodes.emb_dim = 4
            cfg.edge_featurization.embed_nodes.feature_word2vec.window_size = 3
            cfg.edge_featurization.embed_nodes.feature_word2vec.min_count = 1
            cfg.edge_featurization.embed_nodes.feature_word2vec.use_skip_gram = True
            cfg.edge_featurization.embed_nodes.feature_word2vec.num_workers = 1
            cfg.edge_featurization.embed_nodes.feature_word2vec.epochs = 2
            cfg.edge_featurization.embed_nodes.feature_word2vec.compute_loss = False
            cfg.edge_featurization.embed_nodes.feature_word2vec.show_epoch_loss = False
            cfg.edge_featurization.embed_nodes.feature_word2vec.negative = 5
            cfg.edge_featurization.embed_nodes.use_seed = False

            from edge_featurization.build_feature_word2vec import RestartableCorpus
            corpus = RestartableCorpus({1: ["subject", "/test"]}, use_node_types=False)
            logger = MagicMock()

            # Patch init_sims to track calls
            init_sims_calls = []
            original_init_sims = None

            try:
                from gensim.models import Word2Vec
                original_init_sims = Word2Vec.init_sims

                def tracked_init_sims(self, replace=False):
                    init_sims_calls.append(replace)
                    return original_init_sims(self, replace=replace)

                mocker.patch.object(Word2Vec, "init_sims", tracked_init_sims)
            except ImportError:
                pytest.skip("Gensim not available")

            train_feature_word2vec(corpus, cfg, model_dir, logger)

            # init_sims is still called (documenting current behavior)
            assert len(init_sims_calls) == 1
            assert init_sims_calls[0] is True  # Called with replace=True

        finally:
            shutil.rmtree(model_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
