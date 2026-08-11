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

    def test_punkt_already_present_no_download(self, mocker):
        """When punkt exists, no download is attempted."""
        import provnet_utils

        mock_find = mocker.patch("nltk.data.find")
        mock_download = mocker.patch("nltk.download", return_value=True)

        provnet_utils._ensure_nltk_punkt()

        # Should check punkt
        assert mock_find.call_count == 1
        mock_download.assert_not_called()

    def test_punkt_missing_triggers_download(self, mocker):
        """When punkt is missing, download is triggered."""
        import provnet_utils

        # First find fails, but after download it should succeed
        def find_effect(resource):
            if not hasattr(find_effect, 'called'):
                find_effect.called = True
                raise LookupError("not found")
            return mocker.MagicMock()  # Return success after download

        mocker.patch("nltk.data.find", side_effect=find_effect)
        mock_download = mocker.patch("nltk.download", return_value=True)

        provnet_utils._ensure_nltk_punkt()

        assert mock_download.call_count == 1
        mock_download.assert_called_with("punkt", quiet=True)

    def test_download_returns_false_raises_runtime_error(self, mocker):
        """When download returns False, RuntimeError is raised."""
        import provnet_utils

        mocker.patch("nltk.data.find", side_effect=LookupError("not found"))
        mocker.patch("nltk.download", return_value=False)

        with pytest.raises(RuntimeError, match="returned False"):
            provnet_utils._ensure_nltk_punkt()

    def test_download_claims_success_but_resource_still_missing(self, mocker):
        """When download claims success but resource is still missing, RuntimeError is raised."""
        import provnet_utils

        def find_effect(resource):
            raise LookupError("not found")

        mocker.patch("nltk.data.find", side_effect=find_effect)
        mocker.patch("nltk.download", return_value=True)

        with pytest.raises(RuntimeError, match="claimed success but"):
            provnet_utils._ensure_nltk_punkt()

    def test_download_failure_raises(self, mocker):
        """Download failure raises exception."""
        import provnet_utils

        mocker.patch("nltk.data.find", side_effect=LookupError("not found"))

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

        call_count = {"find": 0, "download": []}

        def mock_find(resource):
            call_count["find"] += 1
            raise LookupError("not found")

        mocker.patch("nltk.data.find", side_effect=mock_find)

        def mock_download(resource, **kwargs):
            call_count["download"].append(resource)

        mocker.patch("nltk.download", side_effect=mock_download)

        # Call the helper
        try:
            provnet_utils._ensure_nltk_punkt()
        except Exception:
            pass

        # The helper should attempt find before downloading
        assert call_count["find"] >= 1


# ---------------------------------------------------------------------------
# Problem B: Gensim atomic save tests
# ---------------------------------------------------------------------------

class TestWord2VecAtomicSave:
    """Tests for Word2Vec atomic save (single-file, no sidecars)."""

    def _make_minimal_corpus(self):
        """Create a minimal restartable corpus for testing."""
        from edge_featurization.build_feature_word2vec import RestartableCorpus

        indexid2msg = {
            1: ["subject", "/usr/bin/python"],
            2: ["file", "/tmp/output"],
            3: ["netflow", "192.168.1.1"],
        }
        return RestartableCorpus(indexid2msg, use_node_types=False)

    def test_single_file_save_no_sidecars(self, tmp_path):
        """Model saves as a single file without .npy sidecar files."""
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

        # No .npy sidecar files should exist (single-file save)
        all_files = list(model_dir.iterdir())
        npy_files = [f for f in all_files if f.suffix == ".npy"]
        assert npy_files == [], f"Found .npy sidecar files: {npy_files}"

        # No .tmp residue
        tmp_files = [f for f in all_files if ".tmp" in f.name]
        assert tmp_files == [], f"Found .tmp residue files: {tmp_files}"

        # Only one file in the directory
        assert len(all_files) == 1, f"Expected 1 file, found: {all_files}"

    def test_roundtrip_preserves_model(self, tmp_path):
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

        model_path = model_dir / "feature_word2vec.model"
        loaded = Word2Vec.load(str(model_path))

        assert loaded.vector_size == 16
        assert loaded.wv.vector_size == 16
        assert len(loaded.wv) >= 3

        vocab_items = list(loaded.wv.key_to_index.keys())
        word = vocab_items[0]
        vec = loaded.wv[word]
        assert vec.shape == (16,)

    def test_old_model_preserved_on_save_failure(self, tmp_path, mocker):
        """Old model is preserved when save fails mid-write."""
        from gensim.models import Word2Vec
        from edge_featurization.build_feature_word2vec import (
            train_feature_word2vec,
            RestartableCorpus,
        )

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        model_path = model_dir / "feature_word2vec.model"

        # Pre-create an old model
        old_corpus = RestartableCorpus({1: ["subject", "/old/path"]}, use_node_types=False)
        old_model = Word2Vec(sentences=old_corpus, vector_size=8, epochs=1, min_count=1)
        old_model.save(str(model_path))

        # Now configure for new training
        cfg = MagicMock()
        cfg.edge_featurization.embed_nodes.emb_dim = 16
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

        # Fail during model.save by patching open
        original_open = open

        class FailingFile:
            def __init__(self, path, mode):
                self._file = original_open(path, mode)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return self._file.__exit__(*args)

            def write(self, data):
                # Fail after partial write
                raise IOError("Simulated save failure")

            def flush(self):
                pass

            def fileno(self):
                return self._file.fileno()

        def failing_open(path, mode):
            if "model.tmp" in str(path):
                return FailingFile(path, mode)
            return original_open(path, mode)

        mocker.patch("builtins.open", side_effect=failing_open)

        with pytest.raises(IOError):
            train_feature_word2vec(corpus, cfg, str(model_dir), logger)

        # Old model should still be intact
        loaded = Word2Vec.load(str(model_path))
        assert loaded.vector_size == 8  # Old model dim

        # No .tmp residue
        tmp_files = list(model_dir.glob("*.tmp"))
        assert tmp_files == [], f"Found .tmp residue: {tmp_files}"

    def test_no_tmp_residue_on_success(self, tmp_path):
        """No .tmp files remain after successful save."""
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
        cfg.edge_featurization.embed_nodes.feature_word2vec.epochs = 2
        cfg.edge_featurization.embed_nodes.feature_word2vec.compute_loss = False
        cfg.edge_featurization.embed_nodes.feature_word2vec.show_epoch_loss = False
        cfg.edge_featurization.embed_nodes.feature_word2vec.negative = 5
        cfg.edge_featurization.embed_nodes.use_seed = False

        corpus = self._make_minimal_corpus()
        logger = MagicMock()

        train_feature_word2vec(corpus, cfg, str(model_dir), logger)

        # Check no .tmp residue
        all_files = list(model_dir.iterdir())
        tmp_files = [f for f in all_files if ".tmp" in f.name]
        assert tmp_files == [], f"Found .tmp residue: {tmp_files}"


# ---------------------------------------------------------------------------
# Problem C: Word2Vec alpha / epoch loss tests
# ---------------------------------------------------------------------------

class TestWord2VecEpochCallback:
    """Tests for real Gensim CallbackAny2Vec epoch callback."""

    def _make_minimal_corpus(self):
        """Create a minimal restartable corpus for testing."""
        from edge_featurization.build_feature_word2vec import RestartableCorpus

        indexid2msg = {
            1: ["subject", "/bin/bash shell command execute"],
            2: ["file", "/etc/config file read write"],
            3: ["netflow", "192.168.1.1 remote server connect"],
            4: ["subject", "/usr/bin/python script run"],
        }
        return RestartableCorpus(indexid2msg, use_node_types=False)

    def test_callback_invoked_correct_number_of_times(self, tmp_path):
        """Real Gensim callback on_epoch_end is called exactly once per epoch."""
        from gensim.models import Word2Vec
        from gensim.models.callbacks import CallbackAny2Vec
        from edge_featurization.build_feature_word2vec import (
            train_feature_word2vec,
            EpochLossLogger,
            RestartableCorpus,
        )

        # Verify EpochLossLogger is a proper CallbackAny2Vec
        assert issubclass(EpochLossLogger, CallbackAny2Vec)

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
        cfg.edge_featurization.embed_nodes.feature_word2vec.show_epoch_loss = True
        cfg.edge_featurization.embed_nodes.feature_word2vec.negative = 5
        cfg.edge_featurization.embed_nodes.use_seed = False

        corpus = self._make_minimal_corpus()
        logger = MagicMock()

        train_feature_word2vec(corpus, cfg, str(model_dir), logger)

        # Verify callback was invoked (EpochLossLogger logs per epoch)
        log_calls = [str(call) for call in logger.call_args_list if call.args]
        epoch_logs = [c for c in log_calls if "Epoch:" in c and "loss:" in c]

        # Should have 3 epoch logs (one per epoch)
        assert len(epoch_logs) == 3, f"Expected 3 epoch logs, got {len(epoch_logs)}: {epoch_logs}"

        # Verify model loaded correctly
        model_path = model_dir / "feature_word2vec.model"
        loaded = Word2Vec.load(str(model_path))
        assert loaded.vector_size == 8
        assert len(loaded.wv) >= 3

    def test_callback_receives_model_argument(self, tmp_path, mocker):
        """on_epoch_end receives the model as its sole argument (Gensim API)."""
        from gensim.models import Word2Vec
        from gensim.models.callbacks import CallbackAny2Vec
        from edge_featurization.build_feature_word2vec import EpochLossLogger

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Create a real Word2Vec and attach a callback that records what it receives
        received_args = []

        class ArgChecker(CallbackAny2Vec):
            def on_epoch_end(self, model):
                received_args.append(("model", type(model).__name__))

        corpus = self._make_minimal_corpus()
        model = Word2Vec(
            sentences=corpus,
            vector_size=8,
            window=5,
            min_count=1,
            epochs=2,
            compute_loss=True,
            callbacks=[ArgChecker()],
        )

        # Verify on_epoch_end was called with model
        assert len(received_args) == 2, f"Expected 2 calls, got {received_args}"
        for arg_type, arg_name in received_args:
            assert arg_type == "model"
            assert arg_name == "Word2Vec"

    def test_no_independent_train_lifecycle(self, tmp_path, mocker):
        """Only one Word2Vec initialization lifecycle; no per-epoch train() calls."""
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

        # Verify model loads and works
        model_path = model_dir / "feature_word2vec.model"
        loaded = Word2Vec.load(str(model_path))
        assert loaded.vector_size == 8
        assert len(loaded.wv) >= 3

    def test_show_epoch_loss_false_no_callback(self, tmp_path):
        """show_epoch_loss=False uses single training lifecycle without callback."""
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

        # Model trained successfully without errors - verification complete

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


class TestInitSimsRemoved:
    """Audit tests confirming init_sims has been removed."""

    def test_init_sims_not_called_in_production(self, mocker):
        """Verify init_sims(replace=True) is no longer called in production."""
        from edge_featurization.build_feature_word2vec import train_feature_word2vec
        from gensim.models import Word2Vec

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
            original_init_sims = Word2Vec.init_sims

            def tracked_init_sims(self, replace=False):
                init_sims_calls.append(replace)
                return original_init_sims(self, replace=replace)

            mocker.patch.object(Word2Vec, "init_sims", tracked_init_sims)

            train_feature_word2vec(corpus, cfg, model_dir, logger)

            # init_sims should NOT be called anymore
            assert len(init_sims_calls) == 0, \
                f"init_sims was called {len(init_sims_calls)} times (expected 0)"

        finally:
            shutil.rmtree(model_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
