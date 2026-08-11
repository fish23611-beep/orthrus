import os
import shutil
import tempfile
from pathlib import Path
from provnet_utils import *
from config import *
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import numpy as np
import random
import torch


# ---------------------------------------------------------------------------
# Corpus building with restartable iterator
# ---------------------------------------------------------------------------

def _build_corpus_dict(indexid2msg, use_node_types):
    """
    Build the corpus dictionary from indexid2msg.
    
    The corpus dict has keys that are msg[1] (the semantic label) and values
    that are tokenized sequences. If multiple nodes share the same msg[1],
    the later one overwrites the earlier one (Python dict behavior).
    
    This behavior must be preserved in streaming mode.
    
    Args:
        indexid2msg: dict mapping index_id -> [node_type, msg]
        use_node_types: whether to prefix tokens with node type
    
    Returns:
        dict: corpus mapping semantic label -> tokens
    """
    corpus = {}
    
    for indexid, msg in indexid2msg.items():
        if msg[0] == 'subject':
            if use_node_types:
                tokens = tokenize_subject(msg[0] + ' ' + msg[1])
            else:
                tokens = tokenize_subject(msg[1])
        elif msg[0] == 'file':
            if use_node_types:
                tokens = tokenize_file(msg[0] + ' ' + msg[1])
            else:
                tokens = tokenize_file(msg[1])
        else:
            if use_node_types:
                tokens = tokenize_netflow(msg[0] + ' ' + msg[1])
            else:
                tokens = tokenize_netflow(msg[1])
        
        # Key is msg[1] - later entries with same key overwrite earlier ones
        corpus[msg[1]] = tokens
    
    return corpus


class RestartableCorpus:
    """
    A corpus wrapper that allows multiple iterations over the same data.
    
    Gensim Word2Vec needs to iterate over the corpus multiple times (once per epoch).
    Simply returning a generator would fail on the second epoch because generators
    can only be consumed once.
    
    This class maintains the underlying data and provides a restartable iterator.
    """
    
    def __init__(self, indexid2msg, use_node_types):
        """
        Initialize the restartable corpus.
        
        Args:
            indexid2msg: dict mapping index_id -> [node_type, msg]
            use_node_types: whether to prefix tokens with node type
        """
        self._corpus_dict = _build_corpus_dict(indexid2msg, use_node_types)
        # Convert to list of values to preserve order of first appearance
        self._sentences = list(self._corpus_dict.values())
    
    def __iter__(self):
        """Return an iterator over sentences. Can be called multiple times."""
        return iter(self._sentences)
    
    def __len__(self):
        """Return the number of sentences."""
        return len(self._sentences)
    
    def to_list(self):
        """Return as a list (for compatibility with existing code)."""
        return list(self._sentences)


def load_corpus_from_database(indexid2msg, use_node_types):
    """
    Build corpus from indexid2msg with restartable iterator support.
    
    Returns a RestartableCorpus that can be iterated multiple times for
    multi-epoch Word2Vec training.
    
    Args:
        indexid2msg: dict mapping index_id -> [node_type, msg]
        use_node_types: whether to prefix tokens with node type
    
    Returns:
        RestartableCorpus: restartable corpus for Word2Vec training
    """
    return RestartableCorpus(indexid2msg, use_node_types)


class EpochLossLogger(CallbackAny2Vec):
    """Gensim callback that logs per-epoch training loss delta.

    Gensim Word2Vec accumulates training loss across all epochs.
    This callback tracks the previous cumulative loss and reports
    the delta (actual loss for that epoch).

    Supports three logger interface patterns:
    A. callable logger (e.g., MagicMock in tests)
    B. standard logging.Logger with .info() method
    C. any object with a callable .info() method
    """

    def __init__(self, logger, total_epochs):
        self.logger = logger
        self.total_epochs = total_epochs
        self.epoch = 0
        self.previous_loss = 0.0

    def _emit(self, message):
        """Emit a log message through the configured logger interface."""
        if callable(self.logger):
            self.logger(message)
            return

        info = getattr(self.logger, "info", None)
        if callable(info):
            info(message)
            return

        raise TypeError(
            "EpochLossLogger requires a callable logger "
            "or an object exposing callable .info()"
        )

    def on_epoch_end(self, model):
        cumulative = model.get_latest_training_loss()
        epoch_loss = cumulative - self.previous_loss
        self.previous_loss = cumulative
        self.epoch += 1
        self._emit(f"Epoch: {self.epoch}/{self.total_epochs}; loss: {epoch_loss}")


def train_feature_word2vec(corpus, cfg, model_save_path, logger):
    """
    Train Word2Vec model with bounded memory.

    The corpus must support multiple iterations (restartable iterator).

    Args:
        corpus: RestartableCorpus or list-like object
        cfg: configuration object
        model_save_path: path to save model
        logger: logging handler
    """
    emb_dim = cfg.edge_featurization.embed_nodes.emb_dim
    show_epoch_loss = cfg.edge_featurization.embed_nodes.feature_word2vec.show_epoch_loss
    window_size = cfg.edge_featurization.embed_nodes.feature_word2vec.window_size
    min_count = cfg.edge_featurization.embed_nodes.feature_word2vec.min_count
    use_skip_gram = cfg.edge_featurization.embed_nodes.feature_word2vec.use_skip_gram
    num_workers = cfg.edge_featurization.embed_nodes.feature_word2vec.num_workers
    epochs = cfg.edge_featurization.embed_nodes.feature_word2vec.epochs
    compute_loss = cfg.edge_featurization.embed_nodes.feature_word2vec.compute_loss
    negative = cfg.edge_featurization.embed_nodes.feature_word2vec.negative
    use_seed = cfg.edge_featurization.embed_nodes.use_seed
    SEED = cfg._seed

    # Ensure corpus can be iterated multiple times
    if not hasattr(corpus, '__iter__'):
        raise ValueError("Corpus must be iterable")

    # Train with epoch loss tracking using EpochLossLogger callback.
    # This preserves a single training lifecycle so Gensim handles alpha decay
    # correctly across all epochs, avoiding the "alpha higher than previous
    # cycles" warning that occurs when train() is called multiple times.
    if show_epoch_loss and compute_loss:
        callback = EpochLossLogger(logger, epochs)

        if use_seed:
            model = Word2Vec(corpus,
                             vector_size=emb_dim,
                             window=window_size,
                             min_count=min_count,
                             sg=use_skip_gram,
                             workers=num_workers,
                             epochs=epochs,
                             compute_loss=True,
                             callbacks=[callback],
                             negative=negative,
                             seed=SEED)
        else:
            model = Word2Vec(corpus,
                             vector_size=emb_dim,
                             window=window_size,
                             min_count=min_count,
                             sg=use_skip_gram,
                             workers=num_workers,
                             epochs=epochs,
                             compute_loss=True,
                             callbacks=[callback],
                             negative=negative)
    else:
        if use_seed:
            model = Word2Vec(corpus,
                             vector_size=emb_dim,
                             window=window_size,
                             min_count=min_count,
                             sg=use_skip_gram,
                             workers=num_workers,
                             epochs=epochs,
                             compute_loss=compute_loss,
                             negative=negative,
                             seed=SEED)
        else:
            model = Word2Vec(corpus,
                             vector_size=emb_dim,
                             window=window_size,
                             min_count=min_count,
                             sg=use_skip_gram,
                             workers=num_workers,
                             epochs=epochs,
                             compute_loss=compute_loss,
                             negative=negative)
        loss = model.get_latest_training_loss()
        log(f"Epoch: {epochs}; loss: {loss}")

    # Atomic save: write to .tmp, flush+fsync, then atomically replace.
    model_path = os.path.join(model_save_path, 'feature_word2vec.model')
    tmp_path = os.path.join(model_save_path, 'feature_word2vec.model.tmp')
    tmp_existed = os.path.exists(model_path)

    try:
        # Use file handle for atomic single-file save.
        # Gensim's save(handle) serializes everything into one file,
        # avoiding separate .npy sidecar files that would need manual handling.
        with open(tmp_path, "wb") as handle:
            model.save(handle)
            handle.flush()
            os.fsync(handle.fileno())

        # Only replace after the new model is fully written and synced.
        # If model_path already exists, it remains unchanged until os.replace succeeds.
        os.replace(tmp_path, model_path)
    except Exception:
        # Clean up tmp on any failure.
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise

    log(f"Save word2vec to {model_path}")


def _is_verified_empty_split(split_dir, cfg, split_name):
    """Validate a verified-empty marker against the active config."""
    from graph_construction.empty_day import is_verified_empty_day

    dataset_name = getattr(getattr(cfg, "dataset", None), "name", None)
    day = _day_index_from_split_name(split_name)
    return is_verified_empty_day(
        split_dir,
        dataset=dataset_name,
        graph_name=split_name,
        day=day,
    )


def _day_index_from_split_name(name):
    if not isinstance(name, str) or not name.startswith("graph_"):
        return None
    suffix = name[len("graph_"):]
    return int(suffix) if suffix.isdigit() else None


def _collect_split_node_ids_for_cfg(cfg, split_files):
    """Local variant that threads ``cfg`` into the empty-day identity check
    without using module-level globals."""
    return _collect_split_node_ids_impl(
        cfg.graph_construction.build_graphs._graphs_dir,
        split_files,
        cfg=cfg,
    )


def _collect_split_node_ids_impl(graphs_dir, split_files, *, cfg=None):
    node_ids = set()
    base_dir = Path(graphs_dir)
    for split_name in split_files:
        split_dir = base_dir / split_name
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Training graph split is missing: {split_dir}. "
                "Run build_graphs before train_only Word2Vec fitting."
            )
        graph_paths = sorted(
            path for path in split_dir.iterdir()
            if path.is_file()
            and not path.name.startswith(".preprocess_")
            and not path.name.endswith(".tmp")
        )
        if graph_paths:
            for graph_path in graph_paths:
                try:
                    graph = torch.load(graph_path, weights_only=False)
                except TypeError:
                    graph = torch.load(graph_path)
                node_ids.update(int(node_id) for node_id in graph.nodes)
                del graph
            continue

        if cfg is not None and _is_verified_empty_split(split_dir, cfg, split_name):
            continue
        raise FileNotFoundError(
            f"Training graph split has no graph files: {split_dir}"
        )
    return node_ids


# Keep the public name with the original signature for backward compatibility
# with existing callers and tests; the cfg-aware variant is below.
def collect_split_node_ids(graphs_dir, split_files):
    return _collect_split_node_ids_impl(graphs_dir, split_files)


def _indexid2msg_from_metadata(node_metadata, use_cmd=True, use_port=False):
    """Convert persisted metadata to the legacy ``[node_type, message]`` form."""
    result = {}
    for raw_node_id, meta in node_metadata.items():
        node_id = int(raw_node_id)
        node_type = meta.get("type")
        if node_type == "subject":
            message = str(meta.get("path") or "")
            if use_cmd and meta.get("cmd"):
                message = f"{message} {meta['cmd']}"
        elif node_type == "file":
            message = str(meta.get("path") or "")
        elif node_type == "netflow":
            message = str(meta.get("remote_ip") or "")
            if use_port and meta.get("remote_port") is not None:
                message = f"{message}:{meta['remote_port']}"
        else:
            continue
        result[node_id] = [node_type, message]
    return result


def load_semantic_messages(cfg, use_cmd=True, use_port=False):
    """Load node semantics cache-first, retaining the PostgreSQL fallback."""
    metadata_dir = getattr(cfg, "_metadata_dir", None)
    if metadata_dir:
        from mstc.metadata_cache import MetadataCache
        cache = MetadataCache(metadata_dir)
        if cache.has_node_metadata():
            return _indexid2msg_from_metadata(
                cache.load_node_metadata(), use_cmd=use_cmd, use_port=use_port
            )

    cur, connect = init_database_connection(cfg)
    try:
        return get_indexid2msg(cur, use_cmd=use_cmd, use_port=use_port)
    finally:
        cur.close()
        connect.close()


def select_corpus_messages(indexid2msg, cfg):
    """Apply the configured, split-aware Word2Vec corpus policy."""
    scope = cfg.semantic_features.corpus_scope
    if scope == "official_full_dataset":
        return indexid2msg
    if scope != "train_only":
        raise ValueError(f"Unsupported semantic corpus scope: {scope!r}")

    train_node_ids = _collect_split_node_ids_impl(
        cfg.graph_construction.build_graphs._graphs_dir,
        cfg.dataset.train_files,
        cfg=cfg,
    )
    available_ids = {int(node_id) for node_id in indexid2msg}
    missing = train_node_ids.difference(available_ids)
    if missing:
        sample = sorted(missing)[:10]
        raise KeyError(f"Missing semantic metadata for training node IDs: {sample}")
    return {
        node_id: message for node_id, message in indexid2msg.items()
        if int(node_id) in train_node_ids
    }


def main(cfg):
    model_save_path = cfg.edge_featurization.embed_nodes.feature_word2vec._model_dir
    os.makedirs(model_save_path, exist_ok=True)

    logger = get_logger(
        name="build_feature_word2vec",
        filename=os.path.join(cfg.edge_featurization.embed_nodes._logs_dir, "feature_word2vec.log")
    )
    log(f"Building feature word2vec model and save model to {model_save_path}")

    use_node_types = cfg.edge_featurization.embed_nodes.feature_word2vec.use_node_types
    use_cmd =  cfg.edge_featurization.embed_nodes.feature_word2vec.use_cmd
    use_port = cfg.edge_featurization.embed_nodes.feature_word2vec.use_port
    use_seed = cfg.edge_featurization.embed_nodes.use_seed

    if use_seed:
        SEED = cfg._seed
        np.random.seed(SEED)
        random.seed(SEED)

        torch.manual_seed(SEED)
        # Only call CUDA seed if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    log("Loading node semantics (metadata cache first, PostgreSQL fallback)...")
    indexid2msg = load_semantic_messages(cfg, use_cmd=use_cmd, use_port=use_port)
    corpus_messages = select_corpus_messages(indexid2msg, cfg)

    log("Start building and training feature word2vec model...")

    log(f"Loading and tokenizing {cfg.semantic_features.corpus_scope} corpus...")
    corpus = load_corpus_from_database(
        indexid2msg=corpus_messages, use_node_types=use_node_types
    )

    log(f"Corpus size: {len(corpus)} sentences")

    train_feature_word2vec(
        corpus=corpus,
        cfg=cfg,
        model_save_path=model_save_path,
        logger=logger,
    )

    from mstc.metadata_cache import update_dataset_manifest
    update_dataset_manifest(cfg)

    # Marker publication is the final atomic operation for this stage.
    marker_path = Path(model_save_path) / ".preprocess_embed_nodes_complete"
    marker_tmp_path = marker_path.with_name(marker_path.name + ".tmp")
    from datetime import datetime, timezone
    marker_tmp_path.write_text(
        datetime.now(timezone.utc).isoformat(), encoding="utf-8"
    )
    os.replace(marker_tmp_path, marker_path)
    log(f"Embed nodes complete. Marker written: {marker_path}")

    del indexid2msg
    del corpus_messages
    del corpus


if __name__ == '__main__':
    args =get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
