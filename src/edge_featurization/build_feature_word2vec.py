import os
from provnet_utils import *
from config import *
from tqdm import tqdm
from gensim.models import Word2Vec
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
    
    # Train with epoch loss tracking
    if show_epoch_loss:
        if use_seed:
            model = Word2Vec(corpus,
                             vector_size=emb_dim,
                             window=window_size,
                             min_count=min_count,
                             sg=use_skip_gram,
                             workers=num_workers,
                             epochs=1,
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
                             epochs=1,
                             compute_loss=compute_loss,
                             negative=negative)
        epoch_loss = model.get_latest_training_loss()
        log(f"Epoch: 0/{epochs}; loss: {epoch_loss}")

        for epoch in range(epochs - 1):
            model.train(corpus, epochs=1, total_examples=len(corpus), compute_loss=compute_loss)
            epoch_loss = model.get_latest_training_loss()
            log(f"Epoch: {epoch+1}/{epochs}; loss: {epoch_loss}")
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

    model.init_sims(replace=True)
    
    # Atomic save: write to temp file, then rename
    model_path = os.path.join(model_save_path, 'feature_word2vec.model')
    temp_path = model_path + '.tmp'
    model.save(temp_path)
    os.replace(temp_path, model_path)
    
    log(f"Save word2vec to {model_path}")


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

    log(f"Get indexid2msg from database...")
    cur, connect = init_database_connection(cfg)
    indexid2msg = get_indexid2msg(cur, use_cmd=use_cmd, use_port=use_port)

    log(f"Start building and training feature word2vec model...")

    log("Loading and tokenizing corpus from database...")
    corpus = load_corpus_from_database(indexid2msg=indexid2msg, use_node_types=use_node_types)
    
    log(f"Corpus size: {len(corpus)} sentences")

    train_feature_word2vec(corpus=corpus,
                           cfg=cfg,
                           model_save_path=model_save_path,
                           logger=logger)
    
    # Write completion marker
    marker_path = os.path.join(model_save_path, ".preprocess_embed_nodes_complete")
    from datetime import datetime
    with open(marker_path, 'w') as f:
        f.write(datetime.now().isoformat())
    log(f"Embed nodes complete. Marker written: {marker_path}")
    
    # Clean up
    cur.close()
    connect.close()
    del indexid2msg
    del corpus


if __name__ == '__main__':
    args =get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
