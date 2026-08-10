import os
from provnet_utils import *
from config import *

from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm
from edge_featurization.build_feature_word2vec import load_semantic_messages
from torch_geometric.data import *
import gc


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def cal_word_weight(n, percentage):
    """Calculate word weights with linear decline."""
    d = -1 / n * percentage / 100
    a_1 = 1/n - 0.5 * (n-1) * d
    sequence = []
    for i in range(n):
        a_i = a_1 + i * d
        sequence.append(a_i)
    return sequence


def get_indexid2vec(indexid2msg, model_path, use_node_types, decline_percentage):
    """
    Generate normalized vectors for each node.
    
    Args:
        indexid2msg: dict mapping index_id -> [node_type, msg]
        model_path: path to Word2Vec model
        use_node_types: whether to include node type in tokenization
        decline_percentage: weight decline rate
    
    Returns:
        dict: index_id -> normalized vector
    """
    model = Word2Vec.load(model_path)
    log(f"Loaded model from {model_path}")

    indexid2vec = {}
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

        # Validation/test semantics never update the fitted model. Unknown
        # tokens are ignored; an all-OOV (or empty) message maps to a fixed
        # zero vector. This is deterministic and cannot produce NaN values.
        if tokens:
            weight_list = cal_word_weight(len(tokens), decline_percentage)
            weighted_vectors = [
                weight * model.wv[word]
                for weight, word in zip(weight_list, tokens)
                if word in model.wv
            ]
        else:
            weighted_vectors = []

        if weighted_vectors:
            sentence_vector = np.mean(weighted_vectors, axis=0)
            norm = np.linalg.norm(sentence_vector)
            normalized_vector = (
                sentence_vector / norm if norm > 0 else np.zeros(model.vector_size)
            )
        else:
            normalized_vector = np.zeros(model.vector_size)

        indexid2vec[int(indexid)] = np.asarray(normalized_vector, dtype=np.float32)

    log(f"Finish generating normalized node vectors.")

    return indexid2vec


def gen_relation_onehot(rel2id):
    """Generate one-hot encoding for relations."""
    relvec = torch.nn.functional.one_hot(
        torch.arange(0, len(rel2id.keys())//2), 
        num_classes=len(rel2id.keys())//2
    )
    rel2vec = {}
    for i in rel2id.keys():
        if type(i) is not int:
            rel2vec[i] = relvec[rel2id[i]-1]
            rel2vec[relvec[rel2id[i]-1]] = i
    return rel2vec


def gen_vectorized_graphs(indexid2vec, etype2oh, ntype2oh, split_files, out_dir, logger, cfg):
    """
    Generate vectorized graphs with pre-allocated tensors.
    
    Memory optimization:
    - Pre-allocate tensors based on edge count
    - Avoid appending individual tensors to a list
    - Use atomic writes for safety
    
    Args:
        indexid2vec: dict mapping index_id -> normalized vector
        etype2oh: edge type to one-hot encoding
        ntype2oh: node type to one-hot encoding
        split_files: list of graph folders to process
        out_dir: output directory
        logger: logging handler
        cfg: configuration object
    """
    base_dir = cfg.graph_construction.build_graphs._graphs_dir
    sorted_paths = get_all_files_from_folders(base_dir, split_files)
    
    os.makedirs(out_dir, exist_ok=True)

    for path in tqdm(sorted_paths, "Embedding edges"):
        file = path.split("/")[-1]
        
        # Load single graph at a time
        graph = torch.load(path)

        sorted_edges = list(graph.edges(data=True, keys=True))
        num_edges = len(sorted_edges)
        
        if num_edges == 0:
            continue
        
        # Pre-allocate tensors based on edge count
        # This avoids creating num_edges individual Tensor objects
        dataset = TemporalData()
        
        # Create index tensors directly
        src_indices = [int(u) for u, v, k, attr in sorted_edges]
        dst_indices = [int(v) for u, v, k, attr in sorted_edges]
        timestamps = [int(attr["time"]) for u, v, k, attr in sorted_edges]
        
        # Build message features by iterating edges
        # Use list then tensor construction (can't fully pre-allocate due to varying token lengths)
        msg_features = []
        for u, v, k, attr in sorted_edges:
            msg_features.append(torch.cat([
                ntype2oh[graph.nodes[u]['node_type']],
                torch.from_numpy(indexid2vec[int(u)]),
                etype2oh[attr["label"]],
                ntype2oh[graph.nodes[v]['node_type']],
                torch.from_numpy(indexid2vec[int(v)])
            ]))
        
        # Final tensor stack - only happens once per graph
        dataset.src = torch.tensor(src_indices, dtype=torch.long)
        dataset.dst = torch.tensor(dst_indices, dtype=torch.long)
        dataset.t = torch.tensor(timestamps, dtype=torch.long)
        dataset.msg = torch.stack(msg_features).to(torch.float)
        
        # Clean up intermediate data
        del msg_features
        del src_indices
        del dst_indices
        del timestamps
        
        # Atomic save
        out_path = os.path.join(out_dir, f"{file}.TemporalData.simple")
        temp_path = out_path + ".tmp"
        torch.save(dataset, temp_path)
        os.replace(temp_path, out_path)
        
        # Release graph memory
        del graph
        del dataset
        gc.collect()


def main(cfg):
    logger = get_logger(
        name="embed_edges_by_feature_word2vec",
        filename=os.path.join(cfg.edge_featurization.embed_edges._logs_dir, "embed_edges.log")
    )

    use_node_types = cfg.edge_featurization.embed_nodes.feature_word2vec.use_node_types
    use_cmd =  cfg.edge_featurization.embed_nodes.feature_word2vec.use_cmd
    use_port = cfg.edge_featurization.embed_nodes.feature_word2vec.use_port
    decline_percentage = cfg.edge_featurization.embed_nodes.feature_word2vec.decline_rate

    log("Loading node semantics (metadata cache first, PostgreSQL fallback)...")
    indexid2msg = load_semantic_messages(cfg, use_cmd=use_cmd, use_port=use_port)
    indexid2msg = dict(sorted(indexid2msg.items(), key=lambda item: int(item[0])))

    log("Generating node vectors...")
    feature_word2vec_model_path = cfg.edge_featurization.embed_nodes.feature_word2vec._model_dir + 'feature_word2vec.model'
    indexid2vec = get_indexid2vec(
        indexid2msg=indexid2msg, 
        model_path=feature_word2vec_model_path, 
        use_node_types=use_node_types, 
        decline_percentage=decline_percentage
    )
    
    # Clean up indexid2msg after vectors are generated
    del indexid2msg

    etype2onehot = gen_relation_onehot(rel2id=rel2id)
    ntype2onehot = gen_relation_onehot(rel2id=ntype2id)

    # Vectorize training set
    gen_vectorized_graphs(indexid2vec=indexid2vec,
                          etype2oh=etype2onehot,
                          ntype2oh=ntype2onehot,
                          split_files=cfg.dataset.train_files,
                          out_dir=os.path.join(cfg.edge_featurization.embed_edges._edge_embeds_dir, "train/"),
                          logger=logger,
                          cfg=cfg
                          )

    # Vectorize validation set
    gen_vectorized_graphs(indexid2vec=indexid2vec,
                          etype2oh=etype2onehot,
                          ntype2oh=ntype2onehot,
                          split_files=cfg.dataset.val_files,
                          out_dir=os.path.join(cfg.edge_featurization.embed_edges._edge_embeds_dir, "val/"),
                          logger=logger,
                          cfg=cfg
                          )

    # Vectorize testing set
    gen_vectorized_graphs(indexid2vec=indexid2vec,
                          etype2oh=etype2onehot,
                          ntype2oh=ntype2onehot,
                          split_files=cfg.dataset.test_files,
                          out_dir=os.path.join(cfg.edge_featurization.embed_edges._edge_embeds_dir, "test/"),
                          logger=logger,
                          cfg=cfg
                          )
    
    # Publish the completion marker atomically after every split is saved.
    edge_embeds_dir = cfg.edge_featurization.embed_edges._edge_embeds_dir
    marker_path = os.path.join(edge_embeds_dir, ".preprocess_embed_edges_complete")
    marker_tmp_path = marker_path + ".tmp"
    from datetime import datetime, timezone
    with open(marker_tmp_path, 'w', encoding="utf-8") as f:
        f.write(datetime.now(timezone.utc).isoformat())
    os.replace(marker_tmp_path, marker_path)
    log(f"Embed edges complete. Marker written: {marker_path}")
    
    # Final cleanup
    del indexid2vec
    gc.collect()


if __name__ == '__main__':
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
