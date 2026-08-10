import argparse
import os
import hashlib
import pathlib
import sys
import yaml
from copy import deepcopy
from collections import OrderedDict
from pprint import pprint
from yacs.config import CfgNode as CN
from psycopg2 import extras as ex
import psycopg2

# ================================================================================
# Environment variable names for ORTHRUS C2 configuration
# Priority: CLI/YAML > Environment Variables > Internal Defaults
# ================================================================================
ORTHRUS_ARTIFACT_ROOT_ENV = "ORTHRUS_ARTIFACT_ROOT"
ORTHRUS_DATA_ROOT_ENV = "ORTHRUS_DATA_ROOT"
ORTHRUS_DB_HOST_ENV = "ORTHRUS_DB_HOST"
ORTHRUS_DB_PORT_ENV = "ORTHRUS_DB_PORT"
ORTHRUS_DB_USER_ENV = "ORTHRUS_DB_USER"
ORTHRUS_DB_PASSWORD_ENV = "ORTHRUS_DB_PASSWORD"

# [EDITABLE AREA]: Insert your output path and credentials to the DB
# ================================================================================
ROOT_ARTIFACT_DIR = "./artifacts" # Destination folder for generated files. Will be created if doesn't exist.
ROOT_GROUND_TRUTH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Ground_Truth/darpa/")


DATABASE_DEFAULT_CONFIG = {
     "host": 'postgres',  # Host machine where the db is located
     "user": 'postgres',  # Database user
     "password": 'postgres',  # The password to the database user
     "port": '5432',  # The port number for Postgres
}
# ================================================================================

# --- Dependency graph to follow ---
TASK_DEPENDENCIES = OrderedDict({
     "build_graphs": [],
     "embed_nodes": ["build_graphs"],
     "embed_edges": ["embed_nodes"],
     "gnn_training": ["embed_edges"],
     "gnn_testing": ["gnn_training"],
     "evaluation": ["gnn_testing"],
     "tracing" : ["evaluation"],
})

# --- Tasks, subtasks, and argument configurations ---
TASK_ARGS = {
     "graph_construction": {
          "build_graphs": {
               "used_method": str, # [orthrus | magic]
               "use_all_files": bool,
               "time_window_size": float,
               "use_hashed_label": bool,
               "node_label_features": {
                    "subject": str,  # [type, path, cmd_line]
                    "file": str,  # [type, path]
                    "netflow": str,  # [type, remote_ip, remote_port]
               },
          },
     },
     "edge_featurization": {
          "embed_nodes": {
               "emb_dim": int,
               "used_method": str,
               "use_seed": bool,
               "feature_word2vec": {
                    'show_epoch_loss': bool,
                    'window_size': int,
                    'min_count': int,
                    'use_skip_gram': bool,
                    'num_workers': int,
                    'epochs': int,
                    'compute_loss': bool,
                    'negative': int,
                    'use_node_types': bool,
                    'use_cmd': bool,
                    'use_port': bool,
                    'decline_rate': int,
               },
          },
          "embed_edges": {
               "to_remove": bool, # TODO: remove
          }
     },
     "detection": {
          "gnn_training": {
               "used_method": str, # [ "magic" | "orthrus" | "flash" ]
               "use_seed": bool,
               "num_epochs": int,
               "lr": float,
               "weight_decay": float,
               "resume_checkpoint": str,
               "node_hid_dim": int,
               "node_out_dim": int,
               "encoder": {
                    "backbone": str,  # C7: ["graph_transformer" | "graphsage" | "semantic_mlp"]
                    "use_node_type_in_node_feats": bool,
                    "neighbor_sampling": list,  # [[] | [int, ..., int]]
                    "edge_features": str,  # ["edge_type", "msg", "time_encoding", "none"]
                    "use_node_feats_in_gnn": bool,
                    "neighbor_size": int,
                    "temporal_dim": int,
                    "batch_size": int,
                    "graph_attention": {
                         "dropout": float,
                         "activation": str,
                         "num_heads": int,
                    },
                    "context": {
                         "mode": str,  # ["none" | "recent" | "multiscale"]
                         "multiscale": {
                              "enabled": bool,
                              "candidate_capacity": int,
                              "history_device": str,
                              "scale_quantiles": list,
                              "neighbor_budgets": list,
                              "share_encoder": bool,
                              "fusion": str,  # ["gated" | "equal"]
                              "use_scale_embedding": bool,
                              "gate_hidden_dim": int,
                         },
                    },
               },
               "decoder": {
                    "used_methods": str,
                    "predict_edge_type": {
                         "enabled": bool,
                         "used_method": str,  # ["custom"]
                         "custom": {
                              "dropout": float,
                              "num_layers": int,  # [2 | 3]
                              "activation": str,  # ["sigmoid" | "tanh" | "relu" | "none"]
                         }
                    },
                    "time_gap": {
                         "enabled": bool,
                         "lambda_time": float,
                         "hidden_dim": int,
                         "num_classes": int,
                    },
               },
          },
          "gnn_testing": {
               "threshold_method": str,
          },
          "evaluation": {
               "viz_malicious_nodes": bool,
               "ground_truth_version": str,  # ["darpa_v4"]
               "used_method": str,
               "node_evaluation": {
                    "threshold_method": str,  # ["max_val_loss" | "mean_val_loss"]
                    "use_dst_node_loss": bool,
                    "use_kmeans": bool,
                    "kmeans_top_K": int,
               },
          },
     },
     "attack_reconstruction": {
          "tracing": {
               "used_method": str, #["depimpact"]
               "depimpact": {
                    "used_method": str, #["component" | "shortest_path" | "1-hop" | "2-hop" | "3-hop"]
                    "score_method": str, # ["degree" | "recon_loss" | "degree_recon"]
                    "workers": int,
                    "visualize" : bool,
               },
          },
     },
     "postprocessing":
          {},
}

# C6 post-processing settings are not pipeline tasks and must not take part in
# task-path hashing. They are nevertheless included in defaults, YAML validation,
# and automatic CLI argument registration.
C6_CONFIG_ARGS = {
     "semantic_features": {
          "corpus_scope": str,
     },
     "calibration": {
          "method": str,
          "min_triplet_samples": int,
          "min_type_pair_samples": int,
          "epsilon": float,
     },
     "node_aggregation": {
          "method": str,
          "topk": int,
          "include_dst": bool,
          "score_field": str,
     },
     "node_threshold": {
          "method": str,
          "quantile": float,
     },
     "dataset_view": {
          "mode": str,
     },
}

CONFIG_ARGS = {**TASK_ARGS, **C6_CONFIG_ARGS}

DATASET_DEFAULT_CONFIG = {
     "THEIA_E5": {
          "raw_dir": "/data/",  # NOTE: /path/to/json/files/
          "database": "theia_e5",
          "database_all_file": "theia_e5",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2019-05",
          "start_end_day_range": (8, 18),
          "train_files": ["graph_8", "graph_9", "graph_10"],
          "val_files": ["graph_11"],
          "test_files": ["graph_14", "graph_15"],
          "unused_files": ["graph_12", "graph_13", "graph_16", "graph_17"],
          "ground_truth_relative_path": ["E5-THEIA/node_THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject.csv"],
          "attack_to_time_window" : [
               ["E5-THEIA/node_THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject.csv" , '2019-05-15 14:47:00', '2019-05-15 15:08:00'],
          ]
     },
     "THEIA_E3": {
          "raw_dir": "/data/",  # NOTE: /path/to/json/files/
          "database": "theia_e3",
          "database_all_file": "theia_e3",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2018-04",
          "start_end_day_range": (2, 14),
          "train_files": ["graph_2", "graph_3", "graph_4", "graph_5"],
          "val_files": ["graph_9"],
          "test_files": ["graph_10", "graph_12", "graph_13"],
          "unused_files": ["graph_11"],
          "ground_truth_relative_path": ["E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv",
                                         "E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv",
                                         # "E3-THEIA/node_Phishing_E_mail_Executable_Attachment.csv", # attack failed so we don't use it
                                         # "E3-THEIA/node_Phishing_E_mail_Link.csv" # attack only at network level, not system
                                         ],
          "attack_to_time_window" : [
               ["E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv" , '2018-04-12 12:40:00', '2018-04-12 13:30:00'],
               ["E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv" , '2018-04-10 14:30:00', '2018-04-10 15:00:00'],
          ]
     },
     "CADETS_E5": {
          "raw_dir": "/data/",  # NOTE: /path/to/json/files/
          "database": "cadets_e5",
          "database_all_file": "cadets_e5",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2019-05",
          "start_end_day_range": (8, 18),
          "train_files": ["graph_8", "graph_9", "graph_11"],
          "val_files": ["graph_12"],
          "test_files": ["graph_16", "graph_17"],
          "unused_files": ["graph_15", "graph_10", "graph_13", "graph_14"],
          "ground_truth_relative_path": ["E5-CADETS/node_Nginx_Drakon_APT.csv",
                                         "E5-CADETS/node_Nginx_Drakon_APT_17.csv"],
          "attack_to_time_window" : [
               ["E5-CADETS/node_Nginx_Drakon_APT.csv" , '2019-05-16 09:31:00', '2019-05-16 10:12:00'],
               ["E5-CADETS/node_Nginx_Drakon_APT_17.csv" , '2019-05-17 10:15:00', '2019-05-17 15:33:00'],
          ]
     },
     "CADETS_E3": {
          "raw_dir": "/data/",  # NOTE: /path/to/json/files/
          "database": "cadets_e3",
          "database_all_file": "cadets_e3",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2018-04",
          "start_end_day_range": (2, 14),
          "train_files": ["graph_3", "graph_4", "graph_5", "graph_7", "graph_8", "graph_9", "graph_10"],
          "val_files": ["graph_2"],
          "test_files": ["graph_6", "graph_11", "graph_12", "graph_13"],
          "unused_files": [],
          "ground_truth_relative_path": [
                                         "E3-CADETS/node_Nginx_Backdoor_06.csv",
                                         "E3-CADETS/node_Nginx_Backdoor_12.csv",
                                         "E3-CADETS/node_Nginx_Backdoor_13.csv"],
          "attack_to_time_window": [
               ["E3-CADETS/node_Nginx_Backdoor_06.csv" , '2018-04-06 11:20:00', '2018-04-06 12:09:00'],
               ["E3-CADETS/node_Nginx_Backdoor_12.csv" , '2018-04-12 13:59:00', '2018-04-12 14:39:00'],
               ["E3-CADETS/node_Nginx_Backdoor_13.csv" , '2018-04-13 09:03:00', '2018-04-13 09:16:00'],
          ],
     },
     "CLEARSCOPE_E5": {
          "raw_dir": "/data/",  # NOTE: /path/to/json/files/
          "database": "clearscope_e5",
          "database_all_file": "clearscope_e5",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2019-05",
          "start_end_day_range": (8, 18),
          "train_files": ["graph_8", "graph_9"],
          "val_files": ["graph_11"],
          "test_files": ["graph_14", "graph_15", "graph_17"],
          "unused_files": ["graph_10", "graph_12", "graph_13", "graph_16"],
          "ground_truth_relative_path": [
               "E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv",
               "E5-CLEARSCOPE/node_clearscope_e5_lockwatch_0517.csv",
               "E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv",
          ],
          "attack_to_time_window": [
               ["E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv", '2019-05-15 15:38:00', '2019-05-15 16:19:00'],
               ["E5-CLEARSCOPE/node_clearscope_e5_lockwatch_0517.csv", '2019-05-17 15:48:00', '2019-05-17 16:01:00'],
               ["E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv", '2019-05-17 16:20:00', '2019-05-17 16:28:00'],
          ],
     },
     "CLEARSCOPE_E3": {
          "raw_dir": "/data/",  # NOTE: /path/to/json/files/
          "database": "clearscope_e3",
          "database_all_file": "clearscope_e3",
          "num_node_types": 3,
          "num_edge_types": 10,
          "year_month": "2018-04",
          "start_end_day_range": (2, 14),
          "train_files": ["graph_3", "graph_4", "graph_5", "graph_7", "graph_8", "graph_9", "graph_10"],
          "val_files": ["graph_2"],
          "test_files": ["graph_11", "graph_12"],
          "unused_files": ["graph_6", "graph_13"],
          "ground_truth_relative_path": [
               "E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv",
               # "E3-CLEARSCOPE/node_clearscope_e3_firefox_0412.csv", # due to malicious file downloaded but failed to exec and feture missing, there is no malicious nodes found in database
          ],
          "attack_to_time_window": [
               ["E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv", '2018-04-11 13:54:00', '2018-04-11 14:48:00'],
               # ["E3-CLEARSCOPE/node_clearscope_e3_firefox_0412.csv", '2018-04-12 15:18:00', '2018-04-12 15:25:00'],
          ],
     },
}

def _resolve_artifact_root_from_env():
    """Resolve artifact root from ORTHRUS_ARTIFACT_ROOT env var."""
    env_val = os.environ.get(ORTHRUS_ARTIFACT_ROOT_ENV)
    if env_val:
        return str(env_val)
    return ROOT_ARTIFACT_DIR

def _resolve_data_root_from_env():
    """Resolve data root from ORTHRUS_DATA_ROOT env var."""
    return os.environ.get(ORTHRUS_DATA_ROOT_ENV)

def _resolve_db_config_from_env():
    """Resolve database config from environment variables.

    Priority: Env Var > Internal Default
    Returns dict with host, port, user, password.
    """
    return {
        "host": os.environ.get(ORTHRUS_DB_HOST_ENV, DATABASE_DEFAULT_CONFIG["host"]),
        "port": os.environ.get(ORTHRUS_DB_PORT_ENV, DATABASE_DEFAULT_CONFIG["port"]),
        "user": os.environ.get(ORTHRUS_DB_USER_ENV, DATABASE_DEFAULT_CONFIG["user"]),
        "password": os.environ.get(ORTHRUS_DB_PASSWORD_ENV, DATABASE_DEFAULT_CONFIG["password"]),
    }

def get_default_cfg(args):
     """
     Inits the shared cfg object with default configurations.

     Priority order (highest to lowest):
       1. CLI / YAML (handled by overwrite_cfg_with_args and merge_from_file)
       2. Environment variables (ORTHRUS_*)
       3. Internal defaults
     """
     cfg = CN()

     # Resolve artifact root with env var support
     cfg._artifact_dir = _resolve_artifact_root_from_env()
     cfg._data_root = _resolve_data_root_from_env()  # May be None

     cfg._test_mode = False

     cfg._use_cpu = args.cpu
     cfg._from_weights = args.from_weights
     cfg._from_weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights/")
     cfg._seed = args.seed

     # Pipeline stage control
     cfg.pipeline = CN()
     cfg.pipeline.mode = "full_pipeline"  # C2: "full_pipeline" or "detection_only"
     cfg.pipeline.run_tracing = True  # default True for backward compat; --skip-tracing overrides to False
     if getattr(args, 'skip_tracing', False):
         cfg.pipeline.run_tracing = False

     # C4 model variant; the baseline remains the default.
     cfg.model = CN()
     cfg.model.variant = "orthrus_baseline"

     # Epoch / model selection
     cfg.model_selection = CN()
     cfg.model_selection.method = "min_val_mean_edge_loss"  # ["min_val_mean_edge_loss", "last_epoch"]
     cfg.model_selection.legacy_test_selection_enabled = False  # True=select by test MCC (leakage); DISABLED 鈥?raises ValueError

     # Logging / W&B
     cfg.logging = CN()
     cfg.logging.wandb_mode = None   # None=unset (use --wandb / default), "disabled"/"offline"/"online"

     # Database: we simply create variables for all configurations described in the dict
     cfg.database = CN()
     db_env_config = _resolve_db_config_from_env()
     for attr, value in DATABASE_DEFAULT_CONFIG.items():
         setattr(cfg.database, attr, value)
     # Apply env overrides (env vars take precedence over defaults)
     for attr, value in db_env_config.items():
         setattr(cfg.database, attr, value)

     # Dataset: we simply create variables for all configurations described in the dict
     cfg.dataset = CN()
     cfg.dataset.name = args.dataset
     for attr, value in DATASET_DEFAULT_CONFIG[cfg.dataset.name].items():
          setattr(cfg.dataset, attr, value)

     # C2: Testing configuration
     cfg.testing = CN()
     cfg.testing.include_node_messages = True  # Whether to include node messages in test output

     # C2: Logging configuration
     cfg.logging = CN()
     cfg.logging.wandb_mode = "disabled"  # ["disabled", "offline", "online"]

     # Tasks: we create nested None variables for all arguments
     def create_cfg_recursive(cfg, task_args_dict: dict):
          for task, subtasks in task_args_dict.items():
               if isinstance(subtasks, dict):
                    setattr(cfg, task, CN())
                    task_cfg = getattr(cfg, task)
                    create_cfg_recursive(task_cfg, dict(subtasks.items()))
               else:
                    setattr(cfg, task, None)

     create_cfg_recursive(cfg, CONFIG_ARGS)

     # C8.2: paper experiments fit semantic features on training graphs only.
     # ``official_full_dataset`` remains an explicit compatibility option.
     cfg.semantic_features.corpus_scope = "train_only"

     # C8-A resume path is optional and excluded from the checkpoint config hash.
     cfg.detection.gnn_training.resume_checkpoint = None

     # C4 defaults keep existing configs on the baseline/type-only path.
     cfg.detection.gnn_training.decoder.predict_edge_type.enabled = True
     cfg.detection.gnn_training.decoder.time_gap.enabled = False
     cfg.detection.gnn_training.decoder.time_gap.lambda_time = 0.3
     cfg.detection.gnn_training.decoder.time_gap.hidden_dim = 128
     cfg.detection.gnn_training.decoder.time_gap.num_classes = 6

     # C5 defaults: recent context preserves baseline behavior by default.
     cfg.detection.gnn_training.encoder.backbone = "graph_transformer"
     cfg.detection.gnn_training.encoder.context.mode = "recent"
     cfg.detection.gnn_training.encoder.context.multiscale.enabled = False
     cfg.detection.gnn_training.encoder.context.multiscale.candidate_capacity = 64
     cfg.detection.gnn_training.encoder.context.multiscale.history_device = "cpu"
     cfg.detection.gnn_training.encoder.context.multiscale.scale_quantiles = [0.50, 0.90, 0.99]
     cfg.detection.gnn_training.encoder.context.multiscale.neighbor_budgets = [8, 8, 8]
     cfg.detection.gnn_training.encoder.context.multiscale.share_encoder = True
     cfg.detection.gnn_training.encoder.context.multiscale.fusion = "gated"
     cfg.detection.gnn_training.encoder.context.multiscale.use_scale_embedding = False
     cfg.detection.gnn_training.encoder.context.multiscale.gate_hidden_dim = 64

     # C6 defaults are safe for the baseline variant but do not select MSTC evaluation.
     cfg.calibration.method = "hierarchical_relation"
     cfg.calibration.min_triplet_samples = 100
     cfg.calibration.min_type_pair_samples = 200
     cfg.calibration.epsilon = 1.0e-12
     # C6-B8 preserves baseline input unless a view is explicitly selected.
     cfg.dataset_view.mode = "host_network_full"
     cfg.node_aggregation.method = "topk_mean"
     cfg.node_aggregation.topk = 5
     cfg.node_aggregation.include_dst = True
     cfg.node_aggregation.score_field = "score_calibrated"
     cfg.node_threshold.method = "validation_quantile"
     cfg.node_threshold.quantile = 0.999

     return cfg

def get_runtime_required_args(return_unknown_args=False, args=None):
     parser = argparse.ArgumentParser()
     parser.add_argument('dataset', type=str, help="Name of the dataset")
     parser.add_argument('--model', type=str, help="Name of the model (Orthrus)")
     parser.add_argument('--config', type=str, default=None, metavar='PATH',
                         help="Explicit YAML configuration file. When omitted, uses config/<model>.yml.")
     parser.add_argument('--model.variant', type=str, choices=["orthrus_baseline", "mstc"], default=None,
                        help="Model implementation variant.")
     # Note: detection.gnn_training.encoder.context.mode is added automatically by add_cfg_args_to_parser
     parser.add_argument('--wandb', action="store_true", help="Whether to submit logs to wandb")
     parser.add_argument('--exp', type=str, default="", help="Name of the experiment")
     parser.add_argument('--tags', type=str, default="", help="Name of the tag to use. Tags are used to group runs together")
     parser.add_argument('--cpu', action="store_true", help="Whether to run on CPU rather than GPU")
     parser.add_argument('--run_from_training', action="store_true", help="Runs Orthrus from training when graphs are arleady preprocessed")
     parser.add_argument('--from_weights', action="store_true", help="Whether to load Orthrus from pkl weights")
     parser.add_argument('--seed', type=int, default=0, help="Manual seed")

     parser.add_argument('--show_attack', type=int, help="Number of attack for plotting", default=0)
     parser.add_argument('--gt_type', type=str, help="Type of ground truth", default="orthrus")
     parser.add_argument('--plot_gt', type=bool, help="If we plot ground truth", default=False)
     parser.add_argument('--stages', type=str, default=None,
                         help="Comma-separated pipeline stages to run (e.g. 'preprocess,train,test,evaluate'). "
                              "Standard stages: preprocess, train, test, evaluate, trace. "
                              "'all' runs everything. If omitted, behavior depends on --run_from_training.")
     parser.add_argument('--preprocess-substages', type=str, default=None,
                         help="Comma-separated preprocess substages to run within --stages preprocess. "
                              "Valid substages: build_graphs, embed_nodes, embed_edges. "
                              "Default (None): runs all three in order. "
                              "Examples: 'build_graphs' or 'embed_nodes,embed_edges'.")
     parser.add_argument('--force-preprocess', action='store_true',
                         help="Explicitly rerun selected preprocessing substages even when "
                              "validated completion markers exist.")
     parser.add_argument('--skip-tracing', action='store_true',
                         help="Skip attack reconstruction (tracing) stage. Overrides pipeline.run_tracing to False.")
     parser.add_argument('--artifact-root', type=str, default=None,
                         dest='artifact_root', metavar='PATH',
                         help="Root directory for all artifacts. "
                              "Takes precedence over the ORTHRUS_ARTIFACT_ROOT environment variable. "
                              "Default: ./artifacts (or ORTHRUS_ARTIFACT_ROOT if set).")

     # All args in the cfg can be also set in the arg parser from CLI
     parser = add_cfg_args_to_parser(CONFIG_ARGS, parser)

     try:
          args, unknown_args = parser.parse_known_args(args)
     except:
          parser.print_help()
          sys.exit(1)

     args.model = "orthrus"

     if return_unknown_args:
          return args, unknown_args
     return args

def overwrite_cfg_with_args(cfg, args):
     """
     The framework can be also parametrized using the CLI args.
     These args are priorited compared to yml file parameters.
     This function simply overwrites the cfg with the parameters
     given within args.

     To override a parameter in cfg, use a dotted style:
     ```python orthrus.py --detection.gnn_training.seed=42```
     """
     for arg, value in args.__dict__.items():
          if "." in arg and value is not None:
               cfg_ptr = cfg
               dots = arg.split(".")
               path, attr_name = dots[:-1], dots[-1]

               for attr in path:
                    cfg_ptr = getattr(cfg_ptr, attr)
               setattr(cfg_ptr, attr_name, value)

def set_task_paths(cfg):
     subtask_to_hash = {}
     # Directories common to all tasks
     for task, subtask in TASK_ARGS.items():
          task_cfg = getattr(cfg, task)

          # We first compute a unique hash for each usbtask
          for subtask_name, subtask_args in subtask.items():
               subtask_cfg = getattr(task_cfg, subtask_name)
               restart_values = flatten_arg_values(subtask_cfg)
               # Semantic corpus scope changes Word2Vec training, not graph
               # construction. Add it only to embed_nodes; embed_edges picks
               # it up through the declared upstream dependency.
               if subtask_name == "embed_nodes":
                    restart_values.append(
                         "semantic_features.corpus_scope="
                         f"{cfg.semantic_features.corpus_scope}"
                    )

               clean_hash_args = ["".join([c for c in str(restart_value) if c not in set(" []\"\'")]) for restart_value in restart_values]
               final_hash_string = ",".join(clean_hash_args)
               final_hash_string = hashlib.sha256(final_hash_string.encode("utf-8")).hexdigest()

               subtask_to_hash[subtask_name] = final_hash_string

     # Then, for each subtask, we want its unique hash to also depend from its previous dependencies' hashes.
     # For example, if I run the same subtask A two times, with two different subtasks B and C, the results
     # would be different and would be stored in the same folder A if we don't consider the hash of B and C.
     for task, subtask in TASK_ARGS.items():
          task_cfg = getattr(cfg, task)
          for subtask_name, subtask_args in subtask.items():
               subtask_cfg = getattr(task_cfg, subtask_name)
               deps = sorted(list(get_dependees(subtask_name, TASK_DEPENDENCIES, set())))
               deps_hash = "".join([subtask_to_hash[dep] for dep in deps])

               final_hash_string = deps_hash + subtask_to_hash[subtask_name]
               final_hash_string = hashlib.sha256(final_hash_string.encode("utf-8")).hexdigest()

               if task in ["graph_construction", "edge_featurization"]:
                    subtask_cfg._task_path = os.path.join(cfg._artifact_dir, task, cfg.dataset.name, subtask_name, final_hash_string)
               else:
                    subtask_cfg._task_path = os.path.join(cfg._artifact_dir, task, subtask_name, final_hash_string, cfg.dataset.name)

               # The directory to save logs related to the graph_construction task
               subtask_cfg._logs_dir = os.path.join(subtask_cfg._task_path, "logs/")
               os.makedirs(subtask_cfg._logs_dir, exist_ok=True)

     # graph_construction paths
     cfg.graph_construction.build_graphs._graphs_dir = os.path.join(cfg.graph_construction.build_graphs._task_path, "nx/")
     cfg.graph_construction.build_graphs._tw_labels = os.path.join(cfg.graph_construction.build_graphs._task_path, "tw_labels/")
     cfg.graph_construction.build_graphs._node_id_to_path = os.path.join(cfg.graph_construction.build_graphs._task_path, "node_id_to_path/")

     # Featurization paths
     cfg.edge_featurization.embed_nodes.feature_word2vec._model_dir = os.path.join(cfg.edge_featurization.embed_nodes._task_path, "word2vec_models/")
     cfg.edge_featurization.embed_edges._edge_embeds_dir = os.path.join(cfg.edge_featurization.embed_edges._task_path, "edge_embeds/")

     # Detection paths
     cfg.detection.gnn_training._trained_models_dir = os.path.join(cfg.detection.gnn_training._task_path, "trained_models/")
     cfg.detection.gnn_testing._edge_losses_dir = os.path.join(cfg.detection.gnn_testing._task_path, "edge_losses/")
     cfg.detection.evaluation.node_evaluation._precision_recall_dir = os.path.join(cfg.detection.evaluation._task_path, "precision_recall_dir/") # TODO: move to cfg.detection._precision_recall_dir
     cfg.detection.evaluation._evaluation_results_dir = os.path.join(cfg.detection.evaluation._task_path, "evaluation_results/")

     # Ground Truth paths
     cfg._ground_truth_dir = os.path.join(ROOT_GROUND_TRUTH_DIR, cfg.detection.evaluation.ground_truth_version + '/')

     # Triage paths
     cfg.attack_reconstruction.tracing._tracing_graph_dir = os.path.join(cfg.attack_reconstruction.tracing._task_path, "tracing_graphs")

     # C2: Metadata cache directory (derived from graph_construction task path)
     cfg._metadata_dir = os.path.join(cfg.graph_construction.build_graphs._task_path, "metadata")
     os.makedirs(cfg._metadata_dir, exist_ok=True)

     # TODO
     cfg.postprocessing._task_path = None

def validate_yml_file(yml_file: str):
     with open(yml_file, 'r') as file:
          user_config = yaml.safe_load(file)

     def validate_config(user_config, tasks, path=None):
          if path is None:
               path = []
          if not user_config:
               raise ValueError(f"Config at {' > '.join(path)} is empty but should not be.")

          for key, sub_tasks in tasks.items():
               if key in user_config:
                    sub_config = user_config[key]
                    if isinstance(sub_tasks, dict):
                         # Recursive check for sub-dictionaries
                         validate_config(sub_config, sub_tasks, path + [key])
                    else:
                         # Check for None values in parameters
                         if sub_config is None:
                              raise ValueError(f"Parameter '{' > '.join(path + [key])}' should not be None.")
                              # Optional: check for type correctness
                         if not isinstance(sub_config, sub_tasks):
                              raise TypeError(f"Parameter '{' > '.join(path + [key])}' should be of type {sub_tasks.__name__}.")

     validate_config(user_config, CONFIG_ARGS)
     print(f"YAML configuration file \"{yml_file.split('/')[-1]}\" is valid")

def check_args(args):
     available_models = os.listdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config"))
     if not any([args.model in model for model in available_models]):
          raise ValueError(f"Unknown model {args.model}. Available models are {available_models}")

     available_datasets = DATASET_DEFAULT_CONFIG.keys()
     if args.dataset not in available_datasets:
          raise ValueError(f"Unknown dataset {args.dataset}. Available datasets are {available_datasets}")

def check_task_dependency_graph(yml_file: str, base_yml_file: str = None):
     """Validate pipeline dependencies after resolving an experiment overlay.

     Experiment YAML files intentionally describe only their experimental
     difference. Their omitted pipeline fields are inherited from
     ``config/orthrus.yml``; validating an overlay in isolation incorrectly
     rejected valid backbone and time-task configurations.
     """
     with open(yml_file, 'r') as file:
          user_config = yaml.safe_load(file) or {}

     if base_yml_file is not None:
          with open(base_yml_file, 'r') as file:
               base_config = yaml.safe_load(file) or {}
     else:
          base_config = {}

     # The dependency graph contains only pipeline subtasks. Select those
     # declared by the base configuration or the overlay, not arbitrary
     # values such as ``model.variant``.
     subtasks = set()
     for task, task_subtasks in TASK_ARGS.items():
          for source in (base_config, user_config):
               section = source.get(task, {}) if isinstance(source, dict) else {}
               if isinstance(section, dict):
                    subtasks.update(name for name in section if name in task_subtasks)

     deps = TASK_DEPENDENCIES
     subtask_set = subtasks

     def has_all_dependencies(task):
          return all(dependency in subtask_set and has_all_dependencies(dependency)
               for dependency in deps.get(task, []))

     dependencies_ok = all(has_all_dependencies(subtask) for subtask in subtasks)
     if dependencies_ok:
          print(f"Task dependency graph is valid: {sorted(subtasks)}")
          # log("\nYAML configuration")
          # log(user_config)
     else:
          raise ValueError(("The requested subtasks don't respect the subtask dependency graph."
               f"Tasks: {subtasks}\nTask dependency graph: {deps}"))

def _validate_wandb_mode(mode: str) -> str:
    """Validate wandb mode, return validated mode string."""
    valid_modes = ("disabled", "offline", "online")
    mode_str = str(mode).strip().lower()
    if mode_str not in valid_modes:
        raise ValueError(
            f"Invalid logging.wandb_mode={mode!r}. Allowed values: {', '.join(valid_modes)}"
        )
    return mode_str

def _validate_pipeline_mode(mode: str) -> str:
    """Validate pipeline mode, return validated mode string."""
    valid_modes = ("full_pipeline", "detection_only")
    mode_str = str(mode).strip().lower()
    if mode_str not in valid_modes:
        raise ValueError(
            f"Invalid pipeline.mode={mode!r}. Allowed values: {', '.join(valid_modes)}"
        )
    return mode_str

def _validate_corpus_scope(scope: str) -> str:
    """Validate corpus scope, return validated scope string."""
    valid_scopes = ("official_full_dataset", "train_only")
    scope_str = str(scope).strip().lower()
    if scope_str not in valid_scopes:
        raise ValueError(
            f"Invalid semantic_features.corpus_scope={scope!r}. Allowed values: {', '.join(valid_scopes)}"
        )
    return scope_str
def _validate_dataset_view_mode(mode: str) -> str:
    valid_modes = ("host_only", "host_network_structure", "host_network_full")
    mode_str = str(mode).strip().lower()
    if mode_str not in valid_modes:
        raise ValueError("Invalid dataset_view.mode=%r. Allowed values: %s" % (mode, ", ".join(valid_modes)))
    return mode_str


def _validate_calibration_method(method: str) -> str:
    valid_methods = ("global_empirical", "relation_triplet", "hierarchical_relation")
    method_str = str(method).strip().lower()
    if method_str not in valid_methods:
        raise ValueError(
            "Invalid calibration.method=%r. Allowed values: %s"
            % (method, ", ".join(valid_methods))
        )
    return method_str


def get_yml_cfg(args):
     # Checks that CLI args are OK
     check_args(args)

     # Inits with default configurations
     cfg = get_default_cfg(args)

     # Experiment YAML files are overlays. Resolve the project baseline first
     # so every runnable experiment inherits a complete ORTHRUS configuration.
     root_path = pathlib.Path(__file__).parent.parent.resolve()
     base_yml_file = root_path / "config" / "orthrus.yml"

     # Checks that all configurations are valid (not set to None)
     config_path = getattr(args, "config", None)
     if config_path is None:
          yml_file = base_yml_file
     else:
          yml_file = pathlib.Path(config_path).expanduser()
          if not yml_file.is_file():
               raise FileNotFoundError(
                    f"Config file does not exist or is not a regular file: {yml_file}"
               )
          yml_file = yml_file.resolve()
     validate_yml_file(str(yml_file))

     # Internal defaults < base config < experiment YAML < dotted CLI args.
     # Avoid merging the base file twice for the legacy default invocation.
     if yml_file != base_yml_file:
          cfg.merge_from_file(str(base_yml_file))
     cfg.merge_from_file(str(yml_file))

# Overwrites args to the cfg
     overwrite_cfg_with_args(cfg, args)

     # C8-B runtime checkpoint routing. These values are supplied by the
     # unified experiment wrapper and intentionally do not alter legacy CLI
     # behaviour when absent.
     resume_checkpoint = getattr(args, "resume_checkpoint", None)
     if resume_checkpoint is not None:
          cfg.detection.gnn_training.resume_checkpoint = resume_checkpoint
     cfg._inference_checkpoint = getattr(args, "inference_checkpoint", None)

     # Handle --artifact-root explicitly (non-dotted CLI arg, not processed by overwrite_cfg_with_args)
     artifact_root_raw = getattr(args, "artifact_root", None)
     if artifact_root_raw is not None:
         from artifact_paths import resolve_artifact_root
         cfg._artifact_root_raw = artifact_root_raw

     # C2: Apply environment variable overrides (after CLI but before final paths)
     env_artifact = os.environ.get(ORTHRUS_ARTIFACT_ROOT_ENV)
     if artifact_root_raw is None and env_artifact:
         cfg._artifact_dir = str(env_artifact)

     env_data_root = os.environ.get(ORTHRUS_DATA_ROOT_ENV)
     if env_data_root:
         cfg._data_root = str(env_data_root)

     env_db_host = os.environ.get(ORTHRUS_DB_HOST_ENV)
     if env_db_host:
         cfg.database.host = env_db_host

     env_db_port = os.environ.get(ORTHRUS_DB_PORT_ENV)
     if env_db_port:
         cfg.database.port = str(env_db_port)  # Keep as string for psycopg2 compatibility

     env_db_user = os.environ.get(ORTHRUS_DB_USER_ENV)
     if env_db_user:
         cfg.database.user = env_db_user

     env_db_password = os.environ.get(ORTHRUS_DB_PASSWORD_ENV)
     if env_db_password:
         cfg.database.password = env_db_password

     # C2: Validate new configuration options
     if hasattr(cfg, "logging") and hasattr(cfg.logging, "wandb_mode"):
         cfg.logging.wandb_mode = _validate_wandb_mode(cfg.logging.wandb_mode)

     if hasattr(cfg, "dataset_view") and hasattr(cfg.dataset_view, "mode"):
         cfg.dataset_view.mode = _validate_dataset_view_mode(cfg.dataset_view.mode)

     if hasattr(cfg, "calibration") and hasattr(cfg.calibration, "method"):
         cfg.calibration.method = _validate_calibration_method(cfg.calibration.method)

     if hasattr(cfg, "pipeline") and hasattr(cfg.pipeline, "mode"):
         cfg.pipeline.mode = _validate_pipeline_mode(cfg.pipeline.mode)

     if hasattr(cfg, "semantic_features") and hasattr(cfg.semantic_features, "corpus_scope"):
         cfg.semantic_features.corpus_scope = _validate_corpus_scope(cfg.semantic_features.corpus_scope)

     # Checks args after all overrides are applied
     check_task_dependency_graph(str(yml_file), str(base_yml_file))

     # Based on the defined restart args, computes a unique path on disk
     # to store the files of each task
     set_task_paths(cfg)

     return cfg

def get_dependencies(sub: str, dependencies: dict, result_set: set):
     """
     Returns the set of the subtasks happening after `sub`.
     """
     def helper(sub):
          for subtask, deps in dependencies.items():
               if sub in deps:
                    result_set.add(subtask)
                    helper(subtask)
     helper(sub)
     return result_set

def get_dependees(sub: str, dependencies: dict, result_set: set):
     """
     Returns the set of the subtasks happening before `sub`.
     """
     dependencies = OrderedDict(sorted(dependencies.items(), reverse=True))

     def helper(sub):
          for subtask, deps in dependencies.items():
               if sub == subtask:
                    if len(deps) > 0:
                         dep = deps[0]
                         result_set.add(dep)
                         helper(dep)
     helper(sub)
     return result_set

def flatten_arg_values(cfg):
     def helper(dict_or_val, flatten_list):
          if isinstance(dict_or_val, dict):
               for key, value in dict_or_val.items():
                    if isinstance(value, dict):
                         helper(value, flatten_list)
                    else:
                         helper(f"{key}={value}", flatten_list)
          else:
               flatten_list.append(dict_or_val)

     flatten_list = []
     helper(cfg, flatten_list)
     return flatten_list

def add_cfg_args_to_parser(cfg, parser):
     def str2bool(v):
          if isinstance(v, bool):
               return v
          elif v == "None":
               return None
          if v.lower() in ('true'):
               return True
          elif v.lower() in ('false'):
               return False
          else:
               raise argparse.ArgumentTypeError('Boolean value expected.')

     def nested_dict_to_separator_dict(nested_dict, separator='.'):
          def _create_separator_dict(x, key='', separator_dict={}, keys_to_ignore=[]):
               if isinstance(x, dict):
                    for k, v in x.items():
                         kk = f'{key}{separator}{k}' if key else k
                         _create_separator_dict(x[k], kk, keys_to_ignore=keys_to_ignore)
               else:
                    if not any([ignore in key for ignore in keys_to_ignore]):
                         separator_dict[key] = x
               return separator_dict

          return _create_separator_dict(deepcopy(nested_dict))

     separator_dict = nested_dict_to_separator_dict(cfg)

     for k, v in separator_dict.items():
          is_bool = v == type(True)
          dtype = str2bool if is_bool else v
          parser.add_argument(f'--{k}', type=dtype)

     return parser

def get_darpa_tc_node_feats_from_cfg(cfg):
    features = cfg.graph_construction.build_graphs.node_label_features
    return {
        "subject": list(map(lambda x: x.strip(), features.subject.split(","))),
        "file": list(map(lambda x: x.strip(), features.file.split(","))),
        "netflow": list(map(lambda x: x.strip(), features.netflow.split(","))),
    }

########################################################
#
#               Graph semantics
#
########################################################

# The directions of the following edge types need to be reversed
edge_reversed = [
     'EVENT_EXECUTE',
     'EVENT_LSEEK',
     'EVENT_MMAP',
     'EVENT_OPEN',
     'EVENT_ACCEPT',
     'EVENT_READ',
     'EVENT_RECVFROM',
     'EVENT_RECVMSG',
     'EVENT_READ_SOCKET_PARAMS',
     'EVENT_CHECK_FILE_ATTRIBUTES'
]

# The following edges are not considered to construct the
# temporal graph for experiments.
exclude_edge_type= set([
     'EVENT_FCNTL',                          # EVENT_FCNTL does not have any predicate
     'EVENT_OTHER',                          # EVENT_OTHER does not have any predicate
     'EVENT_ADD_OBJECT_ATTRIBUTE',           # This is used to add attributes to an object that was incomplete at the time of publish
     'EVENT_FLOWS_TO',                       # No corresponding system call event
])

rel2id = {
        1: 'EVENT_CONNECT',
        'EVENT_CONNECT': 1,
        2: 'EVENT_EXECUTE',
        'EVENT_EXECUTE': 2,
        3: 'EVENT_OPEN',
        'EVENT_OPEN': 3,
        4: 'EVENT_READ',
        'EVENT_READ': 4,
        5: 'EVENT_RECVFROM',
        'EVENT_RECVFROM': 5,
        6: 'EVENT_RECVMSG',
        'EVENT_RECVMSG': 6,
        7: 'EVENT_SENDMSG',
        'EVENT_SENDMSG': 7,
        8: 'EVENT_SENDTO',
        'EVENT_SENDTO': 8,
        9: 'EVENT_WRITE',
        'EVENT_WRITE': 9,
        10: 'EVENT_CLONE',
        'EVENT_CLONE': 10,
    }

ntype2id ={
     1: 'subject',
     'subject': 1,
     2: 'file',
     'file': 2,
     3: 'netflow',
     'netflow': 3,
}
