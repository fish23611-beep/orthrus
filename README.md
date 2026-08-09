[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.14641605-ed6a2f?style=flat&labelColor=gray)](https://doi.org/10.5281/zenodo.14641605)


# ORTHRUS: Achieving High Quality of Attribution in Provenance-based Intrusion Detection Systems

This repo contains the official code of the [Orthrus paper](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-103-jiang-baoxiang.pdf).

## Citing our work

```
@inproceedings{jian2025,
	title={{ORTHRUS: Achieving High Quality of Attribution in Provenance-based Intrusion
	Detection Systems}},
	author={Jiang, Baoxiang and Bilot, Tristan  and El Madhoun, Nour and Al Agha, Khaldoun  and Zouaoui, Anis and Iqbal, Shahrear and Han, Xueyuan and Pasquier, Thomas},
	booktitle={Security Symposium (USENIX Sec'25)},
	year={2025},
	organization={USENIX}
}
```

## Updates

[2025.06.06] Orthrus is now available in [PIDSMaker](https://github.com/ubc-provenance/PIDSMaker)!

[2025.06.05] Orthrus' weights are available.

[2025.06.04] Installation guidelines are now simplified. The DARPA TC databases can be directly downloaded and installed locally. No need to fill them locally anymore.

## Setup

### Clone the repo with submodules
```
git clone --recurse-submodules https://github.com/ubc-provenance/orthrus.git
```

### 10-min install of Docker and Datasets

We have made the installation of DARPA TC/OpTC easy and fast, simply follow [these guidelines](https://github.com/ubc-provenance/PIDSMaker/blob/velox/settings/ten-minute-install.md).

## Run experiments

The following commands should be executed within the `pids` container.

### Reproduce results from the paper

Launching Orthrus is as simple as running:

```shell
python src/orthrus.py [dataset] [config args...]
```

Running `orthrus.py` will run by default the `graph_construction`, `edge_featurization`, `detection` and `attack_reconstruction` tasks configured within the `config/orthrus.yml` file. This configuration can be updated directly in the YML file or from the CLI, as shown above.

> [!NOTE]
> The original results could not be exactly replicated due to a missing PYTHONHASHSEED affecting Gensim's Word2Vec, though the following experiments yield similar results in most cases.

#### Expected results
| Name             | TP  | FP  | TN       | FN  | Precision | MCC       |
|------------------|-----|-----|----------|-----|-----------|-----------|
| CADETS_E3_full  | 22  | 10  | 268,075   | 46  | 0.69   | 0.47   |
| CADETS_E3_ano   | 15   | 0   | 268,085   | 53  | 1.00   | 0.47   |
| THEIA_E3_full  | 22  | 0  | 699,177   | 96  | 1.00   | 0.43   |
| THEIA_E3_ano    | 2   | 0   | 699,177   | 116 | 1.00   | 0.13   |
| CADETS_E5_full  | 3   | 1318  | 3,132,823  | 120 | 0.00   | 0.01   |
| CADETS_E5_ano   | 1   | 2   | 3,134,139  | 122 | 0.33   | 0.05   |
| THEIA_E5_full  | 13  | 2   | 747,381   | 56  | 0.86   | 0.40   |
| THEIA_E5_ano    | 2   | 0   | 747,383   | 67  | 1.00   | 0.17   |
| CLEARSCOPE_E3_full  | 1   | 647   | 110,715   | 40 | 0.00  | 0.00 |
| CLEARSCOPE_E3_ano | 1 | 5 | 111,357 | 40  | 0.17  | 0.06  |
| CLEARSCOPE_E5_full  | 4  | 8   | 150,666 | 47  | 0.33   | 0.16   |
| CLEARSCOPE_E5_ano | 2   | 5   | 150,669 | 49  | 0.29   | 0.10   |


#### Experiments

These experiments use pre-trained weights of Orthrus.

**CADETS_E3**
```
PYTHONHASHSEED=0 python src/orthrus.py CADETS_E3 --from_weights --detection.gnn_training.encoder.graph_attention.dropout=0.25 --detection.gnn_training.node_hid_dim=256 --detection.gnn_training.node_out_dim=256 --detection.gnn_training.lr=0.001 --detection.gnn_training.num_epochs=20 --seed=4
```

**THEIA_E3**
```
PYTHONHASHSEED=0 python src/orthrus.py THEIA_E3 --from_weights --detection.gnn_training.encoder.graph_attention.dropout=0.1 --seed=2
```

**CLEARSCOPE_E3**
```
PYTHONHASHSEED=0 python src/orthrus.py CLEARSCOPE_E3 --from_weights --graph_construction.build_graphs.time_window_size=1.0 --detection.gnn_training.encoder.graph_attention.dropout=0.1 --seed=2
```

**CADETS_E5**
```
PYTHONHASHSEED=0 python src/orthrus.py CADETS_E5 --from_weights --detection.gnn_training.node_out_dim=128 --detection.gnn_training.lr=0.0001 --detection.gnn_training.encoder.graph_attention.dropout=0.1 --graph_construction.build_graphs.time_window_size=1.0
```

**THEIA_E5**
```
PYTHONHASHSEED=0 python src/orthrus.py THEIA_E5 --from_weights
```

**CLEARSCOPE_E5**
```
PYTHONHASHSEED=0 python src/orthrus.py CLEARSCOPE_E5 --from_weights --detection.gnn_training.lr=0.0001 --detection.gnn_training.encoder.graph_attention.dropout=0.1 --detection.gnn_training.node_out_dim=64
```

### Subsequent runs

When run once, datasets are preprocessed and stored in the `ROOT_ARTIFACT_DIR` path within `config.py`. There is thus no need to recompute them. To avoid re-computing the `graph_construction` and `edge_featurization` tasks, Orthrus can be run directly from the `detection` task using the arg `--run_from_training`.

```shell
python src/orthrus.py CADETS_E3 --run_from_training
```


## MSTC-PIDS experiment workflow (C8)

This section documents the current experiment interfaces. The paper scope for
this matrix is `THEIA_E3` and `THEIA_E5`. The code and synthetic/test coverage
support both datasets, but the full real-data, five-seed matrix has not been run
as part of C8.

### Environment, data, and detection-only mode

The Dockerfile remains the reference environment. For Colab, run the notebooks
in the order listed below; `00_colab_environment.ipynb` preserves Colab's
existing PyTorch/CUDA installation when it is compatible and installs PyG for
that detected combination. Set the portable roots once:

```bash
export ORTHRUS_ARTIFACT_ROOT=/path/to/persistent/artifacts
export ORTHRUS_DATA_ROOT=/path/to/datasets
```

Initial THEIA preprocessing may require a restored PostgreSQL database. Database
credentials are read from `ORTHRUS_DB_HOST`, `ORTHRUS_DB_PORT`,
`ORTHRUS_DB_USER`, and `ORTHRUS_DB_PASSWORD`; do not place credentials in YAML or
notebooks. Once the TemporalData/preprocessing artifacts, metadata cache,
configuration, and code are available, `train`, `test`, and `evaluate` run in
`detection_only` mode without PostgreSQL. This does not mean that preprocessing
itself is database-free.

Word2Vec corpus scope is either `official_full_dataset` or `train_only`.
Separately trained E3 and E5 Word2Vec spaces are not a zero-shot transfer setup
and must not be reported as one.

### Single experiments

MSTC-PIDS Full:

```bash
python src/experiments/run_experiment.py \
  --dataset THEIA_E3 \
  --config config/experiments/mstc_full.yml \
  --seed 0 \
  --artifact-root "$ORTHRUS_ARTIFACT_ROOT"
```

ORTHRUS-ano baseline:

```bash
python src/experiments/run_experiment.py \
  --dataset THEIA_E3 \
  --config config/experiments/baseline.yml \
  --seed 0 \
  --artifact-root "$ORTHRUS_ARTIFACT_ROOT"
```

Both commands execute the configured detection-only pipeline. Preprocessing
artifacts must already exist; the experiment YAML files intentionally do not
start PostgreSQL or preprocessing.

### Checkpoint resume and inference

A structured training resume includes `train` in `--stages`:

```bash
python src/experiments/run_experiment.py \
  --dataset THEIA_E3 \
  --config config/experiments/mstc_full.yml \
  --seed 0 \
  --artifact-root "$ORTHRUS_ARTIFACT_ROOT" \
  --stages train,test,evaluate \
  --checkpoint /path/to/checkpoint.pt
```

Test/evaluate-only uses the same flag but omits `train`:

```bash
python src/experiments/run_experiment.py \
  --dataset THEIA_E3 \
  --config config/experiments/mstc_full.yml \
  --seed 0 \
  --artifact-root "$ORTHRUS_ARTIFACT_ROOT" \
  --stages test,evaluate \
  --checkpoint /path/to/checkpoint.pt
```

Structured checkpoints save model, optimizer, epoch, configuration hash, and
Python/NumPy/Torch RNG state (plus scheduler state when present). Temporal
history is deliberately reconstructed by chronological replay. Legacy
`state_dict.pkl`/model-only checkpoints remain compatible with inference, but
they are not complete training-resume checkpoints.

### Experiment matrix

The matrix runner uses stable dataset × config × seed identities, skips valid
completed markers by default, records failures, and continues with later runs:

```bash
python src/experiments/run_matrix.py \
  --datasets THEIA_E3,THEIA_E5 \
  --configs config/experiments/baseline.yml,config/experiments/mstc_full.yml \
  --seeds 0,1,2,3,4 \
  --artifact-root "$ORTHRUS_ARTIFACT_ROOT"
```

Do not automatically shrink batch size, candidate capacity, neighbor budgets,
or hidden dimensions after OOM. In the causal micro-batch path these changes can
alter context. Record the run as failed, choose one revised configuration
manually, and rerun every fair comparison with that same configuration.

### Authoritative experiment mapping

Main results:

| Reported model | Configuration |
|---|---|
| ORTHRUS-ano | `baseline.yml` |
| Semantic MLP | `backbone_mlp.yml` |
| GraphSAGE baseline | `backbone_graphsage_baseline.yml` |
| MSTC-PIDS Full | `mstc_full.yml` |
| GraphSAGE + MSTC | `backbone_graphsage.yml` |

A0–A6:

| ID | Configuration | Exact meaning |
|---|---|---|
| A0 | `baseline.yml` | ORTHRUS-ano baseline |
| A1 | `ablation_no_multiscale.yml` | Recent-24 |
| A2 | `ablation_no_gate.yml` | equal fusion |
| A3 | `ablation_no_time.yml` | `lambda_time=0` |
| A4 | `ablation_no_calibration.yml` | raw score + validation threshold |
| A5 | `ablation_no_topk.yml` | mean aggregation |
| A6 | `mstc_full.yml` | full model |

The remaining formal groups are direct YAML lists, not notebook-side model
definitions:

| Group | Configurations |
|---|---|
| Multi-scale | `multiscale_recent20.yml`, `multiscale_recent24.yml`, `multiscale_single_window.yml`, `multiscale_equal.yml`, `multiscale_gate.yml` |
| Time task | `time_type_only.yml`, `time_time_only.yml`, `time_joint.yml` |
| Calibration | `calibration_max.yml`, `calibration_quantile.yml`, `calibration_kmeans.yml`, `calibration_global_p.yml`, `calibration_relation.yml`, `calibration_hierarchical.yml` |
| Backbone | `backbone_graphtransformer.yml`, `backbone_graphsage_baseline.yml`, `backbone_graphsage.yml`, `backbone_mlp.yml` |
| Dataset view | `host_only.yml`, `host_network_structure.yml`, `host_network_full.yml` |
| Efficiency | `baseline.yml`, `efficiency_multiscale.yml`, `efficiency_multiscale_time.yml`, `mstc_full.yml` |

### Artifacts and result collection

A direct single run uses the following layout. Stage directories are created
lazily:

```text
<artifact-root>/<dataset>/runs/<model-variant>/seed_<seed>/
  environment.json
  config_resolved.yml
  runtime.json
  checkpoints/
  edge_scores/
  node_scores/metrics.json
```

A matrix isolates configs with a path-derived config ID and owns separate run
status/result metadata:

```text
<artifact-root>/
  matrix_artifacts/<config-id>/<dataset>/runs/<model-variant>/seed_<seed>/...
  results/
    run_status/<dataset>/<config-id>/seed_<seed>/run_status.json
    matrix_summary.json
    all_runs.csv
    main_results.csv
    ablation_results.csv
    calibration_results.csv
    efficiency_results.csv
```

Collect raw run records and then export paper tables:

```bash
python src/experiments/collect_results.py \
  --artifact-root "$ORTHRUS_ARTIFACT_ROOT"

python src/experiments/export_tables.py \
  --artifact-root "$ORTHRUS_ARTIFACT_ROOT"
```

Failed seeds are listed and are not inserted as zero into mean/std. Undefined
metrics remain NaN. Standard deviation is the sample standard deviation
(`ddof=1`), and each metric's best value follows its declared higher/lower
direction.

### Colab notebook order

#### Recommended: All-in-One Master Notebook

**正式 Colab 实验推荐使用**：
```
notebooks/ORTHRUS_MSTC_PIDS_AllInOne_Colab.ipynb
```

这是一个单一的 Master Notebook，在同一个 Colab Runtime 中依次支持：

- Google Drive 挂载与目录创建
- 冻结代码版本 checkout（固定到 `mstc-pids-c1-c8-exp-v1`）
- GPU/CUDA 验证
- Python/PyG 依赖幂等安装
- 环境记录
- THEIA 数据检查
- PostgreSQL 安装/启动/恢复
- Preprocessing
- Baseline smoke test
- 主模型矩阵（ORTHRUS-ano + MSTC-PIDS Full）
- 消融与专项实验
- Checkpoint resume
- 结果收集与导出

**优势**：
- 单一 Runtime，无需切换 Notebook
- Drive 持久化 artifacts、checkpoints、metrics、results
- PostgreSQL 生命周期内保持
- PyG 只安装一次
- 幂等安装策略

#### Legacy: Component/Reference Notebooks

原 00～05 保留为模块化参考/开发说明：

```
notebooks/00_colab_environment.ipynb
notebooks/01_preprocess_theia.ipynb
notebooks/02_baseline_smoke_test.ipynb
notebooks/03_train_main_models.ipynb
notebooks/04_run_ablations.ipynb
notebooks/05_collect_results.ipynb
```

**注意**：这些 Notebook 各自可能连接新的 Colab Runtime，不推荐作为正式实验的主要运行方式。


### Weights & Biases interface

W&B is used as the default interface to visualize and historize experiments. First log into your account from the CLI using:

```shell
wandb login
```

Set your API key, which can be found on the website. Then you can push the logs and results of experiments to the interface using the `--wandb` arg.
The preferred solution is to run the `run.sh` script, which directly logs the experiments to the W&B interface.

```shell
python src/orthrus.py THEIA_E3 --wandb
```

## License

See [licence](LICENSE).
