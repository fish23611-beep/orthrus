# MAGIC 外部基线统一论文协议接入审计

审计日期：2026-09-03
ORTHRUS 审计分支：`audit/magic-unified-baseline`
ORTHRUS 起始提交：`9f8607d20e0cb3f3c37dbbf76b2479e12ffb4548`
MAGIC 审计提交：`aa0b647eea74b6faa0e52eb444370c4411a32cbe`

## 1. Executive Summary

结论：**CONDITIONAL GO**。MAGIC 可以作为独立 external baseline backend 接入，但官方评估脚本不能原样用于论文主表，也不能现在直接开始正式运行。

已确认的首要协议冲突是 test-label threshold selection：官方 entity-level 代码先在 `y_test` 与 test anomaly scores 上计算 precision-recall curve，再用数据集特定的目标 recall（THEIA 为 `0.99996`）选择 `best_thres`。这不是推测，而是源码事实，必须标记为 **BLOCKER / PROTOCOL INCOMPATIBILITY**。最小适配是保留 MAGIC 原始 KNN anomaly score，改由当前论文的 validation-only threshold 规则固定阈值，test 只应用阈值并计算最终指标。

另外四项在实施前必须闭环：官方只支持 E3-THEIA、没有 THEIA_E5 loader；官方从 train+test 一起确定 type one-hot 维度；entity-level inference 对完整 test graph 做 GAT 编码，不能直接满足当前因果信息边界；官方 seed 固定为 0 且 DGL/确定性控制不完整。这些问题可通过输入、协议、seed 与输出 adapter 解决，不需要改写 GAT、GMAE、mask/loss 或 KNN score 核心。

依据：`paper`、`README`、`source code`、`current project`；上述 GO 判断属于基于这些事实的 `inference`。

## 2. MAGIC 官方来源

| 项目 | 核验结果 | 依据 |
|---|---|---|
| 正式论文 | Zian Jia 等，*MAGIC: Detecting Advanced Persistent Threats via Masked Graph Representation Learning*，33rd USENIX Security Symposium，2024，pp. 5197–5214 | [USENIX 页面](https://www.usenix.org/conference/usenixsecurity24/presentation/jia-zian)、[正式 PDF](https://www.usenix.org/system/files/usenixsecurity24-jia-zian.pdf) |
| 论文预印本 | arXiv:2310.09831，首次提交于 2023-10-15 | [arXiv](https://arxiv.org/abs/2310.09831) |
| 官方仓库 | `FDUDSDE/MAGIC`；论文脚注给出该 URL，README 自称论文官方代码 | [GitHub](https://github.com/FDUDSDE/MAGIC)、paper、README |
| repository owner | GitHub organization `FDUDSDE` | repository |
| 审计 ref | `main`/HEAD：`aa0b647eea74b6faa0e52eb444370c4411a32cbe`，commit date 2024-10-24 | `git ls-remote`、local shallow clone |
| tag/release | 未发现 tag；GitHub 页面也未列 release。因而不存在可核验的 paper tag | `git ls-remote --tags`、repository |
| license | MIT License；文件版权行为 `Copyright (c) 2023 Jimmyokok` | [LICENSE](https://github.com/FDUDSDE/MAGIC/blob/aa0b647eea74b6faa0e52eb444370c4411a32cbe/LICENSE) |

本审计把上述 commit 视为“当前官方仓库快照”，不能声称它就是会议提交时的精确代码快照：该 commit 晚于 USENIX Security 2024 举办时间，且仓库没有 paper tag。依据：repository；这是边界说明，不是版本等同性推测。

## 3. Paper / repository / commit / license

论文、README 与仓库主题一致，论文脚注也直接链接该仓库，因此“官方仓库”身份已可靠核实。当前 HEAD 的提交说明为 `1.0.7 ... Bug fix and requirements.txt`，但 `1.0.7` 只是提交消息中的文本，不是可解析 tag；实施时必须锁定完整 commit SHA，并把源码 URL、commit、license 和 wrapper commit 同时写入 `environment.json`。依据：paper、README、repository/source；实施要求来自 `current project` 与 `inference`。

## 4. MAGIC 方法流程

官方方法链为：

```text
raw audit logs
  → directed provenance graph + noise reduction
  → node/edge label features
  → GAT masked graph autoencoder
  → benign-train node embeddings
  → KNN/KD-tree outlier detector
  → normalized node anomaly score
  → thresholded entity decision
```

图表示模块用多层 GAT；训练时随机 mask 节点特征，decoder 做 masked feature reconstruction，同时用正/负节点对做 sampled structure reconstruction。entity-level 检测把 benign training node embeddings 交给 `sklearn.neighbors.NearestNeighbors`，对 test node embedding 求 KNN mean distance。依据：[paper §4](https://www.usenix.org/system/files/usenixsecurity24-jia-zian.pdf)、[model/autoencoder.py](https://github.com/FDUDSDE/MAGIC/blob/aa0b647eea74b6faa0e52eb444370c4411a32cbe/model/autoencoder.py)、[model/eval.py](https://github.com/FDUDSDE/MAGIC/blob/aa0b647eea74b6faa0e52eb444370c4411a32cbe/model/eval.py)。

## 5. Dataset support

官方支持：StreamSpot、Unicorn Wget，以及 DARPA TC Engagement 3 的 Trace、THEIA、CADETS；entity-level 实现只接受 `trace/theia/cadets` 三个名字。官方 THEIA loader 硬编码 E3 文件名：训练 `.json` 到 `.json.3`，测试 `.json.8`。依据：README、`utils/trace_parser.py`。

| 本论文数据集 | 官方原生支持 | 审计判断 |
|---|---|---|
| `THEIA_E3` | 是，但官方 split/ground truth 与本项目不同 | 需要 canonical input/protocol adapter，不能直接复用官方结果 |
| `THEIA_E5` | 否 | 必须新增外围 adapter；不得假称官方支持 |

当前项目的 canonical split 为：

| Dataset | Train | Validation | Test | Unused |
|---|---|---|---|---|
| THEIA_E3 | `graph_2,3,4,5` | `graph_9` | `graph_10,12,13` | `graph_11` |
| THEIA_E5 | `graph_8,9,10` | `graph_11` | `graph_14,15` | `graph_12,13,16,17` |

依据：`current project` 的 `src/config.py::DATASET_DEFAULT_CONFIG`。

## 6. Input schema

官方原始输入是 DARPA CDM JSON audit logs。parser 提取 `src UUID`、`src type`、`dst UUID`、`dst type`、`event/edge type` 和 `timestampNanos`；对 READ/RECV/LOAD 反转因果方向，按 timestamp 排序，再构造 NetworkX `DiGraph` 与 DGL graph。节点和边分别带 `type`，没有把 timestamp 保留成 DGL 模型输入。依据：`utils/trace_parser.py`、`utils/loaddata.py`。

源码实际构造的是 directed simple graph：同一 `(src,dst)` 只保留首次加入的边。论文描述为去重同 label 多边并平均剩余不同 edge-label embedding；当前源码没有实现这个“多 label 平均”过程。该 paper/code 差异必须记录，后续 source-freeze smoke test 应以锁定源码行为为主，不应擅自“修好”第三方算法。依据：paper §4.1、source code；最后一句是 `inference`。

当前 canonical artifacts 已提供 adapter 所需的最小字段：`src`、`dst`、`t`、`src_type`、`dst_type`、`edge_type_index`、`split`、`global_event_index`。不需要读取或重用 ORTHRUS Word2Vec 向量。依据：`current project` 的 `src/data_utils.py`。

## 7. Feature representation

- MAGIC 不使用 Word2Vec。`README`、requirements 与模型代码均没有 Word2Vec 路径。依据：README/source code。
- 论文称 node/edge labels 通过 lookup embedding 映射；当前代码先生成 type one-hot，再由 GAT 的线性层学习表示。依据：paper、`utils/loaddata.py::transform_graph`、`model/gat.py`。
- 模型输入使用 node type 与 edge/event type。依据：source code。
- parser 会读取 file path、process name、remote address，但这些只用于名称 sidecar/ground-truth 展示，不进入 GMAE 输入。依据：`utils/trace_parser.py`。
- timestamp 只用于排序；不作为 node/edge feature，也没有时间编码。依据：source code。
- entity UUID 只用于建图/映射，不是可学习的 per-entity embedding。依据：source code。

**Transductive 风险（confirmed）**：论文明确称 label lookup 为 transductive；源码在 `preload_entity_level_dataset` 中同时扫描 train/test graphs 才确定 `node_feature_dim` 与 `edge_feature_dim`。这会让 test 出现的 type category 影响输入维度。实际 THEIA 是否出现 train 未见的新 type 尚未用数据验证，但“维度由 test 参与确定”本身已经确认。最小适配是从预先固定的 canonical CDM schema/训练 split 冻结映射，对未知类别 fail-fast 或映射到预先声明的 UNK；禁止从 val/test 扩展词表。依据：paper/source code；影响大小尚未验证；方案为 `inference`。

## 8. Training objective

MAGIC 是 self-supervised/unsupervised anomaly detector，不需要恶意训练样本：

- mask rate：0.5；随机选择 masked nodes；
- feature loss：scaled cosine error，`gamma/alpha=3`；
- structure loss：existing non-masked edges 为正样本，global uniform negative sampling 为负样本，两层 MLP + BCE；
- total loss：feature reconstruction + structure reconstruction；
- entity-level GAT：3 layers、4 heads、hidden/output dimension 64；
- optimizer：Adam，learning rate 0.001，weight decay `5e-4`；
- entity-level epochs：50。

官方 entity-level 代码逐 training graph 优化，最后保存一次模型。依据：paper、`utils/config.py`、`model/autoencoder.py`、`train.py`。

必须保持的核心是 GAT/GMAE architecture、masking、两项 reconstruction objective、published hyperparameters 和 KNN anomaly-score 定义。不能为“统一”而替换成 ORTHRUS encoder/loss/Word2Vec/MSTC calibration/Top-k。依据：paper/source code；集成边界为 `inference`。

## 9. Validation protocol

官方 entity-level pipeline 没有 validation split、validation loader 或 early stopping。论文采用 earliest 80% train / remaining 20% test；代码对 THEIA 使用硬编码 E3 train/test files。依据：paper、source code；全文搜索未发现 validation 路径。

统一协议必须使用当前 `THEIA_E3/E5` validation days，但仅用于：

1. 在冻结 checkpoint 与 train-only KNN reference 下生成 validation raw node scores；
2. 选择 validation-only threshold；
3. 记录 diagnostics。

validation 不进入 GMAE 参数训练、不进入 KNN reference、不扩展 type vocabulary；test 不参与上述任何动作。依据：`current project` 协议；方案为 `inference`。

## 10. Checkpoint selection

官方 entity-level 实现不保存多个 epoch checkpoint，不做 best-checkpoint selection；完成固定 50 epochs 后只保存 `checkpoint-<dataset>.pt`。因此未发现 test-based checkpoint selection，也未发现 validation-based selection。依据：`train.py`。

最小、最忠实且合规的方案是预先冻结 published 50 epochs 并使用 final epoch，不把 test 或 validation metric 用来选 epoch。若以后新增 validation-loss early stopping，必须作为偏离官方方法的独立 sensitivity，而非默认 MAGIC 主表。依据：source code；方案为 `inference`。

## 11. Anomaly score

entity-level detection：

```text
train_reference = all benign train node embeddings
mean_train_distance = mean(KNN distance on a seeded subset of at most 50,000 train embeddings)
raw_score(node) = mean(KNN distance from test node to train_reference) / mean_train_distance
k = 10  # except CADETS=200; THEIA uses 10
```

这是直接 node-level continuous score。`distance_save_<dataset>.pkl` 缓存 `mean_train_distance` 与 test distances，但没有 artifact fingerprint，不能在统一 runner 中跨 config/seed 盲目复用。应在每个 canonical run 下保存并校验 source/checkpoint/config identity。依据：`model/eval.py`；缓存风险与方案为 `inference`。

## 12. Threshold / decision protocol

### 12.1 官方论文

论文称 threshold `theta` 按 dataset 做 simple linear search；Appendix D 同时以 FPR 与 recall 描述可选范围，并建议在只有 benign data 时可按目标 FPR 选阈值。论文没有定义独立 validation set。依据：paper §5、§6.5、Appendix D。

### 12.2 官方代码

官方代码执行：

```python
prec, rec, threshold = precision_recall_curve(y_test, score)
# THEIA: 找到 rec < 0.99996 的位置
best_thres = threshold[best_idx]
```

随后在相同 `y_test/score` 上计算 TP/FP/TN/FN。这是 **confirmed test-label threshold selection**，不是无监督 percentile，也不是只用 training score。依据：[model/eval.py#L164-L234](https://github.com/FDUDSDE/MAGIC/blob/aa0b647eea74b6faa0e52eb444370c4411a32cbe/model/eval.py#L164-L234)。

### 12.3 统一论文方案

**BLOCKER / PROTOCOL INCOMPATIBILITY**：官方 decision protocol 不得原样进入本论文主表。

推荐最小适配：保留 `raw_score`；在正常 validation node scores 上使用当前论文预注册的 `validation_quantile(q=0.999)` 计算一次 threshold；用统一 threshold helper 固定比较规则；test 仅应用冻结 threshold。AUROC/AUPRC 始终从 raw score 计算，Recall/MCC/FPR 从冻结决策计算。若采用“validation target FPR”而不是 quantile，必须先由用户明确修改统一协议，不能在实施时临场选择。依据：source code、`current project` 的 `src/mstc/thresholding.py`；方案为 `inference`。

## 13. Ground-truth definition

官方 README 要求下载 ThreaTrace `.txt` labels；论文主结果称使用相同 ThreaTrace ground truth。官方 parser 读取恶意 entity UUID 列表并映射到 test graph local index，排除 `MemoryObject` 与 `UnnamedPipeObject`；训练图构造还会借助该恶意集合过滤部分 incident edges。依据：README、paper、`utils/trace_parser.py`。

论文 Appendix G 另给一种人工标注方法：从关键攻击实体名称出发匹配，并探索邻域补充 attack-involved entities。这属于 neighborhood-expanded methodology；不能把它等同于当前项目的严格 node-level CSV。依据：paper Appendix G。

当前项目 ground truth 是 `src/config.py` 指定的 attack-specific CSV，通过精确 UUID→global node ID 映射得到 set；E3 当前纳入 Browser Extension 与 Firefox Backdoor，E5 为单个已列攻击文件。正式比较必须对四模型使用这个相同 set。不得使用 MAGIC/ThreaTrace labels 计算 MAGIC 主表，也不得用未来 test labels 清理 training graph；若 canonical training split 被发现含攻击污染，应 fail 并单独处理数据协议，而不是静默过滤。依据：`current project`；最后两项为统一协议要求与 `inference`。

## 14. Detection granularity

MAGIC 同时支持 batched-log-level graph/state score 和 entity-level node score。DARPA E3 路径输出每个系统实体的 KNN anomaly score，因此与当前 node-level 主表在粒度上可对接，不需要把 edge score 伪装成 node score。依据：paper、`eval.py`、`model/eval.py`。

官方代码使用 graph-local contiguous node IDs；canonical output 要求 global node IDs。输入 adapter 必须保存 `local_node_id ↔ canonical_node_id` sidecar，并定义跨 causal snapshot 的重复 node 归并规则。该规则必须在 smoke 前预注册、与 label 无关，并同时用于 validation/test。依据：source/current project；归并需求为 `inference`。

## 15. Metrics

官方实现直接提供/打印 AUROC、Precision、Recall、F1、FPR、TP/FP/TN/FN；没有 AUPRC 与 MCC，也没有 canonical JSON/CSV。依据：`model/eval.py`。

当前统一 evaluator 能从 `node_id + raw_score + frozen y_hat + y_true` 重新计算 Recall、MCC、AUPRC、AUROC、FPR 等。优先链路是“保留 MAGIC raw score → output adapter → current canonical metric function”，不改 MAGIC loss 或 score。依据：`current project` 的 `src/mstc/metrics.py`、`src/mstc/evaluation_runner.py`；方案为 `inference`。

## 16. Random seeds

官方 `set_random_seed` 设置 Python `random`、NumPy、Torch CPU/CUDA seed；mask 由 `torch.randperm`，正边采样由 Python `random.sample`，negative sampling 由 DGL，batch-level split/shuffle 也随机。依据：source code。

已确认缺口：

- `train.py`/`eval.py` 都硬编码 `set_random_seed(0)`，无 CLI seed；
- 未调用 `dgl.seed` / `dgl.random.seed`；
- `torch.backends.cudnn.determinstic` 拼写错误，实际没有设置 `deterministic`；
- 没有设置 `PYTHONHASHSEED`、DataLoader generator/worker seed 或 deterministic algorithms；
- GPU/DGL 算子仍可能非确定。

后续 adapter 必须接收 seeds `0,1,2`，设置并记录 Python/NumPy/Torch CPU/CUDA/DGL、DataLoader 与 deterministic flags；无法保证 bitwise determinism 时要在 `environment.json` 明示。不得改成官方的“100 个 evaluation shuffle seed”报告方式。依据：source code/current protocol；控制方案为 `inference`。

## 17. Dependency environment

| 组件 | MAGIC 官方 | 当前 `/home` | 判断 |
|---|---:|---:|---|
| Python | 3.8 | 3.9.25 | 不同 |
| PyTorch | 1.12.1+cu116 | 1.13.1+cu117 | ABI/CUDA 组合不同 |
| DGL | 1.0.0 | 未安装 | 缺失 |
| Scikit-learn | 1.2.2 | 1.2.0 | 不同 |
| PyG | 未要求 | 2.5.3 | ORTHRUS 核心依赖，不应受影响 |
| NetworkX | 代码使用但 requirements 未固定 | 2.8.7 | 官方环境定义不完整 |

官方 requirements 还依赖 `torchvision 0.13.1+cu116`、`torchaudio 0.12.1`、`xxhash`，但漏列运行时使用的 NumPy、NetworkX、tqdm。依据：[requirements.txt](https://github.com/FDUDSDE/MAGIC/blob/aa0b647eea74b6faa0e52eb444370c4411a32cbe/requirements.txt)、source code、current environment。

方案比较：

| 方案 | 风险 | 结论 |
|---|---|---|
| A. same environment | 需安装 DGL，并可能迫使 Torch/CUDA/sklearn 变更，容易破坏已验收 ORTHRUS/PyG | 不推荐 |
| B. separate conda/venv | Python 包隔离可行，但仍共享 host CUDA/系统库，DGL 旧 wheel 兼容性需验证 | 备选 |
| C. separate container | 可锁 Python/Torch/DGL/CUDA，和 ORTHRUS 只交换 canonical files | **推荐** |

本轮未安装任何依赖。依据：实际执行记录；推荐结论为 `inference`。

## 18. Resource considerations

论文在 E3-Trace 上给出 phase-level 数字，并报告 graph representation training 约 151s（GPU）/685s（CPU）、inference representation 约 5s（GPU）/10s（CPU）；KNN detection inference 报告 825s，峰值内存因 phase 而异，表中最高列值约 2610MB。论文还指出 KNN 检查 684,111 targets 用时约 13.8 分钟。依据：paper Table 6 与 §7。

这些数字来自 E3-Trace，不是当前 THEIA_E3/E5、当前 split、当前硬件或 adapter，因此不能据此承诺本项目成本。可靠结论只有复杂度：整体空间随 `(N+E)*(t+d)` 线性增长，KNN reference 需要 `O(N*d)`；CPU KNN 可能成为主要 inference cost。当前 THEIA_E3/E5 的 GPU、CPU RAM、disk、preprocess/train/test 实际预算**无法从现有来源可靠确认**，M8/M9 必须先测 smoke/pilot。依据：paper；边界与建议为 `inference`。

## 19. Potential leakage risks

| 风险 | 状态 | 证据与处置 |
|---|---|---|
| test-label threshold selection | **confirmed / BLOCKER** | `precision_recall_curve(y_test, score)` + THEIA recall target；改为 validation-only threshold |
| test-based checkpoint selection | **not found** | 固定 50 epochs、只保存 final checkpoint；保持预固定 epoch |
| test-derived type dimension | **confirmed** | train/test 都参与 feature dimension 扫描；改为 frozen canonical/train schema |
| whole-test-graph future topology | **confirmed protocol mismatch** | 完整 test DGL graph 一次性 embed，timestamp 不进模型；需 causal snapshot/input-boundary adapter |
| malicious test labels 清理 train graph | **confirmed source behavior / incompatible** | parser 用 malicious UUID set 过滤 train edges；canonical train 不允许读取 test GT |
| stale cached test distances | **confirmed risk** | cache 无 config/checkpoint fingerprint；改为 per-run identity 与校验 |
| validation leakage into training/KNN | not found in official code（因为没有 val） | 新 adapter 必须禁止 val 进入 GMAE/KNN reference |
| Word2Vec leakage | not applicable to MAGIC | MAGIC 不使用 Word2Vec |
| optional model adaptation on test | paper 有机制，主代码未在当前 eval 路径启用 | 正式比较禁用；避免 analyst feedback/test labels 进入模型 |

“whole-test-graph”是否在原论文的 offline batch setting 中被称作 leakage不是本审计要替作者下定义；但它与当前项目要求的时间因果信息边界冲突，这一点已经确认。依据：paper/source/current protocol；措辞区分事实与评价。

## 20. Differences from our unified paper protocol

| 维度 | MAGIC 官方 | 当前统一协议 | 需要动作 |
|---|---|---|---|
| dataset | E3-THEIA | THEIA_E3 + THEIA_E5 | E5 adapter |
| split | earliest 80%/hardcoded E3 files，无 val | 显式 train/val/test days | 使用 canonical split |
| train labels | 使用恶意列表辅助过滤 train graph | train 阶段不能读取 test labels | 移除 label-driven filter |
| feature vocabulary | train+test 确定维度 | train-independent schema 或 train-only fit | freeze mapping |
| inference context | whole test graph | 因果信息边界 | causal snapshot policy |
| checkpoint | fixed final epoch | 禁止 test selection | 保留 fixed final epoch |
| threshold | test labels/target recall | validation-only | 替换 decision adapter |
| ground truth | ThreaTrace/附录方法 | 当前 strict node CSV | 统一 evaluator 使用当前 GT |
| metrics | 无 MCC/AUPRC artifact | Recall/MCC/AUPRC/AUROC/FPR | output adapter 重算 |
| seeds | hardcoded 0/评估 100 shuffle | 0,1,2 独立训练 run | seed adapter |
| artifacts | repo-relative pickle/pt/stdout | canonical per config/dataset/model/seed | output/runtime adapter |

依据：paper/source/current project。

## 21. Required adapters

推荐数据流：

```text
canonical THEIA artifacts
  → MAGIC input adapter
      - canonical split
      - global↔local node map
      - frozen type schema
      - causal graph snapshots
  → locked original MAGIC GMAE + KNN raw score backend
  → MAGIC raw output archive
  → unified protocol/output adapter
      - validation-only threshold
      - canonical ground truth attached only after prediction fixed
      - canonical metrics
  → node_scores/node_predictions.csv + metrics.json + runtime.json
```

建议新增 `src/baselines/magic/`，包含 source manifest、input adapter、backend wrapper、protocol/seed control、raw/output adapter 与独立 tests。`run_matrix.py` 保持 scheduler；只在通用 experiment dispatch 层按明确 backend identity 调用 MAGIC runner。不要把 MAGIC 注册到 `src/factory.py` 的 MSTC model factory。依据：current project；架构为 `inference`。

当前仓库已有 `config.py` 中关于 `magic` 的注释字段和一个 `compute_tw_labels_for_magic` helper，但没有 official GMAE/KNN model、正式 config、runner 或 artifact output；这些不能被当成“MAGIC 已实现”。依据：current project。

## 22. What must remain original MAGIC behavior

- directed provenance graph semantics 与官方明确的 causal edge direction；
- node/edge type representation，而非 ORTHRUS Word2Vec；
- GAT encoder/decoder architecture；
- node masking；
- scaled-cosine feature reconstruction；
- sampled structure reconstruction + BCE；
- published entity-level dimensions/layers/mask rate/optimizer/epochs/k；
- train-node-embedding KNN reference；
- normalized mean KNN-distance raw anomaly score；
- entity-level detection 含义。

依据：paper/source code。

## 23. What must be standardized

| 统一项 | 固定规则 |
|---|---|
| dataset | THEIA_E3/THEIA_E5 canonical artifacts |
| temporal split | 当前项目 train/val/test/unused 边界 |
| information boundary | train-only fit；validation-only selection；test evaluation only；无未来拓扑输入 |
| ground truth | 当前项目 strict node-level CSV |
| threshold | validation-only，推荐预注册 `q=0.999` |
| final metrics | Recall/MCC/AUPRC/AUROC/FPR 与 canonical confusion counts |
| seeds | 0,1,2；每 seed 独立模型/KNN/threshold/artifacts |
| output schema | canonical `config_resolved.yml/environment.json/runtime.json/node_scores/*` |
| source identity | MAGIC URL + full commit + wrapper commit + environment digest |

threshold 被列在标准化项，是源码审计后的结论，不是审计前预设。依据：source/current protocol；规则选择为 `inference`。

## 24. Blockers

实施正式运行前必须解决：

1. **B1—threshold**：禁止调用官方 test-label threshold 路径，增加 validation-only decision adapter 与防回归测试。
2. **B2—temporal/transductive boundary**：书面冻结 causal snapshot、node 重复归并和 type-vocabulary 规则；测试未来事件不改变既有 score。
3. **B3—THEIA_E5**：证明 canonical E5 input adapter 字段、type mapping 与 graph construction contract 可用。
4. **B4—ground truth**：验证预测固定前不读取 current test GT；只在统一 evaluator 附加标签。
5. **B5—environment**：建立隔离 container/lock，不能升级当前已验收环境。
6. **B6—seed**：补齐 DGL、DataLoader、Torch deterministic 控制并记录残余非确定性。

另外需要用户确认的协议选择：默认采用 `validation_quantile(q=0.999)`，以及 causal snapshot 下重复 node 的无标签归并规则。未确认前可以写单元测试/adapter 草案，但不应跑 pilot 或 formal runs。依据：综合 `inference`。

## 25. Final conclusion

**CONDITIONAL GO**：

- 对“把 MAGIC 作为 external baseline 开始受控实现”：在用户确认 threshold 与 causal snapshot 规则后可 GO；
- 对“直接运行 THEIA_E3/E5 正式 3-seed 主实验”：当前 NO-GO；
- 对“原样使用官方 `eval.py` 指标进入论文”：永久 NO-GO，因为已确认 test-label threshold selection 与 ground-truth/split 不一致；
- 对“保留 original MAGIC model/score，仅做外围标准化”：可行，且是最小侵入方案。

本结论没有宣称 MAGIC 已实现、已运行或已有当前协议结果。本轮未运行训练、未下载 DARPA 数据、未安装依赖、未修改 MSTC。
