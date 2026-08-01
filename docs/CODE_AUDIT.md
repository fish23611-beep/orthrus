# ORTHRUS 代码审计 (CODE_AUDIT)

> 仅代码审计与文档化。本轮不修改任何 Python/YAML/Notebook/测试代码。
> 本审计只覆盖当前 `mstc-pids` 分支所需的最小改动面。所有发现以代码现状为准。
> 用户引用的 `docs/ORTHRUS_MSTC_IMPLEMENTATION_SPEC.md` 不存在；任务说明以仓库内的 `docs/ORTHRUS-MSTC-PIDS_完整改造任务说明.md` 为准（功能目标一致）。

## 1. 仓库当前执行链概览

### 1.1 顶层入口

- `src/orthrus.py` 是顶层流水线入口，按以下顺序串接：
  1. 图构建：`graph_construction/build_orthrus_graphs.py::main()` — 从 PostgreSQL 拉取节点 / 边，按天聚合到时间窗口，存为 `networkx.MultiDiGraph`。
  2. 节点特征化（Word2Vec 节点语义）：`edge_featurization/build_feature_word2vec.py::main()` — 从 PostgreSQL 读 `indexid2msg`，训练 Word2Vec。
  3. 边特征化：`edge_featurization/embed_edges_feature_word2vec.py::main()` — 读 Word2Vec 模型，按 [src_type | src_emb | edge_type_oh | dst_type | dst_emb] 拼接 `msg`，输出 `TemporalData.simple` 文件。
  4. 训练：`detection/orthrus_gnn_training.py::main()`。
  5. 测试：`detection/orthrus_gnn_testing.py::main()`。
  6. 评估：`detection/evaluation.py::main()` → `detection/node_evaluation.py::main()`。
  7. 攻击重建：`attack_reconstruction/tracing.py::main()`。

### 1.2 配置驱动

- `src/config.py`：
  - 提供 `DATASET_DEFAULT_CONFIG`（THEIA_E3/E5、CADETS_E3/E5、CLEARSCOPE_E3/E5），含 `train_files / val_files / test_files` 等关键字段。
  - 提供 `DATABASE_DEFAULT_CONFIG`：`host='postgres'`, `user='postgres'`, `password='postgres'`, `port='5432'`。
  - 通过 `yacs.config.CfgNode` 构造层级 cfg，由 `set_task_paths` 计算各阶段缓存路径。
  - 任务依赖：`build_graphs -> embed_nodes -> embed_edges -> gnn_training -> gnn_testing -> evaluation -> tracing`。
- `config/orthrus.yml` 是当前唯一模型配置；`model: orthrus` 硬编码。

### 1.3 关键运行时入口（CLI）

- `get_runtime_required_args`（`src/config.py`）支持的 flag：
  - `dataset`（位置参数，必填）
  - `--model`、`--wandb`、`--exp`、`--tags`
  - `--cpu`、`--run_from_training`、`--from_weights`、`--seed`
  - `--show_attack`、`--gt_type`、`--plot_gt`
  - 由 `add_cfg_args_to_parser` 自动暴露所有 cfg 字段。

---

## 2. 数据结构与每张 Tensor 的形状

### 2.1 图构建产物

- `build_orthrus_graphs.py::gen_edge_fused_tw` 产出 `nx.MultiDiGraph`：
  - 节点属性：`node_type` ∈ {subject, file, netflow}，`label`（hash 化或明文）。
  - 边属性：`event_uuid`, `time`（int64 纳秒时间戳）, `label`（rel2id 字符串）。
- 落盘位置：`cfg.graph_construction.build_graphs._graphs_dir/graph_{day}/{time_interval}`。
- `_test_mode=True` 时只生成第一个图、至多 2000 条边。

### 2.2 节点特征（Word2Vec 输出）

- 由 `embed_edges_feature_word2vec.py::get_indexid2vec` 生成 `indexid2vec: Dict[int, np.ndarray]`：
  - 形状 `[emb_dim]`（默认 128，来自 `cfg.edge_featurization.embed_nodes.emb_dim`）。
  - 加权平均 + L2 归一化：`v / ||v||`。
- 训练语料：`build_feature_word2vec.py::load_corpus_from_database` 从 PostgreSQL 的 `subject_node_table / file_node_table / netflow_node_table` 抽取，**整库语料**，不分 train/val/test。
- Word2Vec 模型存于 `feature_word2vec._model_dir/feature_word2vec.model`。

### 2.3 `TemporalData`（边特征化产物）

`embed_edges_feature_word2vec.py::gen_vectorized_graphs` 构造：

| 字段 | dtype | 形状 | 含义 |
|------|-------|------|------|
| `src` | long | `[E]` | 源节点 index_id |
| `dst` | long | `[E]` | 目的节点 index_id |
| `t` | long | `[E]` | 纳秒时间戳 |
| `msg` | float | `[E, 2*emb_dim + 2*node_type_dim + edge_type_dim]` | 拼接特征（详见下） |

`msg` 列布局（固定顺序）：
```
[src_type_onehot | src_emb | edge_type_onehot | dst_type_onehot | dst_emb]
   node_type_dim    emb_dim   edge_type_dim      node_type_dim      emb_dim
```
THEIA 默认：`[E, 2*128 + 2*3 + 10] = [E, 290]`。
按 `used_method == "only_type"` 走 `extract_msg_node_type_only`，此时 `msg` 不含 `emb` 段：
```
[src_type_onehot | edge_type_onehot | dst_type_onehot]
   node_type_dim   edge_type_dim       node_type_dim
```
即 `[E, 3 + 10 + 3] = [E, 16]`。

### 2.4 `data_utils.extract_msg_from_data / extract_msg_node_type_only`

为每个时间窗口 `g` 进一步注入：
- `g.x_src`: `[E, emb_dim]` 或 `[E, node_type_dim]` — 来自 `src_emb` 或 `src_type`。
- `g.x_dst`: `[E, emb_dim]` 或 `[E, node_type_dim]`。
- `g.msg`: `[E, ?]` — 若启用 `predict_edge_type`，剔除 `edge_type` 段；否则保留。
- `g.edge_type`: `[E, edge_type_dim]` one-hot。
- `g.edge_feats`: `[E, edge_type_dim]` 或 `[E, msg_dim]` 或 `None`，由 `cfg.encoder.edge_features` 控制。
- `g.edge_index`: `[2, E]` 由 `torch.stack([src, dst])` 重建。
- **注意：`g` 上**不显式保存 `src_type_index / dst_type_index / edge_type_index / event_index` 字段；类型只能通过 `g.x_src / x_dst` 的尾部 one-hot 反推（与任务说明 §7「不要隐式切片推断类型」相冲突，见 §13）。

### 2.5 训练侧 `OrthrusEncoder.forward` 的内部表示

- `n_id`: 批内 + 历史邻居并集，long `[N']`。
- `edge_index`: `[2, E_h]`（`E_h` 为历史邻居构造的边数）。
- `x_proj = src_linear(x_src[n_id]) + dst_linear(x_dst[n_id])`: `[N', temporal_dim]`（默认 temporal_dim=100）。
- `h = GraphTransformer(x_proj, edge_index, edge_feats)`: `[N', node_out_dim]`（默认 64）。
- `h_src = h[assoc[src]]`, `h_dst = h[assoc[dst]]`: `[B, node_out_dim]`，其中 `B = edge_index[0].numel()`（当前批边数）。

### 2.6 `LastNeighborLoader` 的内存布局

- `self.neighbors`: `[num_nodes, size]` long。
- `self.e_id`: `[num_nodes, size]` long。
- `self._assoc`: `[num_nodes]` long。
- `self.cur_e_id`: 标量 int。
- `reset_state()` 把 `e_id` 全部填 `-1`，`cur_e_id=0`。

### 2.7 解码器输出

- `EdgeTypeDecoder.forward` 在 inference 时返回 `[B]` 长度的 per-edge loss（即 `reduction='none'` 的 cross-entropy）；训练时返回 scalar。
- 测试侧：`each_edge_loss[i]` 即单条边交叉熵，作为 edge-level 异常分。
- 在 `Orthrus.forward` 中 `loss_or_scores` 与 decoder 数量同长度，所有 decoder 的 loss 相加。

---

## 3. 当前训练/验证/测试划分

### 3.1 数据集划分来源

- 全部数据集的 `train_files / val_files / test_files / unused_files` 都在 `src/config.py::DATASET_DEFAULT_CONFIG` 中写死：
  - **THEIA_E3**：`train=[graph_2..5]`, `val=[graph_9]`, `test=[graph_10, graph_12, graph_13]`, `unused=[graph_11]`。
  - **THEIA_E5**：`train=[graph_8..10]`, `val=[graph_11]`, `test=[graph_14, graph_15]`。
  - **CADETS_E3/E5 / CLEARSCOPE_E3/E5** 类似（不在本期范围）。
- 文件名由 `get_all_files_from_folders` 按数字部分升序排序。

### 3.2 关键隐患

- **THEIA_E5 训练只覆盖 graph_8/9/10 三天，graph_11（val）与 graph_14/15（test）之间存在 graph_12/13（unused），跨日期跳跃较大**；这会影响任务说明中要求的「时间分位数仅由正常训练集生成」的现实分布。
- **train/val/test 不允许重叠**：通过文件集合划分保证，但 `_test_mode=True` 时 `load_data_set` 强制把 split 改为 train（见 `data_utils.py` L39-41）。

---

## 4. 当前历史邻居维护方式

### 4.1 数据结构

- 单一组件：`src/temporal.py::LastNeighborLoader`。
  - 每个节点保留最近 `size`（默认 20）条邻居（无向化：`neighbors = cat([src, dst])`, `nodes = cat([dst, src])`）。
  - 维护 `e_id`（全局事件 id）。
  - `insert(src, dst)` 在每次 batch 末尾调用（`OrthrusEncoder.forward` 第 83 行），因此**当前批次事件在下一次 batch 才可见**——这就是 `LastNeighborLoader` 的因果性。

### 4.2 查询时序（`OrthrusEncoder.forward`）

```
1. n_id = unique(cat([src, dst]))               # 当前批涉及节点
2. n_id, edge_index, e_id = neighbor_loader(n_id) # 取出这批节点的历史邻居 + 自身
3. assoc[n_id] = arange(N')
4. x_proj = src_linear(x_src[n_id]) + dst_linear(x_dst[n_id])  # 用当前事件 x
5. h = GraphTransformer(x_proj, edge_index, edge_feats)
6. h_src = h[assoc[src]], h_dst = h[assoc[dst]]
7. neighbor_loader.insert(src, dst)             # 当前事件下一轮才可见
```

### 4.3 风险点

- 邻居**只有 1 跳**，没有多尺度。
- `edge_feats` 直接取自 `full_data.edge_type[e_id]` / `full_data.msg[e_id]`，**使用历史事件真实的边类型 one-hot 作为输入特征**——这正是任务说明 §2.2 明确禁止的「把当前事件真实边类型作为当前事件输入特征」。当前实现里 `e_id` 是**历史事件**，不是当前事件，所以并非「当前事件真实边类型作为当前事件输入」，但需在多尺度采样阶段避免给当前事件注入自己的真实 edge_type / 真实时间桶。
- `reset_state()` 只在 `orthrus_gnn_training.py` 每个 epoch 开头调用一次（`OrthrusEncoder` 分支）。**但 `orthrus_gnn_testing.py` 的 `main` 没有显式调用 `reset_state()`**——见 §10。

---

## 5. 当前边异常分数如何产生

### 5.1 流程

1. `model(batch, full_data, inference=True)` →
   - `h = OrthrusEncoder(...)` 输出 `[B, node_out_dim]`。
   - 对每条当前边 `[i]`：
     - `loss_i = EdgeTypeDecoder(h_src[i], h_dst[i]).cross_entropy(logits, target_edge_type_class)`。
     - `reduction="none"` → `[B]` 长度。
2. `loss_or_scores = loss`（scalar 或 `[B]`），由 `inference` 决定。
3. `test()` 在 `detection/orthrus_gnn_testing.py`：
   - 取 `each_edge_loss[i]`，作为该事件的异常分。
   - 同步记录 `srcnode, dstnode, srcmsg, dstmsg, edge_type, time`。
   - 写 CSV：`cfg.detection.gnn_testing._edge_losses_dir/<split>/<model_epoch>/<time_interval>.csv`。
4. CSV 列：`loss, srcnode, dstnode, srcmsg, dstmsg, edge_type, time`（不含时间桶、门控权重、原始分数 vs 校准分数，与任务说明 §9.4 / §18 差距较大）。

### 5.2 边异常分的本质

- 即 **edge-type 多分类的 per-example cross-entropy**，模型预测边类型的负对数似然。值越大 → 越不像训练集中见过的边类型。
- 没有时间桶预测、没有原始 loss 与校准 loss 分离、没有 per-event 原始分数 vs 校准分数。

---

## 6. 当前节点分数如何聚合

### 6.1 节点收集（`node_evaluation.py::get_node_predictions`）

- 遍历每个时间窗口 CSV：
  - `node_to_losses[srcnode].append(loss)`。
  - 若 `cfg.detection.evaluation.node_evaluation.use_dst_node_loss=True`，`node_to_losses[dstnode].append(loss)`。
  - 同步追踪 `node_to_max_loss` 和 `node_to_max_loss_tw`。
- 每节点 `score = reduce_losses_to_score(losses, threshold_method)`：
  - `mean_val_loss` → `np.mean`。
  - `max_val_loss` → `np.max`。
- `y_hat = int(score > thr)` 或由 `compute_kmeans_labels` 给出。

### 6.2 关键事实

- **节点分数完全等价于该节点所有相关边事件 edge-type loss 的聚合**，没有 top-k、没有时间分桶、没有 calibrated score。
- 没有显式 `NodeScoreAggregator` 类、没有 `include_dst` 之外的源/目的权重区分、没有针对 src/dst 分别聚合。

---

## 7. 当前阈值和 K-means 如何工作

### 7.1 阈值来源（`evaluation_utils.py::get_threshold`）

- 仅支持两种方法：
  - `max_val_loss` → `max(validation_losses)`。
  - `mean_val_loss` → `mean(validation_losses)`。
- 阈值来自**正常验证集**所有事件的 edge-loss 列表（在 `calculate_threshold` 中聚合 `cfg.detection.gnn_testing._edge_losses_dir/val/<epoch>/*.csv` 的 `loss` 列）。
- 注意：这里有一个**当前实现的 bug 风险**——`get_threshold(val_tw_path, ...)` 接收 `val_tw_path` 但 `node_evaluation.py::get_node_predictions` 实际传入的是 `val_tw_path`（OK）；但 `evaluation.py::standard_evaluation` 调用 `evaluation_fn(val_tw_path, test_tw_path, ...)` 时同时把模型 epoch 路径和测试路径都传了进去，签名耦合较紧。

### 7.2 K-means（`evaluation_utils.py::compute_kmeans_labels`）

- `use_kmeans=True` 时：
  1. 所有节点按 score 升序。
  2. 取尾部 `topk_K` 个（默认 20）。
  3. `KMeans(n_clusters=2)` 在这一段子集上聚类。
  4. 取中心值最大的 cluster → 这些节点 `y_hat=1`。
- **风险点**：
  - 只看尾部 K 个节点做 2 簇聚类，分数彼此接近时易把正常节点误标为恶意。
  - 整个验证集的负样本没有用于校准——验证集「最大/均值 loss」作为阈值的实现缺乏正负样本分离。

### 7.3 文档化的未实现阈值

- `evaluation.py::standard_evaluation` 内还存有 `calculate_supervised_best_threshold`（基于 ROC 的最小 FPR 选取 TPR≥0.16 处的阈值）但**未在主流程使用**——这是历史遗留，不要轻易删除。
- 任务说明要求的 `validation_quantile` / `kmeans` / `max_validation` 共存三方案还没做（§12.1）。

---

## 8. PostgreSQL 依赖发生在哪些文件

| 文件 | 用途 |
|------|------|
| `src/config.py` | `DATABASE_DEFAULT_CONFIG` 与 `init_database_connection` 调用入口 |
| `src/provnet_utils.py` | `init_database_connection`, `gen_nodeid2msg`, `get_indexid2msg`, `get_node_to_path_and_type`（均 `psycopg2.connect`） |
| `src/graph_construction/build_orthrus_graphs.py` | `get_node_list`（读 `subject/file/netflow_node_table`），`gen_edge_fused_tw`（读 `event_table`） |
| `src/edge_featurization/build_feature_word2vec.py` | `load_corpus_from_database`（读三张 node_table） |
| `src/edge_featurization/embed_edges_feature_word2vec.py` | `get_indexid2msg`（再次读三张 node_table） |
| `src/labelling.py` | `get_ground_truth`（读三张 node_table 解析 uuid → nid），`get_GP_of_each_attack`，`get_t2malicious_node`（读 `event_table`） |
| `src/detection/orthrus_gnn_testing.py` | `init_database_connection`, `gen_nodeid2msg`（把 nid → string msg） |
| `src/detection/evaluation_utils.py` | `get_node_to_path_and_type`（再次读三张 node_table，但已用缓存 `node_to_paths.pkl`，所以有数据库回退的余地） |

### 关键事实

- **没有真正的「PostgreSQL 解耦」**：每次运行 train/test/evaluate 都会显式调用 `psycopg2.connect`。`get_node_to_path_and_type` 看起来有缓存，但**只有 `node_to_paths.pkl` 一份缓存**；`nodeid2msg`、`uuid2nids`、`indexid2msg` 都没有缓存。
- `get_node_to_path_and_type` 自身路径：`src/provnet_utils.py::get_node_to_path_and_type`，**间接被 `evaluation_utils.py::compute_tw_labels` 和 `node_evaluation.main` 调用**。
- 任务说明 §6.2/§6.3 要求的 `metadata_cache.py`（`node_metadata.pkl / uuid_to_node_id.pkl / node_id_to_uuid.pkl / ground_truth_nodes.pkl / attack_to_nodes.pkl / time_to_malicious_nodes.pkl / relation_mapping.json / dataset_manifest.json`）**全部不存在**。

---

## 9. checkpoint 保存了什么

### 9.1 实现位置

- `src/data_utils.py::save_model` / `load_model`。

### 9.2 内容

1. `state_dict.pkl`：模型参数（GraphTransformer + EdgeTypeDecoder 等）。
2. `neighbor_loader.pkl`：`LastNeighborLoader` 实例（包含 `neighbors`, `e_id`, `_assoc`, `cur_e_id`），仅当 `isinstance(model.encoder, OrthrusEncoder)`。

### 9.3 风险与缺陷

- 只在每 epoch 末尾保存一次（`orthrus_gnn_training.py` L89-91：`if cfg._test_mode or epoch % 1 == 0`），但**没有保留「最优模型」选择**——每个 epoch 都存。
- `load_model` 时**会把整个 `LastNeighborLoader` 张量重新加载到内存**，这意味着训练最后一个 epoch 的历史会原封不动带进测试。如果测试阶段未 `reset_state()`，就会存在「未来事件历史可能已被部分带入」的风险（见 §10）。
- **没有保存**：优化器状态、`full_data`、cfg、word2vec 模型 hash、metadata cache。

---

## 10. `--run_from_training` 是否存在变量未初始化问题

### 10.1 当前实现（`src/orthrus.py`）

```python
t0 = time.time()
if not args.run_from_training:
    build_orthrus_graphs.main(cfg)
    t1 = time.time()
    build_feature_word2vec.main(cfg)
    t2 = time.time()
    embed_edges_feature_word2vec.main(cfg)
    t3 = time.time()

orthrus_gnn_training.main(cfg)
torch.cuda.empty_cache()
t4 = time.time()
orthrus_gnn_testing.main(cfg)
t5 = time.time()
evaluation.main(cfg)
t6 = time.time()
tracing.main(cfg)
t7 = time.time()

time_consumption = {
    "time_total": round(t7 - t0, 2),
    "time_build_graphs": round(t1 - t0, 2),       # ← NameError 当 --run_from_training
    "time_embed_nodes": round(t2 - t1, 2),
    "time_embed_edges": round(t3 - t2, 2),
    "time_gnn_training": round(t4 - t3, 2),
    ...
}
```

### 10.2 结论

- **确认存在 bug**：当 `--run_from_training` 为 True 时，`t1, t2, t3` 未被赋值，但 `time_consumption` 计算中使用了 `t1 - t0` 等表达式，会抛 `NameError`。
- 任务说明 §5.2 已明确要求修复：被跳过阶段计时记录为 `None` 或 `0.0`。
- **本轮不修复**，仅审计。

### 10.3 其他潜在未初始化

- `cfg.detection.gnn_testing.threshold_method` 当前在 `config/orthrus.yml` 里写为字符串 `"str"`（占位符），不是合法值；`get_threshold` 解析时会抛 `ValueError("Invalid threshold method ...")`。这也是一个隐式 bug。
- `cfg.detection.gnn_testing._from_weights` 等下划线开头路径变量只有在 `set_task_paths` 后才存在；CLI `--from_weights` 只在 `load_model` 时被读取。

---

## 11. 当前测试流程是否存在历史状态重置或未来信息泄漏风险

### 11.1 历史状态重置

- `OrthrusEncoder.reset_state` 存在（清空 `LastNeighborLoader`），但：
  - 训练时每 epoch 开头会调用（`orthrus_gnn_training.py` L56-57）。
  - **测试时没有显式调用**（`orthrus_gnn_testing.py` L91-141）。模型是从 checkpoint 加载的，但 `neighbor_loader.pkl` 也被加载进来——这意味着 val 段会带着**训练结束时的邻居历史**运行，然后 test 段会带着**val 结束时的邻居历史**运行。

### 11.2 未来信息泄漏

- 严格来说，当前 `LastNeighborLoader.insert` 在每个 batch 末尾执行，所以 batch_i 的事件不会进入 batch_i 的历史，但 batch_i 的事件**会**进入 batch_{i+1} 的历史——这是正常的滑动窗口因果。
- 但 `full_data` 中拼接了 train+val+test 全部事件的 `msg / edge_type`（见 `data_utils.py::load_all_datasets`），用于「取出历史事件的边类型作为 `edge_feats`」。这本身**不会泄漏未来标签**，因为只用作 GNN 输入特征。
- **真正的潜在风险**：
  1. 测试时未 `reset_state` → 测试集第一个事件看到的「历史」实际上包含训练集最末尾若干事件。这是设计上的（**想要持续记忆**），但如果任务说明要求「验证/测试前必须基于训练集重建历史」，则需要在 testing 入口加 `reset_state` + replay train。
  2. `last_h_storage` / `last_h_non_empty_nodes`（`Orthrus.__init__`）只在 `use_contrastive_learning=True` 时启用；当前 `config/orthrus.yml` 未启用 `predict_edge_contrastive`，所以无实际影响。

### 11.3 评估阶段的「未来」

- `evaluation_utils.py::compute_tw_labels` 用 PostgreSQL 真实事件时间戳回填恶意节点列表，**没有读测试集真实 edge-loss**，因此无标签泄漏。
- 但 `standard_evaluation` 在所有 `model_epoch_*` 里**选出 best MCC** 并写入 W&B（`best_mcc` 跟踪）；这里隐含了「按测试集选择最优 epoch」的**指标泄漏**，只是未影响测试标签。任务说明 §2.2 严格禁止测试集选择超参；当前实现的 best-epoch-by-MCC 与之冲突。

### 11.4 总结风险

| 风险 | 严重度 | 说明 |
|------|--------|------|
| 测试未重置历史 | 中 | 测试集沿用训练末尾的邻居历史；若改 replay train 即可修复 |
| `best_mcc` 选 epoch 隐含测试集元信息 | 中-高 | 与任务说明 §2.2 严格无测试泄漏冲突；建议改成按 val MCC 选 |
| `full_data` 拼接三段 | 低 | 仅作特征查找，未用未来标签 |
| `cfg.detection.gnn_testing.threshold_method="str"` 占位 | 高 | 当前配置就跑不起来 |

---

## 12. 当前 Word2Vec 是否按数据集独立训练

### 12.1 实现位置

- `edge_featurization/build_feature_word2vec.py::main`：训练数据来自 `get_indexid2msg(cur, use_cmd, use_port)`，而 `cur` 由 `init_database_connection(cfg)` 打开**当前 cfg.dataset 对应数据库**。
- 不同数据集的 word2vec 模型落在各自 hash 目录下：`cfg.edge_featurization.embed_nodes._task_path` 由 `cfg.dataset.name` 决定（`config.py::set_task_paths` L390-393）。

### 12.2 结论

- **按数据集独立训练**，并各自存盘。E3 与 E5 向量空间不共享。
- **没有把 word2vec 文本语料按 train/val/test 切分**：语料来自全库 node table，但 node table 的内容本身就是由 ground truth 之外的 schema 决定的，不依赖恶意标签，所以不构成「用测试集语料训练 word2vec」的违规；但需要确认 THEIA 的恶意节点同样出现在 `subject/file/netflow_node_table` 中——这与「测试语料不能用于训练 Word2Vec」的边界比较模糊。
- 任务说明 §16 的「禁止把 E3 模型 + E5 word2vec 称为 zero-shot」目前**不存在该风险**（因为 word2vec 模型与 dataset 严格绑定在 hash 路径下）。

---

## 13. 当前实现与任务说明的关键冲突点

| 任务说明章节 | 当前实现 | 冲突描述 |
|--------------|----------|----------|
| §5.1 `--stages` 开关 | 不存在 | 整个流水线硬编码串行；想跳过某个阶段只能靠 `--run_from_training`（仅跳过 graph+embed） |
| §5.2 修复 `--run_from_training` 未初始化变量 | 存在 `NameError` | 见 §10 |
| §5.3 统一输出目录 `artifacts/<dataset>/<model>/<seed>/...` | 输出目录结构为 `artifacts/<task>/<subtask>/<hash>/<dataset>`，由 hash 决定 | 路径规划与任务说明不一致，且 hash 会把同一模型 + 同 seed 但不同代码版本的运行混在一个目录里 |
| §5.4 W&B `disabled/offline/online` | 仅支持 `--wandb`（online vs disabled 切换），无 `offline`；无 `wandb` 时 wandb.init 仍走 disabled 分支 | 缺少 `offline` 模式 |
| §6.1 环境变量路径 | 全部路径硬编码（`ROOT_ARTIFACT_DIR = "./artifacts"`，DATABASE 字段直接读取 `postgres`） | 不支持 `ORTHRUS_ARTIFACT_ROOT / ORTHRUS_DATA_ROOT / ORTHRUS_DB_*` |
| §6.2 metadata_cache | 不存在 | 每次 train/test/evaluate 都连 PostgreSQL |
| §6.3 DB 回退 | `get_node_to_path_and_type` 有 pkl 缓存但其余路径无 | 测试阶段仍强制 `nodeid2msg` 必须从 DB 读取 |
| §7 显式 src_type / dst_type / edge_type_index / event_index | 不存在；只能从 `x_src` 切片推断 | 与「不要隐式切片推断」冲突 |
| §8 时间分位数（Q50/Q90/Q99, Q20..Q80） | 不存在；唯一时间相关是 `edge_featurization.embed_edges.to_remove`（占位） | 完全缺失 |
| §9 TimeGapDecoder / 联合损失 | 不存在；只有 `EdgeTypeDecoder` | 完全缺失 |
| §10 MultiScaleNeighborLoader / 共享编码器 / 门控融合 | 不存在；只有 `LastNeighborLoader` | 完全缺失 |
| §11 分层关系条件校准 | 不存在；只有 max/mean threshold 和 KMeans | 完全缺失 |
| §12 NodeScoreAggregator (top-k) | 不存在；只有 `np.max / np.mean` | 完全缺失 |
| §13 评估指标 (FP/M, Attack Detection Rate, MCC 等) | 部分存在（MCC、AP、ROC、DOR 等），缺少 FP/M、Attack Detection Rate、efficiency 指标 | 部分缺失 |
| §14 GraphSAGE / Semantic MLP | 只支持 `GraphTransformer` | 完全缺失 |
| §15 dataset_view (host_only / host_network_structure / host_network_full) | 不存在 | 完全缺失 |
| §16 cross_domain feature_mode | 不存在 | 完全缺失 |
| §18 case study 输出 | 不存在 | 完全缺失 |
| §20 run_experiment.py / run_matrix.py / collect_results.py | 不存在 | 完全缺失 |
| §21 Colab Notebooks | 不存在 | 完全缺失 |
| §22 单元测试 | 仅有 `pytest` 隐含测试（`_test_mode` 限制边数），没有 `tests/test_*.py` | 完全缺失 |

---

## 14. 其他审计发现

### 14.1 模型/解码器接口与任务说明 §9.1 不兼容

- 当前 `EdgeTypeDecoder.forward` 签名 `(h_src, h_dst, edge_type, inference, **kwargs)` —— 直接返回 loss，没有 `logits` / `loss` 分离。任务说明要求 `logits()` 与 `loss(logits, target, reduction)` 分离。
- 时间任务头 `TimeGapDecoder` 完全缺失。
- `model_factory` 接收 `decoders` 为 list，扩展需要在此层新增 switch。

### 14.2 `encoders.py::GraphTransformer` 与 `OrthrusEncoder`

- `OrthrusEncoder` 硬编码 `LastNeighborLoader`，没有 `MultiScaleNeighborLoader` 通道。
- `forward` 中用 `full_data.msg[e_id]` / `full_data.edge_type[e_id]` 取历史事件特征——多尺度采样时要复用此模式，但**不能给当前事件注入未来信息**。当前实现中 `e_id` 是历史 id，没有泄漏风险。
- `reset_state` 只调用 `neighbor_loader.reset_state()`，**不包括任何 scale 子模块**——这是新组件的接入点。

### 14.3 `model.py::Orthrus`

- 返回值仅 `loss_or_scores`（scalar 或 `[B]`），不满足任务说明 §9.3 的字典结构 `{"loss", "loss_type", "loss_time", "loss_time_src", "loss_time_dst"}` 或 `{"score_raw", "loss_type", "loss_time", ...}`。
- `last_h_storage` / `last_h_non_empty_nodes` 只有在 contrastive 模式下启用；本期不需要 contrastive。

### 14.4 `factory.py`

- `encoder_factory` 强制返回 `GraphTransformer + OrthrusEncoder`，不支持 `graph_sage` / `mlp`。
- `decoder_factory` 只支持 `predict_edge_type`，不支持 `predict_edge_time`。
- `optimizer_factory` 固定 Adam，没有 warmup / scheduler。

### 14.5 `data_utils.py::GraphReindexer`

- `node_features_reshape` 维护了一个 `[num_nodes, in_dim]` 的 `x_src_cache / x_dst_cache`，每 batch 调用 `detach()`。这是一个**写时替换语义**——`x_src_cache[edge_index[0]] = x_src` 会把上次未被覆盖的节点位置保留为上次的值。这对于「相同节点在不同 batch 中可能需要不同表示」的语义来说是安全的（因为每次 batch 的 `x_src` 都覆盖相关节点），但**未初始化节点会保留上一次的值，可能引起隐性状态泄漏**。
- `reindex_graph` 会把 `original_edge_index` 存入 `data.original_edge_index`，`orthrus_gnn_testing.py` 利用这点回查原始 nid。
- 但 **`batch_loader_factory` 调用 `custom_temporal_data_loader(data, batch_size=...)` 永远走 `TemporalDataLoader`，从未真正调用 `reindex_graph`**（L124-125 是死代码）——所以 `original_edge_index` 永远不会被设置；`orthrus_gnn_testing.py` 的 `if hasattr(batch, "original_edge_index")` 分支永远走 else 路径。

### 14.6 `eval_utils` / `node_evaluation`

- `analyze_false_positives` 计算的是「FP 是否落在 malicious TW 内」，**不参与 MCC/PR**——它是诊断性的，对论文写作有用。
- `compute_kmeans_labels` 在 `topk_K=20` 下，对 THEIA 这种每 TW 数百节点的场景，结果非常稀疏。

### 14.7 测试阶段的数据流

- `orthrus_gnn_testing.py::test()` 一次性跑完单个 `TemporalData` 内的所有 batch，**没有显式 batch 粒度的 ground truth 标注**——只输出 edge loss。要做 per-event analysis 需要额外写 CSV 列。
- `edge_types = torch.argmax(batch.edge_type, dim=1) + 1` 用来查 `rel2id` 还原字符串。注意 `rel2id` 同时存 int→str 和 str→int 两种映射，对 `argmax+1` 之后取 `rel2id[idx]` 返回 str——OK。

### 14.8 与 THEIA 数据相关的隐含约束

- `cfg.dataset.attack_to_time_window`（THEIA_E3 两条、THEIA_E5 一条）只用于 `evaluation_utils.compute_tw_labels` 把恶意事件时间对齐到 TW——**不进入训练或阈值计算**。
- `_test_mode=True` 时 TW 边界由 `use_all_files` + 2000 edges 控制；CI / smoke test 可走这条路径。

---

## 15. 关键风险一览（高/中/低）

### 15.1 高风险

1. **`--run_from_training` 触发 `NameError`**：`src/orthrus.py` L71-79 计算 `time_consumption` 时使用了未初始化的 `t1/t2/t3`。任务说明 §5.2 要求修。
2. **`cfg.detection.gnn_testing.threshold_method` 当前值是占位符 `"str"`**：跑测试评估必报错。任务说明 §12 要求支持至少 3 种方法。
3. **`evaluation.standard_evaluation` 按 test MCC 选 best epoch**：违反任务说明 §2.2（不得用测试集选择超参）。
4. **测试入口未 `reset_state` + 未 replay train**：导致验证/测试时邻居历史非纯训练历史。任务说明 §10.8 要求严格 replay。

### 15.2 中风险

5. **没有 metadata cache**：train/test/evaluate 阶段硬依赖 PostgreSQL，违背任务说明 §6.3 的「已有 artifacts 时不需要 PostgreSQL」。
6. **`EdgeTypeDecoder` 接口耦合 loss 与 logits**，无法满足任务说明 §9.1 要求的 `logits()` / `loss(logits, target, reduction)` 拆分。
7. **`OrthrusEncoder` 直接读 `full_data.edge_type[e_id]`** 作为边特征——历史事件的真实边类型是允许的，但实现时需在新采样器中明确「e_id 集合不包含当前事件」。
8. **数据视图（host_only / host_network_structure / host_network_full）缺失**：任务说明 §15 要求三视图共享同一 train/val/test 日期与同一 cfg。

### 15.3 低风险

9. **`save_model` 把 `LastNeighborLoader` 整体持久化**：在大节点集下体积较大，建议改 replay 模式（任务说明 §10.8）。
10. **`evaluation.py::compute_tw_labels` 总是先 `os.remove(out_file)` 再重算**——若已有 cache 被合法生成，会被强制删除。任务说明希望 cache 复用。
11. **`evaluation_utils.calculate_supervised_best_threshold` 处于死代码状态**，但不影响主流程；不要在本期删除，避免破坏 baseline 兼容。

---

## 16. 审计结论

- 当前仓库是**单基线（ORTHRUS + GraphTransformer + EdgeTypeDecoder）**实现，缺少 MSTC-PIDS 任务说明要求的全部创新模块（多尺度采样、双任务解码、分层校准、Top-k 聚合、跨骨干、metadata 缓存、统一输出目录、stage 开关、单元测试、Colab Notebook）。
- 当前实现存在 **1 个必然崩溃的 bug**（`--run_from_training`）、**1 个配置占位符 bug**（`threshold_method="str"`）、**1 个违反任务约束的选择**（best epoch by test MCC）。
- 本期最小改动需要：
  1. 修复 `orthrus.py` 计时变量（5 行内）；
  2. 替换 `threshold_method` 默认值为合法方法（1 行）；
  3. 改 `evaluation.py` 用 val MCC 选 best epoch（5 行内）；
  4. 测试入口 `reset_state` + train replay（需要新增 replay 函数，约 30 行）；
  5. 新增 `src/mstc/` 8 个文件（任务说明 §4 已列出）；
  6. 新增 `config/experiments/*.yml`（任务说明 §4 已列出）；
  7. 新增 `tests/test_*.py`（任务说明 §22 已列出）；
  8. 新增 `src/experiments/*.py`（任务说明 §4 已列出）；
  9. 新增 `notebooks/*.ipynb`（任务说明 §21 已列出）。

任何超出上述列表的大范围重构都视为「违反 §2.4 不要大规模重写」。