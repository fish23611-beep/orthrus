# ORTHRUS-MSTC-PIDS 完整改造任务说明

## 1. 项目目标

基于官方 ORTHRUS 仓库实现一个研究分支，暂定名称：

**MSTC-PIDS: Multi-Scale Temporal Consistency Learning with Hierarchical Relation-Conditional Calibration**

本篇 SCI 小论文的研究范围仅包括：

- THEIA_E3；
- THEIA_E5；
- 异常事件评分；
- 异常节点检测与节点异常排名；
- 多尺度时间上下文；
- 边类型与时间间隔双任务预测；
- 分层关系条件校准；
- Top-k 节点聚合；
- GraphTransformer 与 GraphSAGE 跨骨干验证；
- Semantic MLP 简单语义基线；
- All-in-One Notebook 下的训练、测试、恢复、评估和结果汇总。

本篇论文明确不实现、也不纳入实验矩阵：

- auditd + Zeek 多源日志融合；
- 自建靶场；
- 日志缺失实验；
- 攻击路径重建；
- CALDERA；
- Host-only / Host + Network Structure / Host + Network Full 三视图消融；
- E3→E5 或 E5→E3 的跨 Engagement / Zero-shot 迁移；
- LLM；
- 对比学习；
- 复杂图增强；
- history_device CPU/CUDA 对比实验；
- official_full_dataset 与 train_only 的论文对比实验。

主要比较对象是：

1. 原始 ORTHRUS-ano；
2. Semantic MLP；
3. GraphSAGE；
4. 完整 MSTC-PIDS。

论文实验遵循以下总原则：

- **主结果：THEIA_E3 与 THEIA_E5 都运行**；
- **完整消融和专项分析：主要集中在 THEIA_E3**；
- 主结果与完整消融默认使用 3 个随机种子 `0,1,2`，报告 `mean ± standard deviation`；
- 可直接复用已有事件级产物的后处理实验不得重新训练模型。

## 2. 总体开发原则

### 2.1 保留原始基线

必须保留一个完全兼容原始ORTHRUS的运行模式：

```yaml
model:
  variant: orthrus_baseline
```

当所有新模块关闭时，模型的行为、损失和输出应尽可能与原仓库一致。

不得直接删除原有：

- `LastNeighborLoader`；
- `OrthrusEncoder`；
- `EdgeTypeDecoder`；
- 原始阈值方法；
- 原始K-means方法。

新功能通过配置开关启用。

### 2.2 不允许测试集泄漏

禁止：

- 使用测试集标签选择阈值；
- 使用测试集攻击节点调节超参数；
- 使用测试集损失选择最优校准方式；
- 使用测试集语料训练论文主实验的 Word2Vec；
- 在计算当前事件表示前把当前事件插入编码器历史；
- 把当前事件真实边类型作为当前事件的模型输入特征；
- 把当前事件真实时间桶直接作为模型输入特征。

所有时间分位数、时间桶、校准参考分布和模型超参数只能来自：

- 正常训练集；
- 正常验证集；
- 预先固定的配置。

论文正式主结果中，所有模型必须统一使用：

```yaml
semantic_features:
  corpus_scope: train_only
```

包括 ORTHRUS-ano、Semantic MLP、GraphSAGE 和 MSTC-PIDS。不得让 baseline 使用 `official_full_dataset` 而 MSTC-PIDS 使用 `train_only`，否则比较不公平。

`official_full_dataset` 仅作为官方 ORTHRUS 兼容/复现模式保留，必须明确标记为 compatibility-only，不得与 `train_only` 的论文主结果混入同一主表。

### 2.3 模块化

所有新模块必须：

- 可单独启用或关闭；
- 可通过YAML和CLI控制；
- 可独立做消融；
- 可在GraphTransformer和GraphSAGE骨干上运行；
- 支持CPU和CUDA；
- 支持断点恢复；
- 支持Google Drive持久化；
- 不依赖W&B才能运行。

### 2.4 不要大规模重写原项目

优先新增模块并做适配，不要把整个项目改造成另一套框架。

建议创建分支：

```bash
git checkout -b mstc-pids
```

每个阶段单独提交。

---

# 3. 首先完成代码审计

在开始修改前，检查以下文件并生成：

```text
docs/CODE_AUDIT.md
```

需要说明：

- 当前训练入口；
- 当前测试入口；
- 当前数据结构；
- 当前节点和边特征；
- 当前历史邻居保存方式；
- 当前模型状态保存方式；
- 当前异常分数生成方式；
- 当前节点聚合方式；
- 当前阈值方法；
- 当前数据库依赖；
- 每个拟修改文件的职责。

重点检查：

```text
src/orthrus.py
src/config.py
src/data_utils.py
src/factory.py
src/model.py
src/encoders.py
src/decoders.py
src/temporal.py
src/labelling.py
src/detection/orthrus_gnn_training.py
src/detection/orthrus_gnn_testing.py
src/detection/evaluation.py
src/detection/node_evaluation.py
src/detection/evaluation_utils.py
```

如果当前仓库内容与本说明中的文件结构不同，以实际代码为准，但保持本说明的功能目标不变。

---

# 4. 建议的新目录结构

新增或维护：

```text
src/mstc/
    __init__.py
    history_store.py
    multiscale_sampler.py
    multiscale_encoder.py
    time_gap.py
    calibration.py
    calibration_runner.py
    aggregation.py
    metrics.py
    metadata_cache.py
    experiment_utils.py

src/experiments/
    __init__.py
    run_experiment.py
    run_matrix.py
    run_topk_sensitivity.py
    collect_results.py
    export_tables.py

config/experiments/
    baseline.yml
    mstc_full.yml
    ablation_no_multiscale.yml
    ablation_no_gate.yml
    ablation_no_time.yml
    ablation_no_calibration.yml
    ablation_no_topk.yml
    multiscale_recent20.yml
    multiscale_recent24.yml
    multiscale_single_window.yml
    multiscale_equal.yml
    multiscale_gate.yml
    time_type_only.yml
    time_time_only.yml
    time_joint.yml
    calibration_none.yml
    calibration_global_p.yml
    calibration_relation.yml
    calibration_hierarchical.yml
    threshold_quantile.yml
    threshold_max_validation.yml
    threshold_kmeans.yml
    backbone_graphtransformer.yml
    backbone_graphsage.yml
    backbone_mlp.yml

config/analysis/
    topk_sensitivity.yml

notebooks/
    <现有 All-in-One Notebook 的实际文件名>.ipynb

tests/
    test_time_gap.py
    test_time_gap_decoder.py
    test_multiscale_sampler.py
    test_multiscale_encoder.py
    test_calibration.py
    test_aggregation.py
    test_no_future_leakage.py
    test_checkpoint_resume.py
    test_baseline_compatibility.py
    test_topk_sensitivity.py
```

说明：

- 不再为 `K=1/3/10/20` 创建独立训练配置；Top-k 敏感性通过专用后处理入口完成。
- 不再创建 `host_only.yml / host_network_structure.yml / host_network_full.yml`。
- 不再维护 6 个互相独立的 Colab Notebook；All-in-One Notebook 是唯一正式 Notebook 入口。
- `calibration_runner.py` 是校准 I/O 的唯一 owner；`calibration.py` 只保存算法类。

# 5. 阶段一：基线保护与流水线重构

## 5.1 增加独立流水线阶段

修改`src/orthrus.py`，增加：

```bash
--stages preprocess,train,test,evaluate
```

允许：

```bash
--stages preprocess
--stages train
--stages test,evaluate
--stages train,test,evaluate
--stages all
```

定义：

```text
preprocess = graph_construction + node_embedding + edge_embedding
train      = gnn_training
test       = gnn_testing
evaluate   = node_evaluation
trace      = attack reconstruction
```

默认不运行攻击重建：

```bash
--skip-tracing
```

或者：

```yaml
pipeline:
  run_tracing: false
```

### 5.2 修复`run_from_training`

检查原始`src/orthrus.py`中：

```text
t1
t2
t3
```

是否只在非`run_from_training`分支初始化、却在最后统一使用。

确保：

```bash
--run_from_training
```

不会因未定义计时变量而失败。

被跳过阶段的耗时应记录为：

```python
None
```

或者：

```python
0.0
```

### 5.3 统一输出目录

论文与实验工具统一采用一个 canonical artifact contract，避免训练、评估和结果收集各自猜路径。

```text
artifacts/
  graph_construction/
  edge_featurization/
  metadata/
  matrix_artifacts/
    <config_id>/
      <dataset>/
        runs/
          <model_variant>/
            seed_<seed>/
              config_resolved.yml
              environment.json
              checkpoints/
              edge_scores/
                test/
                  event_predictions.csv
              node_scores/
                metrics.json
                node_predictions.csv
              evaluation_results/
                calibration/
              runtime.json
  analyses/
    topk_sensitivity/
      <source_config_id>/
        <dataset>/
          results/
            topk_sensitivity_results.csv
            topk_event_count_summary.csv
```

`collect_results.py`、`export_tables.py` 和 All-in-One Notebook 都必须按上述 contract 查找文件，不得再假设 `artifact_root/runs/...` 这种缺少 dataset/config-id 层的路径。

每次训练运行保存：

- 数据集；
- 模型名和 model variant；
- config-id / config hash；
- seed；
- Git commit；
- 完整解析后的配置；
- Python版本；
- PyTorch版本；
- PyG版本；
- GPU型号；
- CUDA版本；
- 开始和结束时间；
- 峰值GPU显存；
- 峰值CPU内存；
- 每秒处理事件数量。

### 5.4 W&B可选

支持：

```yaml
logging:
  wandb_mode: disabled
```

合法值：

```text
disabled
offline
online
```

即使没有W&B API key，也必须能完成全部实验。

### 5.5 模型选择协议

论文正式结果统一使用 validation objective 选择 checkpoint，不使用 test MCC。

建议：

```yaml
model_selection:
  method: min_val_objective
```

其中 validation objective 与当前模型训练目标一致：

```text
Type-only / ORTHRUS baseline:
    mean validation loss_type

Joint MSTC:
    mean validation score_raw
    = mean(loss_type + lambda_time * loss_time)
```

也允许预先固定：

```yaml
model_selection:
  method: last_epoch
```

但同一实验组必须使用预先声明的协议。`legacy_test_selection` 只为官方兼容保留，默认关闭并明确标记 `NOT FOR PAPER`。

---

# 6. 阶段二：Colab适配、数据库解耦与语义语料边界

## 6.1 路径使用环境变量

修改 `src/config.py`，支持：

```bash
export ORTHRUS_ARTIFACT_ROOT=/content/drive/MyDrive/mstc_pids/artifacts
export ORTHRUS_DATA_ROOT=/content/data
export ORTHRUS_DB_HOST=127.0.0.1
export ORTHRUS_DB_PORT=5432
export ORTHRUS_DB_USER=postgres
export ORTHRUS_DB_PASSWORD=postgres
```

代码中使用 `os.environ.get(...)`，保留原默认值作为回退。

## 6.2 预处理后导出数据库元数据

增加：

```text
src/mstc/metadata_cache.py
```

在数据库可用时导出：

```text
metadata/node_metadata.pkl
metadata/uuid_to_node_id.pkl
metadata/node_id_to_uuid.pkl
metadata/ground_truth_nodes.pkl
metadata/attack_to_nodes.pkl
metadata/time_to_malicious_nodes.pkl
metadata/relation_mapping.json
metadata/dataset_manifest.json
```

`node_metadata.pkl` 结构：

```python
{
    node_id: {
        "uuid": str,
        "type": "subject" | "file" | "netflow",
        "path": str | None,
        "cmd": str | None,
        "local_ip": str | None,
        "local_port": str | None,
        "remote_ip": str | None,
        "remote_port": str | None,
        "display": str,
    }
}
```

不再单独要求一个定义不清的 `nodeid2msg.pkl`。如果测试 CSV 需要可读文本，优先由 `node_metadata[node_id]["display"]` 生成；只有为了兼容旧产物时才允许读取旧 `nodeid2msg.pkl`。

`dataset_manifest.json` 至少包含：

```json
{
  "dataset": "THEIA_E3",
  "num_node_types": 3,
  "num_edge_types": 10,
  "train_files": [],
  "val_files": [],
  "test_files": [],
  "word2vec_dim": 128,
  "semantic_corpus_scope": "train_only",
  "preprocess_config_hash": "",
  "created_at": ""
}
```

## 6.3 数据库回退机制

修改：

```text
src/labelling.py
src/detection/orthrus_gnn_testing.py
src/detection/evaluation_utils.py
```

行为：

```text
元数据缓存存在
    ↓
直接读取缓存，不连接 PostgreSQL

元数据缓存不存在
    ↓
回退到原始 PostgreSQL 查询
```

训练、测试和评估阶段在已有预处理缓存时不得强制要求 PostgreSQL。

`srcmsg/dstmsg` 只用于展示，不应成为测试必须项：

```yaml
testing:
  include_node_messages: false
```

## 6.4 Word2Vec 语料范围必须真实生效

仅增加配置字段是不够的，必须修改实际语料构建逻辑，使以下两种模式语义可验证：

```yaml
semantic_features:
  corpus_scope: train_only
```

论文主实验模式：只允许训练日期/训练 split 中出现的文本或节点语义进入 Word2Vec 语料；validation/test 只做 lookup/OOV 处理，不参与 Word2Vec fit。

```yaml
semantic_features:
  corpus_scope: official_full_dataset
```

仅用于官方 ORTHRUS 兼容复现；允许沿用原始整库语料行为，但运行目录和结果必须带 `compatibility_only=true` 标记。

必须增加测试确认 `train_only` 模式下 validation/test 专属 token 不会进入 Word2Vec 训练语料。

## 6.5 运行模式统一到 stages

不再维护第二套独立 pipeline API。统一使用：

```bash
--stages preprocess,train,test,evaluate
```

文档中：

```text
full_pipeline  = preprocess,train,test,evaluate
detection_only = train,test,evaluate（已有预处理 artifacts）
evaluate_only  = evaluate（已有事件级/节点级所需产物）
```

这些名称只作为运行场景描述，真正执行仍由 `--stages` 控制，避免 `pipeline.mode` 与 `--stages` 两套状态机互相冲突。

# 7. 阶段三：显式保留事件类型信息

修改`src/data_utils.py`。

在解析消息时，为每个TemporalData显式保存：

```python
g.src_type
g.dst_type
g.edge_type_index
g.event_index
```

格式：

```python
g.src_type.shape == [num_events]
g.dst_type.shape == [num_events]
g.edge_type_index.shape == [num_events]
g.event_index.shape == [num_events]
```

其中类型索引从0开始。

不要在后处理阶段依赖：

```python
x_src[:, -3:]
```

这种隐式切片推断类型。

保留现有one-hot特征以兼容原模型。

为每个时间窗口检查：

- 时间戳非递减；
- `src`、`dst`、`t`长度一致；
- 类型索引合法；
- 边类型one-hot与索引一致。

---

# 8. 阶段四：训练数据时间统计

新增：

```text
src/mstc/time_gap.py
```

实现：

```python
class TimeGapStatistics
class TimeGapTargetBuilder
```

必须把“时间目标状态”和“编码器历史状态”分开，避免 micro-batch 语义混淆。

## 8.1 精确事件级时间间隔

按训练数据的事件时间顺序扫描。对于当前事件：

```text
e_t = (u, v, r, t)
```

定义真实监督目标：

```python
delta_src = t - target_last_seen[src]
delta_dst = t - target_last_seen[dst]
```

未见过的节点标记为 `NO_HISTORY`。

`TimeGapTargetBuilder` 在一个 batch 内按 `(timestamp, global_event_index)` 稳定排序，逐事件：

1. 先根据 `target_last_seen` 计算当前事件 target；
2. 再更新 `target_last_seen`；
3. target 只用于 loss/评估，绝不作为模型输入。

因此同一 batch 内后出现的事件可以获得精确的“距离上一真实事件”的 target；这不构成输入泄漏，因为真实 target 不参与表示计算。

编码器的 MultiScaleHistory 仍遵守 §10 的 micro-batch 查询后插入协议，两套状态不得共用。

时间戳转换：

```python
delta_seconds = delta_ns / 1_000_000_000
z = log1p(delta_seconds)
```

禁止直接对纳秒值做时间分类分桶。

## 8.2 多尺度边界

仅使用正常训练集有限 `delta_seconds` 计算：

```python
tau_short_seconds = Q50(delta_seconds)
tau_medium_seconds = Q90(delta_seconds)
tau_extreme_seconds = Q99(delta_seconds)
```

其中：

- Q50 和 Q90 是多尺度采样的实际边界；
- Q99 只作为极端长间隔诊断值和 `Single-window-24` 的时间窗上界；
- **Q99 不再作为 long scale 的硬截断**，避免把最老 1% 历史直接排除。

保存：

```text
metadata/time_statistics.json
```

建议结构：

```json
{
  "raw_unit": "seconds",
  "time_bucket_space": "log1p_seconds",
  "scale_quantiles": [0.5, 0.9, 0.99],
  "scale_boundaries_seconds": [0.0, 0.0, 0.0],
  "time_bucket_quantiles": [0.2, 0.4, 0.6, 0.8],
  "time_bucket_boundaries_log1p": [0.0, 0.0, 0.0, 0.0]
}
```

## 8.3 时间预测类别

定义六类：

```text
0 = NO_HISTORY
1 = VERY_SHORT
2 = SHORT
3 = MEDIUM
4 = LONG
5 = VERY_LONG
```

有限时间间隔先计算：

```python
z = log1p(delta_seconds)
```

再使用正常训练集 `z` 的：

```text
Q20
Q40
Q60
Q80
```

划分五个有限区间。

时间桶边界只能由训练集生成，验证和测试直接加载固定边界。

## 8.4 数据检查

必须处理：

- 重复时间戳；
- 相同时间戳下按 `global_event_index` 稳定排序；
- 非递增输入序列；
- 负时间间隔；
- 节点第一次出现；
- 极大时间间隔；
- 空训练数据。

负间隔应抛出明确错误或记录异常，不能默默取绝对值。

# 9. 阶段五：双任务一致性学习

## 9.1 解码器接口

保留原始：

```python
class EdgeTypeDecoder
```

不要为了 MSTC 修改其对外行为。

新增：

```python
class TimeGapDecoder(nn.Module):
    def forward(self, h_src, h_dst):
        return src_logits, dst_logits
```

时间头采用：

```text
concat(h_src, h_dst)
    ↓
共享隐藏层 hidden_dim
    ↓
src 时间桶分类头 / dst 时间桶分类头
```

不要先把共享层压到 6 维再接两个分类头，也不要把真实时间桶作为输入。

MSTC 使用独立 `MSTCOrthrus` wrapper 组合原始 edge decoder 与新的 time decoder；原始 `Orthrus` 和 `OrthrusEncoder` 保持兼容。

## 9.2 损失与原始异常分数

边类型逐事件损失：

```python
loss_type_i = CE(edge_logits_i, edge_type_target_i)
```

时间逐事件损失：

```python
loss_time_src_i = CE(src_time_logits_i, src_time_target_i)
loss_time_dst_i = CE(dst_time_logits_i, dst_time_target_i)
loss_time_i = 0.5 * (loss_time_src_i + loss_time_dst_i)
```

训练 batch 总损失：

```python
loss_total = mean(loss_type) + lambda_time * mean(loss_time)
```

默认：

```yaml
lambda_time: 0.3
```

**必须唯一明确 `score_raw` 的定义：**

```python
# Joint / MSTC-PIDS Full
score_raw_i = loss_type_i + lambda_time * loss_time_i

# Type-only
score_raw_i = loss_type_i

# Time-only
score_raw_i = loss_time_i
```

后续校准器只接收该配置对应的 `score_raw`。不得在不同脚本里再用另一套“总 loss”定义。

## 9.3 模型返回值

训练模式：

```python
{
    "loss": scalar_total_loss,
    "loss_type": scalar,
    "loss_time": scalar,
    "loss_time_src": scalar,
    "loss_time_dst": scalar,
}
```

测试模式：

```python
{
    "score_raw": [E],
    "loss_type": [E],
    "loss_time": [E],
    "loss_time_src": [E],
    "loss_time_dst": [E],
    "edge_logits": [E, R],
    "src_time_logits": [E, B],
    "dst_time_logits": [E, B],
}
```

训练循环使用：

```python
outputs["loss"].backward()
```

## 9.4 测试 CSV 字段

测试输出至少包含：

```text
global_event_index
time
srcnode
dstnode
src_type
dst_type
edge_type
edge_type_index
loss_type
loss_time_src
loss_time_dst
loss_time
score_raw
src_time_target
dst_time_target
src_time_prediction
dst_time_prediction
```

不要只保留一个总 loss。

# 10. 阶段六：数据驱动多尺度时间邻域

## 10.1 新历史加载器

新增：

```text
src/mstc/history_store.py
src/mstc/multiscale_sampler.py
```

实现：

```python
class HistoryStore
class MultiScaleNeighborLoader
```

不要破坏原 `LastNeighborLoader`。

配置：

```yaml
multiscale:
  enabled: true
  candidate_capacity: 64
  history_device: cpu
  short_budget: 8
  medium_budget: 8
  long_budget: 8
  share_encoder: true
  fusion: gated
  use_scale_embedding: true
```

历史存储规则：

- `timestamp_ns` **必须使用 int64**；
- `direction` 可使用 int8；
- `node_id/event_id` 只有在运行时范围检查证明安全后才允许 int32，否则保持 int64；
- 不把 edge-type one-hot 或节点 embedding 复制进历史存储，通过 `event_id` 从 `full_data` 读取。

`candidate_capacity=64` 是资源上限，不新增容量参数扫描。正式实验必须记录：

```text
short_empty_ratio
medium_empty_ratio
long_empty_ratio
candidate_truncation_ratio
```

若 long scale 经常因容量截断而为空，只能依据 train/validation 统计统一调整容量，然后所有正式对照重新使用同一容量；不得依据 test 指标调节。

## 10.2 查询顺序

对于每个 micro-batch：

```text
1. 计算当前节点的 reference_time
2. 查询编码器历史
3. 构建三个尺度子图
4. 编码当前 batch
5. 计算预测和 loss
6. 最后把当前 batch 事件插入编码器历史
```

绝对禁止先插入当前事件再查询编码器历史。

注意：§8 的 `TimeGapTargetBuilder` 是独立监督状态，可以在 batch 内逐事件更新；它不等于这里的编码器 HistoryStore。

## 10.3 批次中同一节点的参考时间

一个节点可能在当前 batch 多次出现。编码器上下文对该节点使用：

```python
reference_time[node] = min(current_event_times_of_node)
```

因此同一 micro-batch 中所有表示只看到 batch 之前的历史，称为：

```text
causal micro-batch temporal context
```

不要声称编码器中 batch 后部事件能看到 batch 前部事件。

## 10.4 时间分组

对历史边：

```python
delta_seconds = (current_reference_time - historical_edge_time) / 1e9
```

划分：

```python
short:
    0 < delta_seconds <= tau_short_seconds       # Q50

medium:
    tau_short_seconds < delta_seconds <= tau_medium_seconds  # Q90

long:
    delta_seconds > tau_medium_seconds
```

`tau_extreme_seconds=Q99` 不截断 long，只用于诊断与 Single-window 对照。

每个尺度只保留该尺度中最近的 K 条边；不足 K 不重复填充；为空提供显式 mask。

## 10.5 公平邻居预算

完整模型默认最大预算：

```text
8 + 8 + 8 = 24
```

专项比较定义固定为：

```text
Recent-20:
    官方 ORTHRUS 邻居预算参考；忽略时间窗，取最近 20 条历史边

Recent-24:
    公平预算对照；忽略时间窗，取最近 24 条历史边

Single-window-24:
    仅保留 0 < delta <= tau_extreme_seconds(Q99) 的历史，再取最近 24 条

Multi-scale Equal-24:
    short/medium/long = 8/8/8，非空尺度等权融合

Multi-scale Gated-24:
    short/medium/long = 8/8/8，学习门控融合
```

结论中比较 Multi-scale 与 Recent 时，以 `Recent-24` 作为公平预算主对照；`Recent-20` 只用于与官方邻居预算衔接。

只有 `Multi-scale Gated-24` 报告 learned gate weight distribution；Recent 和 Equal 方案不要求“门控权重分布”。

## 10.6 多尺度编码器

新增：

```python
class MultiScaleOrthrusEncoder(nn.Module)
```

三个尺度共享的是一个**可配置 graph backbone 实例**，而不是硬编码 GraphTransformer：

```python
self.shared_graph_encoder = backbone_factory(cfg.encoder.backbone)
```

允许：

```text
graph_transformer
graphsage
```

三个尺度：

```text
短期子图 → shared graph backbone → h_src_short, h_dst_short
中期子图 → shared graph backbone → h_src_medium, h_dst_medium
长期子图 → shared graph backbone → h_src_long, h_dst_long
```

GraphSAGE 不使用 edge features 时必须显式忽略 `edge_attr`，但其多尺度采样和门控语义保持一致。

每个尺度可加入可学习 scale embedding。

## 10.7 门控融合

对每个当前事件构造：

```python
z_short = concat(h_src_short, h_dst_short)
z_medium = concat(h_src_medium, h_dst_medium)
z_long = concat(h_src_long, h_dst_long)
```

门控输入：

```python
gate_input = concat(
    z_short,
    z_medium,
    z_long,
    current_src_projection,
    current_dst_projection
)
```

输出：

```python
gate_logits = gate_mlp(gate_input)
beta = masked_softmax(gate_logits, non_empty_scale_mask)
```

融合：

```python
h_src = beta_short*h_src_short + beta_medium*h_src_medium + beta_long*h_src_long
h_dst = beta_short*h_dst_short + beta_medium*h_dst_medium + beta_long*h_dst_long
```

要求：

```python
beta.sum(dim=-1) == 1
```

空尺度不得获得权重；三个尺度都为空时回退到当前节点投影。

## 10.8 历史状态

接口：

```python
reset_state()
warmup(data)
state_dict()
load_state_dict()
insert(...)
query(...)
```

默认：

```yaml
history:
  checkpoint_mode: replay
```

测试 checkpoint 前：

```text
加载模型权重
→ reset encoder history
→ replay train，仅构建编码器历史
→ 连续处理 validation
→ 不 reset
→ 连续处理 test
```

训练 checkpoint 与评估产物分开：

- training checkpoint 保存模型/优化器/RNG/epoch/config hash；
- calibration、threshold、node aggregation 结果属于 evaluation artifacts，不塞进训练 checkpoint。

## 10.9 历史设备

论文正式实验统一：

```yaml
history_device: cpu
```

CUDA 作为代码兼容能力可以保留，但不做 CPU vs CUDA 论文实验。效率表直接报告整体 CPU 内存、GPU 显存、推理时间和 events/s。

# 11. 阶段七：分层关系条件校准

新增：

```text
src/mstc/calibration.py
src/mstc/calibration_runner.py
```

实现：

```python
class HierarchicalRelationCalibrator
```

`calibration.py` 只负责算法；`calibration_runner.py` 是读取事件分数、拟合、转换和保存产物的唯一 I/O owner。

## 11.1 条件组

一级：

```python
c1 = (src_type, edge_type, dst_type)
```

二级：

```python
c2 = (src_type, dst_type)
```

三级：

```python
c3 = GLOBAL
```

配置：

```yaml
calibration:
  method: hierarchical_relation
  min_triplet_samples: 100
  min_type_pair_samples: 200
  epsilon: 1.0e-12
```

回退：

```text
三元组支持数 >= 100 → triplet
否则类型对支持数 >= 200 → type_pair
否则 → global
```

`100/200` 是预先固定的最小支持数，不依据 test 搜索最优值。本篇论文不额外增加这两个参数的网格扫描，但必须报告各层 fallback rate 和样本数分布。

## 11.2 经验 p 值

只用正常 validation 建立参考分布。

测试事件：

```python
p = (1 + count(reference_scores >= test_score)) / (n_reference + 1)
score_calibrated = -log(max(p, epsilon))
```

每个条件组排序并使用 `numpy.searchsorted`，避免逐事件遍历全部参考分数。

**关于真实 `edge_type`：** 条件校准发生在模型已经输出 `score_raw` 之后。真实 relation 只用于选择后处理参考分布，不参与当前事件表示或 edge-type 预测，因此不得把这种 post-hoc conditioning 描述成模型输入。

## 11.3 验证集自身评分

validation 用于确定节点阈值时，必须使用 leave-one-out empirical p-value，避免事件把自己作为参考样本。

## 11.4 保存校准器

由 `calibration_runner.py` 保存：

```text
calibrator.pkl
calibration_summary.json
```

摘要至少包含：

- 每个 triplet 样本数；
- 每个 type-pair 样本数；
- triplet/type_pair/global fallback 比例；
- 全局样本数；
- 各组 score 分位数；
- test 事件实际使用各层的比例。

# 12. 阶段八：节点 Top-k 聚合

新增：

```text
src/mstc/aggregation.py
```

实现：

```python
class NodeScoreAggregator
```

建议拆成两个明确步骤：

```python
build_node_event_scores(event_records, include_dst=True)
aggregate(node_to_event_scores, method, topk)
```

`include_dst` 必须在 event→node 映射阶段生效，而不是等 `node_to_event_scores` 已经构造完成后再传给 reducer。

默认：

```yaml
node_aggregation:
  method: topk_mean
  topk: 5
  include_dst: true
```

对于节点 v，其关联事件校准分数为 `S_v`：

```python
if len(S_v) >= K:
    node_score = mean(K largest values in S_v)
else:
    node_score = mean(S_v)
```

代码可以保留：

```text
max
mean
topk_mean
topk_sum
```

但本篇论文的聚合对照只比较：

```text
mean
max
topk_mean
```

`topk_sum` 不进入论文实验矩阵。

## 12.1 Top-k 参数约束与敏感性分析

论文主模型预先固定：

```text
K = 5
```

敏感性：

```text
K ∈ {1, 3, 5, 10, 20}
```

该实验是**纯后处理**，不再为不同 K 创建新的训练 config-id/checkpoint。

新增：

```text
src/experiments/run_topk_sensitivity.py
config/analysis/topk_sensitivity.yml
```

配置示例：

```yaml
source_config: mstc_full
ks: [1, 3, 5, 10, 20]
seeds: [0, 1, 2]
dataset: THEIA_E3
```

对每个 seed 必须复用同一个 `mstc_full`：

- checkpoint；
- validation/test `score_raw`；
- 同一个已拟合 calibrator；
- validation/test `score_calibrated`。

仅重新执行：

```text
validation event→node 映射与聚合
→ validation node threshold
→ test event→node 映射与聚合
→ metrics
```

每个 K 的阈值**方法和 quantile 固定**，但数值阈值必须根据该 K 的正常 validation 节点分数重新计算，禁止复用 K=5 的数值阈值。

输出：

```text
results/topk_sensitivity_results.csv
results/topk_event_count_summary.csv
```

`topk_event_count_summary.csv` 至少包含：

```text
split
seed
K
num_nodes
num_nodes_lt_k
ratio_nodes_lt_k
```

用于解释较大 K 下多少节点退化为 `mean(all_event_scores)`。

敏感性结果只证明合理 K 范围内的稳定性；不得根据 THEIA_E3 test 指标重新选择主模型 K。

## 12.2 节点阈值

论文主方法：

```yaml
node_threshold:
  method: validation_quantile
  quantile: 0.999
```

`0.999` 是预先固定的保守正常验证分位数，用于极端类别不平衡下控制告警量。本篇论文不再为该 quantile 做大规模参数搜索；正文应把它声明为 fixed operating point，而不是声称其为 test-optimal threshold。

另外保留：

```text
max_validation
kmeans
```

作为节点决策对照。

禁止在 test 节点标签上搜索阈值。

# 13. 阶段九：评估指标

新增或补充：

```text
src/mstc/metrics.py
```

节点级主指标：

```text
TP
FP
TN
FN
Precision
Recall
F1
MCC
AUPRC
AUROC
False Positive Rate
FP per million benign nodes
Number of inspected nodes per attack
Attack Detection Rate
```

定义必须固定：

```python
num_benign_nodes = number of unique benign nodes in the TEST split
fp_per_million = FP / num_benign_nodes * 1_000_000
```

`Number of inspected nodes per attack`：

1. 对 TEST split 的 unique nodes 按 `node_score` 从高到低排序；
2. 对每个攻击 a，找到其真实恶意节点集合中排名最高的节点；
3. 该节点的 1-based rank 定义为 `inspected_nodes_for_attack[a]`；
4. 报告 per-attack 值以及跨攻击 mean/median。

该指标是排名/分析员检视负担指标，与分类阈值无关。

`Attack Detection Rate`：

```text
若某攻击的真实恶意节点中至少有一个节点在固定阈值下被预测为 anomalous，
则该攻击视为 detected。
ADR = detected_attacks / evaluable_attacks
```

THEIA_E3/E5 攻击场景数量较少，因此 ADR 只作辅助指标，不替代节点级 MCC/AUPRC/Recall/FPR。

效率指标：

```text
parameter_count
train_seconds_per_epoch
total_train_seconds
test_seconds
peak_gpu_memory_mb
peak_cpu_memory_mb
events_per_second
```

结果统一保存：

```text
metrics.json
node_predictions.csv
event_predictions.csv
runtime.json
```

论文正文只报告多 seed 的 `mean ± std`；`best` 仅可作为调试字段，不能用来挑最好 seed 作为论文主结果。

# 14. 阶段十：其他骨干

## 14.1 GraphTransformer

保留原 ORTHRUS GraphTransformer 作为主骨干：

```yaml
encoder:
  backbone: graph_transformer
```

## 14.2 GraphSAGE

新增：

```python
class GraphSAGEBackbone(nn.Module)
```

配置：

```yaml
encoder:
  backbone: graphsage
  num_layers: 2
```

GraphSAGE 不使用 edge features 时必须明确说明，但它必须能够接入与 GraphTransformer 相同的：

- multi-scale sampler；
- shared-backbone 三尺度编码；
- time task；
- calibration；
- node aggregation。

跨骨干专项仅在 THEIA_E3 运行，用来回答 MSTC 模块是否只对 GraphTransformer 有效。

## 14.3 Semantic MLP

新增简单基线：

```yaml
encoder:
  backbone: semantic_mlp
```

输入：

```python
concat(x_src, x_dst)
```

不读取历史图，不进入 `MultiScaleOrthrusEncoder`，不使用 gate。

预测边类型，可按配置启用 time head。MLP 只回答：

```text
复杂图历史建模是否明显优于简单语义映射？
```

因此不要为 Semantic MLP 调整多尺度 gate 输入维度；两条路径必须解耦。

# 15. 本篇论文的数据输入范围

本篇论文**不做 Host-only / Host + Network Structure / Host + Network Full 三视图消融**。

所有正式实验统一使用 THEIA 原始完整 provenance 输入范围，包括：

- subject；
- file；
- netflow；
- 原始关系结构；
- 在 `train_only` 语料协议下得到的语义特征。

因此不新增：

```text
host_only.yml
host_network_structure.yml
host_network_full.yml
src/mstc/dataset_views.py
```

这项删减是为了让本篇 SCI 小论文集中验证多尺度时间上下文、时间一致性学习、分层校准和 Top-k 聚合，不把“网络信息贡献”扩展成另一个独立研究问题。

未来若在硕士论文中接入 auditd + Zeek，可单独设计真正的主机—网络多源日志融合实验。

# 16. 本篇论文不做跨 E3/E5 迁移

本篇论文会分别在 THEIA_E3 和 THEIA_E5 上完成主结果，但**不做**：

```text
E3 训练 → E5 zero-shot
E5 训练 → E3 zero-shot
type_only cross-domain 技术验证
```

原因是 E3/E5 独立训练的 Word2Vec 向量空间不保证对齐，强行迁移会引入新的共享语义编码/domain adaptation 问题，超出本文三个核心创新点。

因此删除：

```yaml
cross_domain:
  feature_mode: type_only | shared_hashing | shared_encoder
```

以及对应实现/实验要求。

可在 Future Work 中说明：未来若研究跨 Engagement 泛化，应采用 feature hashing、共享冻结编码器或专门的 domain adaptation 方案。

# 17. 精简后的实验配置矩阵

总体原则：**两个数据集都做主实验；完整消融和专项分析主要集中在 THEIA_E3。**

## 17.1 主结果：THEIA_E3 + THEIA_E5

两个数据集都运行：

```text
Semantic MLP
GraphSAGE
ORTHRUS-ano
MSTC-PIDS Full
```

随机种子：

```text
0
1
2
```

报告：

```text
mean ± standard deviation
```

所有模型统一：

```yaml
semantic_features:
  corpus_scope: train_only
```

因此主结果共 `2 datasets × 4 models × 3 seeds = 24` 个模型运行，不再使用 5 seeds 扩大主矩阵。

## 17.2 完整消融：仅 THEIA_E3

运行 3 seeds：

```text
A0 ORTHRUS-ano
A1 Full w/o Multi-scale
A2 Full w/o Adaptive Gate
A3 Full w/o Time Prediction
A4 Full w/o Conditional Calibration
A5 Full w/o Top-k Aggregation
A6 Full
```

定义：

```text
w/o Multi-scale:
    使用 Recent-24，其他模块保持不变

w/o Gate:
    保留三尺度，对非空尺度等权平均

w/o Time Prediction:
    lambda_time = 0，score_raw = loss_type

w/o Calibration:
    保留同一 topk_mean 与 validation_quantile 决策规则，
    仅把 score_calibrated 替换为 score_raw，避免同时改变校准和阈值策略

w/o Top-k:
    保留同一 calibrated event score 和 validation_quantile，
    节点聚合由 topk_mean(K=5) 改为 mean
```

E5 不重复完整消融，E5 的作用是验证完整方法的跨数据集稳定性。

## 17.3 多尺度专项：仅 THEIA_E3，诊断性 1 seed

使用 seed 0：

```text
Recent-20
Recent-24
Single-window-24
Multi-scale Equal-24
Multi-scale Gated-24
```

由于 §17.2 已用 3 seeds 验证“有/无多尺度”和“有/无门控”，本专项主要解释具体时间组织方式，不再对每个变体重复 3–5 seeds。

报告：

- MCC / AUPRC / Recall / FPR；
- 参数量；
- 推理时间；
- 峰值显存；
- short/medium/long empty ratio；
- 仅对 Gated-24 报 gate weight distribution。

## 17.4 时间任务专项：仅 THEIA_E3，诊断性 1 seed

使用 seed 0：

```text
Type-only
Time-only
Joint
```

主模型预先固定：

```yaml
lambda_time: 0.3
```

仅做稳健性曲线：

```text
lambda_time ∈ {0.1, 0.3, 0.5, 1.0}
```

该曲线不用于根据 test 重新选择 lambda，也不要求把曲线最佳值替换主模型的 0.3。

## 17.5 校准与节点决策专项：仅 THEIA_E3

不要把不同层次的方法混成一个榜单，拆成两组。

### A. Score Calibration（3 seeds，可复用 raw event scores）

固定：`topk_mean(K=5) + validation_quantile(0.999)`。

比较：

```text
No Calibration (raw score)
Global Empirical p-value
Relation Triplet Calibration
Hierarchical Relation Calibration
```

### B. Node Decision / Threshold（3 seeds，可复用 calibrated event scores）

固定：Hierarchical Relation Calibration + `topk_mean(K=5)`。

比较：

```text
Validation Quantile 0.999
Max Validation
ORTHRUS K-means
```

这样每组实验只改变一层，避免把“校准”和“阈值”同时变化。

## 17.6 跨骨干：仅 THEIA_E3

3 seeds：

```text
GraphTransformer + MSTC modules
GraphSAGE + MSTC modules
```

Plain ORTHRUS-ano 和 plain GraphSAGE 已出现在主结果，不再另建重复专项运行。

该实验回答：MSTC 的多尺度/时间/校准模块是否能跨图骨干工作。

## 17.7 效率

不为了效率表重新训练。直接复用主结果和消融 run 的 `runtime.json`，汇总：

```text
ORTHRUS-ano
MSTC-PIDS Full
必要时增加 Full w/o Multi-scale 作为开销定位
```

报告 parameter_count、train time、test time、peak memory、events/s。

## 17.8 Top-k 参数敏感性：仅 THEIA_E3，纯后处理

使用主结果中 MSTC-PIDS Full 的 seeds：

```text
0, 1, 2
```

对每个 seed：

```text
K ∈ {1, 3, 5, 10, 20}
```

不重新训练，不新建 checkpoint，不重新拟合 calibrator。

至少报告：

```text
Recall
MCC
AUPRC
FPR
FP per million benign nodes
Number of inspected nodes per attack
Attack Detection Rate
```

同时输出 `num_events < K` 节点比例。

## 17.9 明确删除的实验

本篇论文不再运行：

```text
Host-only / Host + Network Structure / Host + Network Full
E3↔E5 zero-shot/type_only 迁移
CPU vs CUDA history_device 对比
Top-k Sum 对照
legacy_test_selection 对照
full-dataset Word2Vec vs train-only Word2Vec 对照
```

其中 `legacy_test_selection` 和 `official_full_dataset` 只作为代码兼容模式保留，不进入论文表格。

# 18. 案例分析输出

对每个被检测的高分事件保存：

```text
event_index
timestamp
source
destination
relation
raw_score
calibrated_score
type_loss
time_loss
source_time_gap
destination_time_gap
source_time_bucket
destination_time_bucket
short_gate_weight
medium_gate_weight
long_gate_weight
calibration_level
```

`calibration_level`：

```text
triplet
type_pair
global
```

生成两类案例图：

## 案例一：长时间依赖

展示：

- 早期历史事件；
- 中间正常活动；
- 后续攻击事件；
- 三个尺度的邻居；
- 门控权重。

## 案例二：时间节奏异常

展示：

- 边类型损失较低；
- 时间损失较高；
- 联合原始分数；
- 校准后分数；
- 最终节点排名。

---

# 19. 配置示例

创建：

```text
config/experiments/mstc_full.yml
```

建议结构：

```yaml
pipeline:
  stages:
    - train
    - test
    - evaluate
  run_tracing: false

model:
  variant: mstc
  seed: 0

semantic_features:
  corpus_scope: train_only

detection:
  gnn_training:
    num_epochs: 6
    lr: 0.00001
    weight_decay: 0.00001
    node_hid_dim: 128
    node_out_dim: 64

    encoder:
      backbone: graph_transformer
      batch_size: 512
      edge_features: edge_type
      temporal_dim: 100

      context:
        mode: multiscale
        recent_k: 24

        multiscale:
          enabled: true
          candidate_capacity: 64
          history_device: cpu
          scale_quantiles: [0.50, 0.90, 0.99]
          neighbor_budgets: [8, 8, 8]
          share_encoder: true
          fusion: gated
          use_scale_embedding: true

    decoder:
      edge_type:
        enabled: true
      time_gap:
        enabled: true
        include_no_history_class: true
        finite_bins: 5
        finite_quantiles: [0.20, 0.40, 0.60, 0.80]
        lambda_time: 0.30

calibration:
  method: hierarchical_relation
  min_triplet_samples: 100
  min_type_pair_samples: 200
  epsilon: 1.0e-12

node_aggregation:
  method: topk_mean
  topk: 5
  include_dst: true

node_threshold:
  method: validation_quantile
  quantile: 0.999

logging:
  wandb_mode: disabled
  save_event_scores: true
  save_node_scores: true
  save_gate_weights: true
  save_environment: true
```

Joint 模式的逐事件原始异常分数固定为：

```python
score_raw = loss_type + 0.30 * loss_time
```

不得由测试脚本自行改成其他定义。

# 20. 实验命令

统一入口：

```bash
python src/experiments/run_experiment.py \
  --dataset THEIA_E3 \
  --config config/experiments/mstc_full.yml \
  --seed 0 \
  --artifact-root "$ORTHRUS_ARTIFACT_ROOT"
```

主实验矩阵：

```bash
python src/experiments/run_matrix.py \
  --datasets THEIA_E3,THEIA_E5 \
  --configs \
    config/experiments/backbone_mlp.yml,\
    config/experiments/backbone_graphsage.yml,\
    config/experiments/baseline.yml,\
    config/experiments/mstc_full.yml \
  --seeds 0,1,2 \
  --artifact-root "$ORTHRUS_ARTIFACT_ROOT"
```

只训练：

```bash
python src/experiments/run_experiment.py \
  --dataset THEIA_E3 \
  --config config/experiments/mstc_full.yml \
  --seed 0 \
  --stages train
```

只测试和评估：

```bash
python src/experiments/run_experiment.py \
  --dataset THEIA_E3 \
  --config config/experiments/mstc_full.yml \
  --seed 0 \
  --stages test,evaluate \
  --checkpoint path/to/checkpoint
```

Top-k 敏感性使用专用后处理入口，不通过训练矩阵：

```bash
python src/experiments/run_topk_sensitivity.py \
  --dataset THEIA_E3 \
  --source-config mstc_full \
  --seeds 0,1,2 \
  --ks 1,3,5,10,20 \
  --artifact-root "$ORTHRUS_ARTIFACT_ROOT"
```

该命令必须读取已有 `mstc_full` validation/test 事件级校准结果；若源产物不存在则明确报错，不允许静默启动重新训练。

# 21. All-in-One Notebook 要求

本篇论文只维护一个正式 Notebook：

```text
notebooks/<现有 All-in-One Notebook 的实际文件名>.ipynb
```

Notebook 只负责环境准备、参数组织和调用 Python CLI，不在单元格里复制一套独立训练/评估实现。

建议固定 Section：

```text
Section 0   环境与依赖检查
Section 1   Artifact / 数据路径配置
Section 2   数据与缓存完整性检查
Section 3   Preprocess（可跳过）
Section 4   Baseline smoke test
Section 5   Main experiments：E3 + E5
Section 6   Ablation：E3 only
Section 7   Diagnostic studies：multi-scale / time / calibration / backbone
Section 8   Top-k sensitivity：postprocess only
Section 9   Evaluate-only / Recovery
Section 10  Result collection
Section 11  Paper artifact export
```

必须支持：

- Colab / PAI-DSW 中配置统一 artifact root；
- 已完成步骤自动检查并可跳过；
- checkpoint 恢复；
- evaluate-only；
- 结果汇总；
- 输出 `config_resolved.yml` 与环境信息；
- Section 10/11 至少汇总 `main_results.csv`、`ablation_results.csv`、`score_calibration_results.csv`、`node_decision_results.csv`、`efficiency_results.csv`、`topk_sensitivity_results.csv` 和 `topk_event_count_summary.csv`。

沿用仓库现有 All-in-One Notebook 的实际文件名，不因为本次文档修订再复制或重命名出第二个“官方 Notebook”。

## OOM 处理

正式论文实验不得自动改变：

```text
candidate_capacity
neighbor budgets
node dimensions
model layers
```

这些会改变实验语义。

只有在确认数学语义不变时允许统一调整 `batch_size`。其他 OOM 必须让当前 run 失败并记录原因，再人工确定一个对所有对照一致的资源配置后重跑。

Notebook 不得静默“为了跑通”修改实验语义。

# 22. 单元测试要求

## 22.1 时间统计与目标

验证：

- 第一次出现为 NO_HISTORY；
- 第二次出现 exact gap 正确；
- 同一 batch 内重复节点按 `(timestamp, global_event_index)` 逐事件计算 target；
- target state 可在 batch 内更新，但不会进入模型输入；
- 纳秒正确转换为秒；
- `log1p` 正确；
- scale quantile 使用 seconds；
- time-bucket quantile 使用 log1p_seconds；
- 负时间差报错。

## 22.2 多尺度采样

构造人工历史，验证：

```text
short: delta <= Q50
medium: Q50 < delta <= Q90
long: delta > Q90
```

额外验证 `delta > Q99` 仍属于 long，不被静默丢弃。

## 22.3 无未来泄漏

分别验证两类状态：

1. Encoder HistoryStore：query 前当前 batch 不在历史，batch 末才 insert；重复节点 reference_time 使用 batch 内最小时间。
2. TimeGapTargetBuilder：可以逐事件更新监督 last_seen，但 target 不进入模型输入。

两套状态不得引用同一个可变对象。

## 22.4 参数共享与跨骨干

验证三个尺度调用同一个 `shared_graph_encoder` 实例；分别在 GraphTransformer 和 GraphSAGE 下运行。

## 22.5 门控

验证：

- 权重和为1；
- 空尺度权重为0；
- 全空时正常回退；
- 无 NaN。

## 22.6 校准

验证：

- 分数越高，经验 p 值不增；
- 支持数不足正确回退；
- add-one 正确；
- leave-one-out 正确；
- 空组回退 global；
- p 值不为0；
- calibration runner 是保存 calibrator 的唯一 I/O owner。

## 22.7 Top-k

验证：

- 少于 K 时平均全部；
- 多于 K 时只取最高 K；
- `include_dst` 在 event→node 映射阶段生效；
- K∈{1,3,5,10,20} 都可后处理；
- 改 K 不改 event-level input；
- 每个 K 单独重算 validation threshold；
- 生成 `topk_event_count_summary.csv`。

## 22.8 Checkpoint 与评估产物

training checkpoint 验证：

- model parameters；
- optimizer；
- epoch；
- Python/NumPy/Torch RNG；
- config hash；
- scheduler（若有）。

calibrator、threshold、node predictions 属于 evaluation artifacts，不要求塞进 training checkpoint。

history 使用 replay 模式时验证重放结果一致。

## 22.9 基线兼容

关闭全部 MSTC 新模块后，在固定人工数据上逐 tensor 比较原始代码和新代码：

```python
torch.testing.assert_close(..., atol=1e-6, rtol=1e-5)
```

不要再写“state_dict hash 使用 atol/rtol”；若使用 hash，只能要求 exact equality。

# 23. 验收标准

整个项目完成后必须满足：

1. 原 ORTHRUS baseline 仍可运行；
2. THEIA_E3 和 THEIA_E5 都能完成论文主结果流程；
3. 主结果统一使用 `semantic_features.corpus_scope=train_only`；
4. `official_full_dataset` 仅用于兼容复现并带 compatibility-only 标记；
5. 已有 artifacts 时 train/test/evaluate 不强制需要 PostgreSQL；
6. 时间 target 与 encoder history 使用分离状态，均无未来输入泄漏；
7. `score_raw` 在所有代码路径中遵循 §9.2 唯一定义；
8. 完整模型可在 GraphTransformer 上运行；
9. MSTC 模块可在 GraphSAGE 上运行；
10. Semantic MLP 不进入多尺度/门控路径；
11. 时间分位数仅由训练集生成，数据域明确；
12. 校准分布仅由正常 validation 建立；
13. calibration 与 threshold 对照实验分层，不混为同一变量；
14. Top-k 敏感性只做后处理，不创建新的训练 checkpoint；
15. 每个 K 根据自己的 validation node-score 重算 threshold；
16. 输出 `topk_sensitivity_results.csv` 和 `topk_event_count_summary.csv`；
17. 输出事件级和节点级结果及完整 runtime/environment；
18. All-in-One Notebook 是唯一正式 Notebook 入口；
19. 正式 OOM 恢复不得静默改变 candidate capacity / neighbor budget / model dimension；
20. 主结果和完整消融均可汇总 3 seeds 的 mean±std；
21. 不运行 Host-network 三视图、跨 E3/E5 zero-shot、history device 对比、Top-k Sum 对照；
22. 不运行攻击重建也不会报错；
23. `--run_from_training` 不会报未定义变量；
24. 所有测试通过；
25. README 与两份设计文档同步更新。

# 24. 开发顺序

仍保持 8 个逻辑阶段，避免扩大 Git 历史；若代码已经完成，则把下列内容作为“文档/实现一致性修复项”，不要求重写历史 commit。

## Commit 1：基线保护

```text
流水线 stages
关闭 tracing
修复 run_from_training
canonical artifact path
validation-based model selection
legacy_test_selection 仅兼容，默认关闭并标记 NOT FOR PAPER
```

## Commit 2：Colab、元数据缓存与 train-only 语义语料

```text
环境变量路径
数据库缓存/回退
train_only Word2Vec corpus 真正生效
official_full_dataset 仅 compatibility-only
All-in-One 所需 artifact 检查接口
```

## Commit 3：时间统计与精确时间 target

```text
显式 src/dst 类型与 global event id
scale quantiles（seconds）
time buckets（log1p_seconds）
TimeGapTargetBuilder exact per-event target
```

## Commit 4：双任务预测

```text
MSTCOrthrus wrapper
TimeGapDecoder
联合损失
score_raw 唯一公式
详细事件输出
```

## Commit 5：多尺度采样和跨骨干共享编码

```text
HistoryStore：timestamp int64
MultiScaleNeighborLoader
Q50/Q90 三尺度，Q99 不截断 long
shared configurable graph backbone
门控融合
history replay
```

## Commit 6：条件校准和节点聚合

```text
HierarchicalRelationCalibrator
calibration_runner 独占 I/O
经验 p 值 + leave-one-out
NodeScoreAggregator
mean/max/topk_mean 论文对照
validation threshold
```

本 Commit 不再实现 DatasetViews。

## Commit 7：GraphSAGE 和 Semantic MLP

```text
GraphSAGE 接入 shared multi-scale backbone
Semantic MLP 独立非图路径
跨骨干兼容测试
```

## Commit 8：精简实验矩阵 + All-in-One + 后处理分析

```text
主结果：E3+E5，3 seeds
完整消融：E3，3 seeds
专项：E3
run_topk_sensitivity.py
结果汇总与 topk_event_count_summary
单一 All-in-One Notebook
paper artifacts 导出
```

每个阶段完成后运行：

```bash
pytest -q
```

并做对应 smoke test。

# 25. Cursor最终交付内容

完成后输出：

1. 修改文件列表；
2. 新增文件列表；
3. 每个模块的实现说明；
4. 所有配置字段说明；
5. 单元测试结果；
6. THEIA_E3 smoke test命令；
7. THEIA_E5 smoke test命令；
8. All-in-One Notebook Section 运行顺序；
9. 已知限制；
10. 尚未实现的内容；
11. 从零到完整实验的命令清单；
12. 论文消融实验与配置文件的对应关系。

不要只给伪代码。请实际修改代码、生成配置、测试和Notebook。

如果完整数据不可用，使用合成TemporalData完成所有单元测试和smoke test，确保代码逻辑可运行。
