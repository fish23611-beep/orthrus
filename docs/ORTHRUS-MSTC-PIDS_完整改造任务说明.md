# ORTHRUS-MSTC-PIDS 完整改造任务说明

## 1. 项目目标

基于官方 ORTHRUS 仓库实现一个研究分支，暂定名称：

**MSTC-PIDS: Multi-Scale Temporal Consistency Learning with Hierarchical Relation-Conditional Calibration**

研究范围仅包括：

- THEIA_E3；
- THEIA_E5；
- 异常事件评分；
- 异常节点检测；
- 节点异常排名；
- 多尺度时间上下文；
- 边类型与时间间隔双任务预测；
- 分层关系条件校准；
- GraphTransformer与GraphSAGE跨骨干验证；
- 主机侧和主机—网络关联消融；
- Colab训练、测试和结果汇总。

暂不实现：

- auditd日志；
- Zeek日志；
- 自建靶场；
- 日志缺失实验；
- 攻击路径重建；
- CALDERA；
- 跨采集平台实验；
- LLM；
- 对比学习；
- 复杂图增强。

主要比较对象是：

1. 原始ORTHRUS-ano；
2. Semantic MLP；
3. GraphSAGE；
4. 完整MSTC-PIDS。

---

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
- 使用测试集语料训练Word2Vec；
- 在计算当前事件表示前插入当前事件；
- 把当前事件真实边类型作为当前事件输入特征；
- 把当前事件真实时间桶直接作为输入特征。

所有时间分位数、时间桶和超参数只能来自：

- 正常训练集；
- 正常验证集；
- 预先固定的配置。

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

新增：

```text
src/mstc/
    __init__.py
    history_store.py
    multiscale_sampler.py
    multiscale_encoder.py
    time_gap.py
    calibration.py
    aggregation.py
    metrics.py
    metadata_cache.py
    experiment_utils.py

src/experiments/
    __init__.py
    run_experiment.py
    run_matrix.py
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
    calibration_max.yml
    calibration_quantile.yml
    calibration_kmeans.yml
    calibration_global_p.yml
    calibration_relation.yml
    calibration_hierarchical.yml
    backbone_graphtransformer.yml
    backbone_graphsage.yml
    backbone_mlp.yml
    host_only.yml
    host_network_structure.yml
    host_network_full.yml

notebooks/
    00_colab_environment.ipynb
    01_preprocess_theia.ipynb
    02_baseline_smoke_test.ipynb
    03_train_main_models.ipynb
    04_run_ablations.ipynb
    05_collect_results.ipynb

tests/
    test_time_gap.py
    test_multiscale_sampler.py
    test_calibration.py
    test_aggregation.py
    test_no_future_leakage.py
    test_checkpoint_resume.py
    test_baseline_compatibility.py
```

---

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

输出目录结构：

```text
artifacts/
  THEIA_E3/
    preprocessing/
    metadata/
    runs/
      orthrus_baseline/
        seed_0/
          config_resolved.yml
          environment.json
          checkpoints/
          edge_scores/
          node_scores/
          metrics.json
          runtime.json
      mstc_full/
        seed_0/
          ...
```

每次运行保存：

- 数据集；
- 模型名；
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

---

# 6. 阶段二：Colab适配与数据库解耦

## 6.1 路径使用环境变量

修改`src/config.py`，支持：

```bash
export ORTHRUS_ARTIFACT_ROOT=/content/drive/MyDrive/mstc_pids/artifacts
export ORTHRUS_DATA_ROOT=/content/data
export ORTHRUS_DB_HOST=127.0.0.1
export ORTHRUS_DB_PORT=5432
export ORTHRUS_DB_USER=postgres
export ORTHRUS_DB_PASSWORD=postgres
```

代码中使用：

```python
os.environ.get(...)
```

保留原默认值作为回退。

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

`node_metadata.pkl`结构：

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

`dataset_manifest.json`包含：

```json
{
  "dataset": "THEIA_E3",
  "num_node_types": 3,
  "num_edge_types": 10,
  "train_files": [],
  "val_files": [],
  "test_files": [],
  "word2vec_dim": 128,
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
直接读取缓存，不连接PostgreSQL

元数据缓存不存在
    ↓
回退到原始PostgreSQL查询
```

训练、测试和评估阶段在已有预处理缓存时不得强制要求PostgreSQL。

`orthrus_gnn_testing.py`中的`srcmsg`和`dstmsg`仅用于展示，不应成为测试必须项。

允许：

```yaml
testing:
  include_node_messages: false
```

当关闭时，只输出节点ID、类型、关系和分数。

## 6.4 Colab运行模式

创建两个模式：

```text
full_pipeline
detection_only
```

`detection_only`只需要：

- `TemporalData`文件；
- 标签缓存；
- 元数据缓存；
- 配置文件；
- 模型代码。

不需要：

- PostgreSQL；
- 数据库dump；
- 图构建；
- Word2Vec重新训练。

---

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
```

## 8.1 计算实体时间间隔

按训练数据时间顺序扫描。

对于当前事件：

```text
e_t = (u, v, r, t)
```

计算：

```python
delta_src = t - last_seen[src]
delta_dst = t - last_seen[dst]
```

未见过的节点标记为：

```text
NO_HISTORY
```

时间戳转换：

```python
delta_seconds = delta_ns / 1_000_000_000
z = log1p(delta_seconds)
```

禁止直接对纳秒值做分桶。

## 8.2 多尺度边界

仅使用正常训练集有限时间间隔计算：

```python
tau_short = Q50
tau_medium = Q90
tau_max = Q99
```

保存：

```text
metadata/time_statistics.json
```

例如：

```json
{
  "unit": "seconds",
  "transform": "log1p",
  "scale_quantiles": [0.5, 0.9, 0.99],
  "scale_boundaries": [0.0, 0.0, 0.0],
  "time_bucket_quantiles": [0.2, 0.4, 0.6, 0.8],
  "time_bucket_boundaries": [0.0, 0.0, 0.0, 0.0]
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

有限时间间隔按照训练集：

```text
Q20
Q40
Q60
Q80
```

划分五个桶。

时间桶边界只能由训练集生成，验证和测试直接加载。

## 8.4 数据检查

必须处理：

- 重复时间戳；
- 非递增时间戳；
- 负时间间隔；
- 节点第一次出现；
- 极大时间间隔；
- 空训练数据。

负间隔应抛出明确错误或记录异常，不能默默取绝对值。

---

# 9. 阶段五：双任务一致性学习

## 9.1 重构解码器接口

修改`src/decoders.py`。

保留：

```python
class EdgeTypeDecoder
```

新增：

```python
class TimeGapDecoder
```

建议接口：

```python
class EdgeTypeDecoder(nn.Module):
    def logits(self, h_src, h_dst):
        ...

    def loss(self, logits, target, reduction):
        ...
```

```python
class TimeGapDecoder(nn.Module):
    def forward(self, h_src, h_dst):
        return src_logits, dst_logits
```

时间头采用：

```text
共享隐藏层
    ↓
src时间桶分类头
dst时间桶分类头
```

不要把真实时间桶作为输入。

## 9.2 损失

边类型损失：

```python
loss_type = cross_entropy(edge_logits, edge_type_target)
```

时间损失：

```python
loss_time_src = cross_entropy(src_time_logits, src_time_target)
loss_time_dst = cross_entropy(dst_time_logits, dst_time_target)

loss_time = 0.5 * (loss_time_src + loss_time_dst)
```

总损失：

```python
loss_total = loss_type + lambda_time * loss_time
```

默认：

```yaml
lambda_time: 0.3
```

必须可配置。

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

## 9.4 测试CSV字段

测试输出至少包含：

```text
event_index
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

不要只保留一个总loss。

---

# 10. 阶段六：数据驱动多尺度时间邻域

## 10.1 新历史加载器

新增：

```text
src/mstc/history_store.py
src/mstc/multiscale_sampler.py
```

实现：

```python
class MultiScaleNeighborLoader
```

不要直接破坏原`LastNeighborLoader`。

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

候选历史每个节点最多保存：

```text
candidate_capacity
```

建议内部存储尽量使用：

```text
int32
```

取出用于PyTorch索引时转换为：

```text
int64
```

避免为几十万节点分配过大的int64历史矩阵。

## 10.2 查询顺序

对于每个批次：

```text
1. 根据当前批次计算查询时间
2. 查询历史
3. 构建三个尺度子图
4. 编码当前批次
5. 计算损失
6. 最后插入当前批次事件
```

绝对禁止：

```text
先插入当前事件，再查询历史
```

## 10.3 批次中同一节点的参考时间

一个节点可能在当前批次多次出现。

为避免未来泄漏，对该节点使用当前批次中的：

```python
reference_time[node] = min(current_event_times_of_node)
```

这样构建的上下文对该批次中的所有事件都是因果的。

记录这一设计，并在论文中称为：

```text
causal micro-batch temporal context
```

不要声称批次中后面的事件能够看到同批次前面的事件，除非代码确实逐事件更新。

## 10.4 时间分组

对历史边：

```python
delta = current_reference_time - historical_edge_time
```

划分：

```python
short:
    0 < delta <= tau_short

medium:
    tau_short < delta <= tau_medium

long:
    tau_medium < delta <= tau_max
```

每个尺度只保留该范围内最近的K条边。

如果某尺度不足K条，不进行重复填充。

如果某尺度为空，必须提供显式mask。

## 10.5 公平邻居预算

完整模型默认总预算：

```text
8 + 8 + 8 = 24
```

专项比较包含：

```text
Recent-20
Recent-24
Single-window-24
Multi-scale Equal-24
Multi-scale Gated-24
```

比较Multi-scale和Recent-24时，必须使用相同的最大历史边数量。

## 10.6 多尺度编码器

新增：

```python
class MultiScaleOrthrusEncoder(nn.Module)
```

三个尺度共享同一个GraphTransformer实例：

```python
self.shared_graph_encoder
```

禁止直接创建三个互不共享参数的GraphTransformer，除非专门做额外对照。

每个尺度可加入可学习尺度嵌入：

```python
scale_embedding[SHORT]
scale_embedding[MEDIUM]
scale_embedding[LONG]
```

处理流程：

```text
短期子图 → 共享GraphTransformer → h_src_short, h_dst_short
中期子图 → 共享GraphTransformer → h_src_medium, h_dst_medium
长期子图 → 共享GraphTransformer → h_src_long, h_dst_long
```

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
h_src = (
    beta_short * h_src_short
    + beta_medium * h_src_medium
    + beta_long * h_src_long
)

h_dst = (
    beta_short * h_dst_short
    + beta_medium * h_dst_medium
    + beta_long * h_dst_long
)
```

要求：

```python
beta.sum(dim=-1) == 1
```

空尺度不得获得权重。

三个尺度都为空时，回退到当前节点投影：

```python
h_src = current_src_projection
h_dst = current_dst_projection
```

## 10.8 历史状态

新增接口：

```python
reset_state()
warmup(data)
state_dict()
load_state_dict()
insert(...)
query(...)
```

默认推荐：

```yaml
history:
  checkpoint_mode: replay
```

测试前：

```text
加载模型权重
    ↓
重置历史
    ↓
重放训练集，只构建历史，不计算梯度
    ↓
依次处理验证集和测试集
```

这样避免每个epoch保存巨大的历史矩阵。

保留：

```yaml
history:
  checkpoint_mode: save
```

作为可选模式。

## 10.9 历史设备

默认：

```yaml
history_device: cpu
```

历史索引保存在CPU，选出的尺度子图再移动到GPU。

允许：

```yaml
history_device: cuda
```

用于显存充足环境。

---

# 11. 阶段七：分层关系条件校准

新增：

```text
src/mstc/calibration.py
```

实现：

```python
class HierarchicalRelationCalibrator
```

## 11.1 条件组

一级条件：

```python
c1 = (src_type, edge_type, dst_type)
```

二级条件：

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
三元组样本足够 → 使用三元组
否则类型对足够 → 使用类型对
否则 → 使用全局分布
```

## 11.2 经验p值

验证集建立正常参考分布。

对测试事件原始分数：

```python
p = (
    1
    + count(reference_scores >= test_score)
) / (
    n_reference + 1
)
```

最终事件分数：

```python
score_calibrated = -log(max(p, epsilon))
```

实现时对每个条件组排序，并使用：

```python
numpy.searchsorted
```

避免逐事件遍历全部验证分数。

## 11.3 验证集自身评分

使用验证集确定节点阈值时，必须使用：

```text
leave-one-out empirical p-value
```

避免当前验证事件把自己当作参考样本造成偏差。

## 11.4 保存校准器

保存：

```text
calibrator.pkl
calibration_summary.json
```

摘要包含：

- 每个三元组样本数；
- 每个类型对样本数；
- 回退比例；
- 全局样本数；
- 每个组的分数分位数；
- 测试事件使用三元组、类型对和全局分布的比例。

---

# 12. 阶段八：节点Top-k聚合

新增：

```text
src/mstc/aggregation.py
```

实现：

```python
class NodeScoreAggregator
```

默认：

```yaml
node_aggregation:
  method: topk_mean
  topk: 5
  include_dst: true
```

对于节点：

```python
node_score = mean(top_k(event_calibrated_scores))
```

如果事件数少于K：

```python
node_score = mean(all_event_scores)
```

支持以下对照方法：

```text
max
mean
topk_mean
topk_sum
```

默认只使用：

```text
topk_mean
```

## 12.1 节点阈值

使用正常验证集节点分数确定阈值：

```yaml
node_threshold:
  method: validation_quantile
  quantile: 0.999
```

同时保留：

```text
max_validation
kmeans
```

用于对比。

禁止在测试节点标签上搜索最佳阈值。

---

# 13. 阶段九：评估指标

新增或补充：

```text
src/mstc/metrics.py
```

主指标：

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

其中：

```python
fp_per_million = FP / num_benign_nodes * 1_000_000
```

Attack Detection Rate：

```text
某个攻击对应的真实恶意节点中
至少有一个被检测到
则该攻击视为被检测
```

由于THEIA_E3只有少量攻击场景、THEIA_E5更少，Attack Detection Rate只作为辅助指标，不替代节点级指标。

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

结果统一保存为：

```text
metrics.json
node_predictions.csv
event_predictions.csv
runtime.json
```

---

# 14. 阶段十：其他骨干

## 14.1 GraphTransformer

保留原ORTHRUS GraphTransformer作为主骨干：

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

GraphSAGE不使用边特征时，应在文档中明确说明。

比较：

```text
GraphSAGE
GraphSAGE + Multi-scale + Time-task + Calibration

GraphTransformer
GraphTransformer + Multi-scale + Time-task + Calibration
```

跨骨干实验至少在THEIA_E3上运行。

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

不读取历史图。

预测：

- 边类型；
- 可选时间桶。

MLP用于回答：

```text
复杂图模型是否明显优于简单语义模型？
```

---

# 15. 阶段十一：主机—网络关联数据视图

增加：

```yaml
dataset_view:
  mode: full
```

合法值：

```text
host_only
host_network_structure
host_network_full
```

## 15.1 Host-only

删除：

```text
src或dst为netflow的事件
```

只保留：

```text
subject
file
```

## 15.2 Host + Network Structure

保留netflow节点和关系，但将netflow语义嵌入置零。

保留：

- netflow节点类型；
- 网络边结构；
- 边类型。

不使用：

- IP语义；
- 端口语义；
- netflow Word2Vec语义。

## 15.3 Host + Network Full

使用完整原始THEIA输入。

必须确保三个数据视图：

- 使用相同训练、验证和测试日期；
- 使用相同模型配置；
- 使用相同随机种子；
- 仅改变网络相关输入。

---

# 16. 暂不直接实现的跨E3/E5迁移

当前Word2Vec由每个数据集自己的语料分别训练。

即使：

```text
E3嵌入维度 = 128
E5嵌入维度 = 128
```

两个向量空间也不保证对齐。

因此禁止直接做：

```text
E3模型权重
    +
E5独立训练的Word2Vec特征
```

并把结果称为Zero-shot。

第一版只预留接口：

```yaml
cross_domain:
  feature_mode: type_only | shared_hashing | shared_encoder
```

本阶段只需要实现：

```text
type_only
```

作为技术验证，不纳入主结果。

以后若做正式跨Engagement实验，应选择：

1. 节点类型特征；
2. 固定Feature Hashing；
3. E3训练、可处理OOV的共享文本编码器；
4. 预训练且冻结的统一语义编码器。

---

# 17. 实验配置矩阵

## 17.1 主结果

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
3
4
```

报告：

```text
mean ± standard deviation
```

## 17.2 完整消融

两个数据集运行：

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
    使用Recent-24

w/o Gate:
    对非空尺度等权平均

w/o Time Prediction:
    lambda_time = 0

w/o Calibration:
    使用原始验证阈值

w/o Top-k:
    所有事件分数取mean
```

主消融至少3个种子，最终结果稳定后建议补齐5个种子。

## 17.3 多尺度专项

```text
Recent-20
Recent-24
Single-window-24
Multi-scale Equal-24
Multi-scale Gated-24
```

必须同时报告：

- 检测指标；
- 参数量；
- 推理时间；
- 峰值显存；
- 门控权重分布。

## 17.4 时间任务专项

```text
Type-only
Time-only
Joint
```

测试：

```python
lambda_time in [0.1, 0.3, 0.5, 1.0]
```

只在THEIA_E3完成参数敏感性。

根据正常验证损失和正常验证误报倾向选择固定值，然后把该值直接用于THEIA_E5。

禁止根据THEIA_E5测试标签重新选择。

## 17.5 校准专项

```text
Max Validation Loss
Global Quantile
ORTHRUS K-means
Global Empirical p-value
Relation Triplet Calibration
Hierarchical Relation Calibration
```

重点比较：

```text
Precision
MCC
FP
FP per million
Attack Detection Rate
```

## 17.6 主机—网络专项

```text
Host-only
Host + Network Structure
Host + Network Full
```

## 17.7 跨骨干

至少在THEIA_E3运行：

```text
GraphSAGE
GraphSAGE + MSTC modules
GraphTransformer
GraphTransformer + MSTC modules
```

## 17.8 效率

比较：

```text
ORTHRUS-ano
ORTHRUS + Multi-scale
ORTHRUS + Multi-scale + Time-task
MSTC-PIDS Full
```

---

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

dataset_view:
  mode: host_network_full

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
          scale_quantiles:
            - 0.50
            - 0.90
            - 0.99
          neighbor_budgets:
            - 8
            - 8
            - 8
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
        finite_quantiles:
          - 0.20
          - 0.40
          - 0.60
          - 0.80
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

---

# 20. 实验命令

增加统一入口：

```bash
python src/experiments/run_experiment.py \
  --dataset THEIA_E3 \
  --config config/experiments/mstc_full.yml \
  --seed 0 \
  --artifact-root "$ORTHRUS_ARTIFACT_ROOT"
```

基线：

```bash
python src/experiments/run_experiment.py \
  --dataset THEIA_E3 \
  --config config/experiments/baseline.yml \
  --seed 0 \
  --artifact-root "$ORTHRUS_ARTIFACT_ROOT"
```

矩阵运行：

```bash
python src/experiments/run_matrix.py \
  --datasets THEIA_E3,THEIA_E5 \
  --configs \
    config/experiments/baseline.yml,\
    config/experiments/mstc_full.yml \
  --seeds 0,1,2,3,4
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

---

# 21. Colab Notebook要求

## 21.1 `00_colab_environment.ipynb`

完成：

```text
挂载Google Drive
打印Python版本
打印PyTorch版本
打印CUDA版本
执行nvidia-smi
检查磁盘空间
安装依赖
设置环境变量
克隆或更新仓库
运行import smoke test
```

不要盲目强制安装与当前Colab不兼容的旧CUDA wheel。

如果保留Colab预装PyTorch，则安装与其兼容的PyG版本。

记录：

```bash
pip freeze
```

保存到运行目录。

## 21.2 `01_preprocess_theia.ipynb`

参数：

```python
DATASET = "THEIA_E3"
```

功能：

- 启动本地PostgreSQL；
- 恢复一个数据库；
- 运行预处理；
- 导出metadata；
- 验证artifacts；
- 复制到Drive；
- 可选删除本地数据库；
- 支持断点检查。

E3和E5一次只处理一个。

## 21.3 `02_baseline_smoke_test.ipynb`

使用：

- 少量时间窗口；
- 1个epoch；
- 1个seed；
- 不运行攻击重建。

验证完整训练、测试和评估链路。

## 21.4 `03_train_main_models.ipynb`

运行：

```text
ORTHRUS-ano
MSTC-PIDS Full
```

支持：

- 断点恢复；
- Drive结果保存；
- 自动跳过已完成任务；
- 运行结束后压缩结果。

## 21.5 `04_run_ablations.ipynb`

读取实验配置列表逐个运行。

发生OOM时允许自动降低：

```text
batch_size
candidate_capacity
neighbor budget
```

但必须把实际配置写入结果文件，不能悄悄变化。

## 21.6 `05_collect_results.ipynb`

输出：

```text
results/main_results.csv
results/ablation_results.csv
results/calibration_results.csv
results/efficiency_results.csv
results/all_runs.csv
```

生成：

```text
mean
standard deviation
best
median
number of successful seeds
number of failed seeds
```

---

# 22. 单元测试要求

## 22.1 时间目标

验证：

- 第一次出现为NO_HISTORY；
- 第二次出现时间差正确；
- 纳秒正确转换为秒；
- log1p正确；
- 分位数确定；
- 边界值所属桶稳定；
- 负时间差报错。

## 22.2 多尺度采样

构造人工事件：

```text
t=10
t=20
t=50
t=100
当前t=110
```

验证短、中、长期分配准确。

## 22.3 无未来泄漏

验证当前事件在查询前不在历史中。

验证批次中参考时间使用最小时间。

## 22.4 参数共享

验证：

```python
short_encoder is medium_encoder
medium_encoder is long_encoder
```

或者三个尺度调用同一模块。

## 22.5 门控

验证：

- 权重和为1；
- 空尺度权重为0；
- 全空时正常回退；
- 无NaN。

## 22.6 校准

验证：

- 分数越高，经验p值不增；
- 样本不足正确回退；
- add-one计算正确；
- leave-one-out正确；
- 空组回退全局；
- p值不为0。

## 22.7 Top-k

验证：

- 少于K时平均全部；
- 多于K时只取最高K；
- `include_dst`开关有效。

## 22.8 Checkpoint

验证保存后加载：

- 模型参数一致；
- 时间统计一致；
- 校准器一致；
- 历史重放结果一致。

## 22.9 基线兼容

关闭全部新模块后，在固定人工数据上比较原始代码和新代码：

```text
edge loss
node score
预测结果
```

允许的数值误差：

```python
atol = 1e-6
rtol = 1e-5
```

---

# 23. 验收标准

整个项目完成后必须满足：

1. 原ORTHRUS基线仍可运行；
2. THEIA_E3可以完成预处理、训练、测试和评估；
3. THEIA_E5可以完成相同流程；
4. 已有artifacts时训练和评估不需要PostgreSQL；
5. 每个创新模块可独立开关；
6. 完整模型可在GraphTransformer上运行；
7. 完整模块可在GraphSAGE上运行；
8. 所有测试通过；
9. 测试集标签仅在最终评估函数中读取；
10. 时间分位数仅由训练集生成；
11. 校准分布仅由正常验证集生成；
12. 输出事件级和节点级结果；
13. 输出完整配置和环境信息；
14. Colab断线后可从checkpoint继续；
15. 所有实验可由配置文件复现；
16. 不运行攻击重建也不会报错；
17. `--run_from_training`路径不会报未定义变量；
18. 结果收集脚本能汇总多个seed；
19. 代码有类型提示和必要注释；
20. README新增完整使用说明。

---

# 24. 开发顺序

严格按以下顺序执行：

## Commit 1：基线保护

```text
流水线阶段开关
关闭tracing
修复run_from_training
输出目录整理
基线smoke test
```

## Commit 2：Colab和元数据缓存

```text
环境变量路径
数据库缓存
数据库回退
detection-only模式
```

## Commit 3：时间统计

```text
显式src/dst类型
时间间隔
时间桶
训练集分位数
```

## Commit 4：双任务预测

```text
TimeGapDecoder
联合损失
详细测试输出
```

## Commit 5：多尺度采样和编码

```text
MultiScaleNeighborLoader
共享编码器
门控融合
历史状态
```

## Commit 6：条件校准和Top-k

```text
分层校准
经验p值
leave-one-out
节点聚合
```

## Commit 7：GraphSAGE和MLP

```text
骨干工厂
跨骨干实验
简单基线
```

## Commit 8：实验矩阵和Notebook

```text
配置文件
Colab notebooks
结果汇总
表格导出
```

每个Commit完成后运行：

```bash
pytest -q
```

并做小规模smoke test。

---

# 25. Cursor最终交付内容

完成后输出：

1. 修改文件列表；
2. 新增文件列表；
3. 每个模块的实现说明；
4. 所有配置字段说明；
5. 单元测试结果；
6. THEIA_E3 smoke test命令；
7. THEIA_E5 smoke test命令；
8. Colab运行顺序；
9. 已知限制；
10. 尚未实现的内容；
11. 从零到完整实验的命令清单；
12. 论文消融实验与配置文件的对应关系。

不要只给伪代码。请实际修改代码、生成配置、测试和Notebook。

如果完整数据不可用，使用合成TemporalData完成所有单元测试和smoke test，确保代码逻辑可运行。
