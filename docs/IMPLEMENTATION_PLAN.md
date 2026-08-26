# ORTHRUS 实施规划 (IMPLEMENTATION_PLAN)

> 严格遵循任务说明 §24「按 8 个独立 Commit 提交」的开发顺序。本规划仅为最小改动方案，
> 不进行大规模重构，不修改基线组件（`LastNeighborLoader` / `OrthrusEncoder` / `EdgeTypeDecoder` /
> 阈值方法 / KMeans）的对外行为。
>
> 保留原始 `Orthrus` 类不变；新增 `MSTCOrthrus`（或 `MultiTaskOrthrus`）由 `model.variant` 选择。
>
> 用户引用的 `docs/ORTHRUS_MSTC_IMPLEMENTATION_SPEC.md` 不存在；本规划以仓库内
> `docs/ORTHRUS-MSTC-PIDS_完整改造任务说明.md` 为准。

## 0. 总览：修订后的 8 个逻辑阶段

本规划已按最终论文减负方案统一：

- 主结果：THEIA_E3 + THEIA_E5，4 个模型，3 seeds；
- 完整消融：THEIA_E3，3 seeds；
- 专项分析主要只在 THEIA_E3；
- 删除 Host-network 三视图、跨 E3/E5 zero-shot/type_only、history device 对比、Top-k Sum 论文实验；
- `official_full_dataset` 与 `legacy_test_selection` 只保留兼容能力，不进入论文结果；
- 6 个 Notebook 收敛为一个 All-in-One Notebook；
- Top-k K 敏感性改成纯后处理，不再制造新的训练 config-id/checkpoint。

依赖：

```text
C1 Baseline/路径合同
  ↓
C2 Metadata + train-only semantic corpus
  ↓
C3 Exact time targets + time statistics
  ↓
C4 MSTC dual-task + score_raw contract
  ↓
C5 Multi-scale shared graph backbone
  ↓
C6 Calibration + node aggregation
  ↓
C7 GraphSAGE + Semantic MLP
  ↓
C8 Reduced matrix + All-in-One + analysis/export
```

每个阶段必须满足：

- 不破坏 `Orthrus / OrthrusEncoder / LastNeighborLoader / EdgeTypeDecoder` 的 baseline 对外行为；
- 关闭 MSTC 时使用逐 tensor `assert_close(atol=1e-6, rtol=1e-5)` 验证数值兼容；
- 提交前 `pytest -q`；
- 正式论文路径不得读取 test labels 做 model/threshold/hyperparameter selection。

---

## Commit 1 — Baseline Protection + Canonical Artifact Contract

### 目标

保护原 ORTHRUS，修正阶段控制、test-based epoch selection 和路径合同。

### 修改文件

- `src/orthrus.py`
  - `--stages preprocess,train,test,evaluate`；
  - `--skip-tracing`；
  - 修复 `run_from_training` 下 t1/t2/t3 未定义；
  - `--artifact-root`；
  - 所有 run 写入 canonical artifact path。
- `src/config.py`
  - `pipeline.run_tracing=false`；
  - `model_selection.method=min_val_objective | last_epoch`；
  - `legacy_test_selection=false`，若开启必须打印 `COMPATIBILITY ONLY — NOT FOR PAPER`。
- `src/detection/evaluation.py`
  - 正式结果禁止 test MCC 选择 epoch；
  - 默认按 validation objective 选择，或显式 last_epoch；
  - type-only/baseline objective = mean val loss_type；
  - joint MSTC objective = mean val score_raw = mean(loss_type + lambda_time*loss_time)。
- `src/detection/orthrus_gnn_testing.py`
  - 每 checkpoint：reset history → replay train → val → 不 reset → test。

### Canonical artifact path

统一：

```text
<artifact_root>/matrix_artifacts/<config_id>/<dataset>/runs/<model_variant>/seed_<seed>/
```

内部：

```text
config_resolved.yml
environment.json
checkpoints/
edge_scores/test/event_predictions.csv
node_scores/metrics.json
node_scores/node_predictions.csv
evaluation_results/calibration/
runtime.json
```

`collect_results.py` 后续必须扫描这个 schema，不允许假设 `<artifact_root>/runs/...`。

### 测试

- pipeline stages；
- run_from_training no NameError；
- baseline 逐 tensor `assert_close`；
- legacy_test_selection 默认关闭并有醒目标记；
- artifact path contract。

### 修正原计划中的不合理点

- 不再写“state_dict hash 使用 atol/rtol”；hash 若使用只检查 exact equality。
- Commit 1 的离线 smoke 仍是合成/mock；真实无 PostgreSQL 运行放 C2。

---

## Commit 2 — Metadata Cache + PostgreSQL Fallback + Train-only Word2Vec

### 目标

在已有 artifacts 下脱离 PostgreSQL，并让论文主实验的 `train_only` 语料范围真正生效。

### 修改文件

- `src/config.py`
  - 读取 artifact/data/database 环境变量；
  - `semantic_features.corpus_scope ∈ {train_only, official_full_dataset}`。
- `src/provnet_utils.py`
  - DB 参数支持 env/cfg。
- `src/labelling.py`
  - metadata cache 优先。
- `src/detection/orthrus_gnn_testing.py`
  - 节点展示文本优先从 `node_metadata[node_id]["display"]` 生成；
  - 不再强制依赖单独 `nodeid2msg.pkl`。
- Word2Vec / semantic corpus 构建相关模块
  - **必须实际按 split 过滤语料**；不能只加 config 字段。
  - `train_only`：只使用 train dates/split 语料 fit Word2Vec；validation/test 专属 token 不进入 fit。
  - `official_full_dataset`：原始兼容模式，产物标记 `compatibility_only=true`。

### 新增/维护

- `src/mstc/metadata_cache.py`
- `tests/test_metadata_cache.py`
- `tests/test_semantic_corpus_scope.py`

### 论文公平性约束

正式主表中：

```text
ORTHRUS-ano
Semantic MLP
GraphSAGE
MSTC-PIDS
```

全部统一 `corpus_scope=train_only`。

不做 `official_full_dataset vs train_only` 论文对比实验。

### smoke

`--stages train,test,evaluate` 在完整预处理 artifacts + metadata cache 存在时不连接 PostgreSQL。

---

## Commit 3 — Exact Time Targets + Unambiguous Time Statistics

### 目标

把时间监督目标与编码器历史状态彻底分离，消除原计划“batch 入口更新/末尾更新”的冲突。

### 修改文件

- `src/data_utils.py`
  - `src_type / dst_type / edge_type_index / local_event_index / global_event_index / window_id / split`。
- `src/mstc/time_gap.py`
  - `TimeGapStatistics`；
  - `TimeGapTargetBuilder`。

### 语义

`TimeGapTargetBuilder` 在 batch 内按 `(timestamp, global_event_index)` 稳定排序，逐事件：

```text
compute target from target_last_seen
→ update target_last_seen
```

它只产生 label/loss target，不把真实 gap/bucket喂给模型，因此允许在 batch 内逐事件更新。

MultiScale encoder 的 HistoryStore 是另一套状态，在 C5 仍严格 batch query 后 insert。

### Quantile contract

- scale：对 train `delta_seconds` 取 Q50/Q90/Q99；
- time classification：对 `log1p(delta_seconds)` 取 Q20/Q40/Q60/Q80；
- JSON 同时写 `scale_boundaries_seconds` 和 `time_bucket_boundaries_log1p`。

Q99 是诊断/Single-window 上界，不作为 long scale 硬截断。

### 测试

- first seen / exact second occurrence；
- same-batch repeated node exact target；
- negative gap error；
- quantile data-space；
- global event id 对齐。

---

## Commit 4 — MSTC Dual-task Wrapper + Raw Score Contract

### 目标

新增双任务，不污染原 `Orthrus` / `OrthrusEncoder`。

### 修改文件

- `src/model.py`
  - 新增 `MSTCOrthrus`；原 `Orthrus` 保持不变。
- `src/decoders.py`
  - 原 `EdgeTypeDecoder` 保持对外行为；
  - 新增 `TimeGapDecoder`：shared hidden → src/dst 6-class heads。
- `src/factory.py`
  - variant 选择 MSTC/legacy。
- training/testing
  - MSTC 路径读取字典输出。

### 唯一 raw score 定义

```python
loss_time_i = 0.5 * (loss_time_src_i + loss_time_dst_i)

joint:     score_raw_i = loss_type_i + lambda_time * loss_time_i
type-only: score_raw_i = loss_type_i
time-only: score_raw_i = loss_time_i
```

`lambda_time=0.3` 是论文主配置预先固定值。

### 高风险修正

- 不再写“Orthrus.forward 返回结构变化”；变化只发生在新增 `MSTCOrthrus`。
- 不在 `OrthrusEncoder.forward` 注入 time target；target builder 在 MSTC wrapper/training-testing orchestration 中使用。

### 测试

- decoder shape/loss；
- score_raw 逐事件公式；
- no target-as-input。

---

## Commit 5 — Multi-scale History + Shared Configurable Graph Backbone

### 目标

实现多尺度上下文，但不硬编码 GraphTransformer。

### HistoryStore

只保存必要索引：

```text
neighbor_id
event_id
timestamp_ns
direction
```

规则：

- `timestamp_ns=int64` 强制；
- node/event id 只有范围安全才可 int32；
- 不缓存 node embedding / edge one-hot。

### Multi-scale semantics

```text
short:  0 < delta <= Q50
medium: Q50 < delta <= Q90
long:   delta > Q90
```

Q99 不截断 long。

对照定义：

```text
Recent-20       = latest 20, no time cutoff
Recent-24       = latest 24, fair-budget control
Single-window-24= delta <= Q99, latest 24
Equal-24        = 8+8+8, equal non-empty scale fusion
Gated-24        = 8+8+8, learned masked gate
```

只 Gated-24 报 gate distribution。

### Encoder

```python
self.shared_graph_encoder = backbone_factory(cfg.encoder.backbone)
```

支持 GraphTransformer / GraphSAGE。

Semantic MLP 不进入该模块。

### Runtime diagnostics

保存：

```text
short_empty_ratio
medium_empty_ratio
long_empty_ratio
candidate_truncation_ratio
```

不增加 candidate_capacity 参数扫描。

### smoke

C5 只要求 gate/scale/history 产物，不提前要求 C6 才生成的 `calibration_level`。

---

## Commit 6 — Calibration + Node Aggregation + Node Decision

### 目标

只做校准和节点级后处理；**删除 DatasetViews/Host-network 三视图**。

### 校准职责

- `src/mstc/calibration.py`：纯算法；
- `src/mstc/calibration_runner.py`：唯一 I/O owner，负责 read raw events → fit val → LOO val → calibrate test → save artifacts。

条件：

```text
triplet(src_type, edge_type, dst_type)
→ type_pair(src_type, dst_type)
→ global
```

`min_triplet_samples=100 / min_type_pair_samples=200` 是预固定支持下限；报告 fallback rate，不做额外网格搜索。

真实 `edge_type` 只用于模型输出后的 post-hoc conditioning，不作为当前事件模型输入。

### Node aggregation API

```python
build_node_event_scores(event_records, include_dst=True)
aggregate(node_to_event_scores, method, topk)
```

避免把 `include_dst` 放在映射已经完成之后。

代码支持：`max / mean / topk_mean / topk_sum`；论文只比较前三者。

### Node decision

```text
validation_quantile(0.999)
max_validation
kmeans
```

KMeans 属于 node-decision 方法，不把它当普通 scalar threshold 描述。

### Ablation correctness

`w/o Calibration`：

```text
raw event score
+ same topk_mean(K=5)
+ same validation_quantile(0.999)
```

只去掉 calibration，不同时改变 threshold strategy。

### 训练 checkpoint 边界

training checkpoint 只保存训练状态；calibrator/threshold/node predictions 属于 evaluation artifacts。

---

## Commit 7 — GraphSAGE + Semantic MLP

### 目标

验证跨骨干通用性和简单语义基线。

### GraphSAGE

- 两层 SAGEConv；
- 无 edge features 时显式忽略 edge_attr；
- 能接入 C5 的 shared multi-scale sampler/encoder、C4 time task、C6 calibration/aggregation。

### Semantic MLP

- 输入 `concat(x_src,x_dst)`；
- 不读 edge_index/history；
- 不进入 MultiScaleOrthrusEncoder；
- 不使用 gate。

因此删除旧计划“Semantic MLP 需要调整 gate 输入维度”的描述。

### smoke

只要求：

- 三条路径能 forward/evaluate；
- parameter_count 可计算；
- 不把“GraphSAGE 参数量必须显著低于 Transformer”作为测试 invariant。

---

## Commit 8 — Reduced Experiment Matrix + All-in-One + Post-processing Analysis

### 目标

把实验工程收敛到真正用于 SCI 小论文的矩阵，并统一 Notebook/结果导出。

### 运行器

- `run_experiment.py`
- `run_matrix.py`
- `run_topk_sensitivity.py`
- `collect_results.py`
- `export_tables.py`

### 主实验

```text
Datasets: THEIA_E3, THEIA_E5
Models: Semantic MLP, GraphSAGE, ORTHRUS-ano, MSTC-PIDS Full
Seeds: 0,1,2
```

全部 `corpus_scope=train_only`。

### E3-only 实验

- 完整消融：3 seeds；
- multi-scale detailed study：seed 0；
- time-task detailed study：seed 0；
- calibration / node-decision：3 seeds，复用事件分数；
- GraphTransformer+MSTC vs GraphSAGE+MSTC：3 seeds；
- efficiency：复用现有 runtime，不重新训练；
- Top-k sensitivity：3 seeds，纯后处理。

### 删除的正式实验

```text
Host-network 3 views
cross-engagement type_only/zero-shot
history_device CPU vs CUDA
Top-k Sum
legacy_test_selection comparison
official_full_dataset vs train_only
```

### Top-k dedicated postprocess

`run_topk_sensitivity.py`：

```text
source = existing mstc_full run
K = 1,3,5,10,20
seeds = 0,1,2
```

硬约束：

- source event scores/calibration missing → fail；
- 禁止自动 fallback 到 training；
- per-K recompute validation threshold；
- 输出 `topk_sensitivity_results.csv`；
- 输出 `topk_event_count_summary.csv`。

### Result export

`export_tables.py` 分组：

```text
main
ablation
multiscale_diagnostic
time_diagnostic
score_calibration
node_decision
backbone_generalization
efficiency
topk_sensitivity
```

论文表只用 mean±std；`best` 若保留只能标记 `debug_only`。

### All-in-One Notebook

唯一正式 Notebook：

```text
notebooks/<现有 All-in-One Notebook 的实际文件名>.ipynb
```

Sections：

```text
0 environment
1 artifact/data paths
2 artifact validation
3 preprocess(optional)
4 smoke
5 main E3+E5
6 ablation E3
7 diagnostics E3
8 topk postprocess
9 evaluate/recovery
10 collect
11 paper export
```

Notebook 只调用 Python CLI，不复制训练逻辑。沿用仓库现有 All-in-One Notebook 的实际文件名，不新建/重命名第二个官方入口。

### OOM 规则

正式实验只允许在语义等价时统一调整 `batch_size`。

禁止自动降低：

```text
candidate_capacity
neighbor budgets
model dimensions
number of layers
```

发生其他 OOM：记录 failure → 人工选定统一资源配置 → 所有受影响对照一起重跑。

---

## 跨 Commit 关键风险表

| 阶段 | 风险 | 修订后的约束 |
|---|---|---|
| C1 | test-based epoch selection | compatibility-only，论文默认关闭 |
| C1 | artifact path 漂移 | 单一 canonical matrix_artifacts schema |
| C2 | `train_only` 只写配置不生效 | 必须测试 val/test token 不进入 W2V fit |
| C3 | time target 与 history 状态混用 | 两套 state 分离 |
| C3 | quantile 数据域含糊 | scale=seconds；bucket=log1p_seconds |
| C4 | raw anomaly score 多版本 | 唯一 score_raw contract |
| C5 | Q99 丢掉极长历史 | long 为 >Q90，Q99 不截断 |
| C5 | GraphTransformer 硬编码 | shared configurable graph backbone |
| C6 | calibration/threshold 同时变化 | 分层实验，一次只改一层 |
| C6 | include_dst 时机错误 | event→node 映射阶段生效 |
| C8 | Top-k 变成重复训练 | dedicated postprocess，missing source 就 fail |
| C8 | 单 seed K 曲线 | 复用主模型 seeds 0,1,2 |
| C8 | OOM 静默改实验语义 | 仅允许语义等价 batch-size 调整 |

---

## 验收对照

1. E3/E5 主结果均完成，4 模型 × 3 seeds；
2. 所有论文主模型统一 train-only semantic corpus；
3. baseline compatibility mode 与论文正式协议明确分离；
4. exact time targets 与 encoder history 无状态混淆；
5. score_raw 公式唯一；
6. multi-scale long 不被 Q99 硬截断；
7. GraphSAGE 能接 MSTC multi-scale/time/calibration；
8. Semantic MLP 不进入 multi-scale/gate；
9. calibration I/O 单一 owner；
10. `w/o Calibration` 是单变量消融；
11. Top-k sensitivity 不重新训练；
12. per-K threshold 由各自 validation 分布计算；
13. `topk_event_count_summary.csv` 可解释 K 退化；
14. 只保留一个 All-in-One Notebook；
15. 不存在 Host-network/zero-shot 等已删除实验的正式配置；
16. 正式 OOM 策略不静默改变模型语义；
17. paper tables 只使用 mean±std，不挑 best seed；
18. `pytest -q` 全部通过。

---

## 已知限制与论文边界

- 本篇论文不研究 auditd+Zeek 多源融合；
- 不研究跨 Engagement zero-shot；
- 不研究网络语义三视图贡献；
- 不做 calibration support threshold / node quantile 的大规模超参搜索；
- multi-scale 和 time-task 的详细变体使用 E3 seed 0 作为诊断性分析，核心模块效应由 E3 3-seed 完整消融提供统计支撑；
- E5 主要承担第二数据集主结果验证，不重复完整专项矩阵。
