# ORTHRUS 实施规划 (IMPLEMENTATION_PLAN)

> 严格遵循任务说明 §24「按 8 个独立 Commit 提交」的开发顺序。本规划仅为最小改动方案，
> 不进行大规模重构，不修改基线组件（`LastNeighborLoader` / `OrthrusEncoder` / `EdgeTypeDecoder` /
> 阈值方法 / KMeans）的对外行为。
>
> 保留原始 `Orthrus` 类不变；新增 `MSTCOrthrus`（或 `MultiTaskOrthrus`）由 `model.variant` 选择。
>
> 用户引用的 `docs/ORTHRUS_MSTC_IMPLEMENTATION_SPEC.md` 不存在；本规划以仓库内
> `docs/ORTHRUS-MSTC-PIDS_完整改造任务说明.md` 为准。

## 0. 总览：8 个 Commit 的全局依赖

```
C1 baseline 保护 ─► C2 Colab + 元数据缓存 ─► C3 时间统计 ─► C4 双任务
   │                                                         │
   │                                                         ▼
   │                                          C5 多尺度采样 + 编码
   │                                                         │
   │                                                         ▼
   │                                          C6 条件校准 + Top-k
   │                                                         │
   │                                                         ▼
   │                                          C7 GraphSAGE + MLP
   │                                                         │
   │                                                         ▼
   │                                                       C8 实验矩阵 + Notebook
   └───────────────────────────────────────────────► C8 收尾
```

每个 Commit 必须满足：
- 修改面尽量小，禁止重写 `OrthrusEncoder / LastNeighborLoader / EdgeTypeDecoder`。
- 关闭所有 MSTC 开关（`model.variant == orthrus_baseline`）时输出与原 ORTHRUS 一致（atol=1e-6, rtol=1e-5）。
- 必须新增至少 1 个单元测试和 1 个 smoke test。
- 提交前必须 `pytest -q` 通过。

---

## Commit 1 — 基线保护（Baseline Protection）

### 目标

在不重写原流水线的前提下提供阶段开关、关闭 tracing、修复 `NameError`、统一输出目录。

### 修改文件

- `src/orthrus.py`
  - 新增 `--stages preprocess,train,test,evaluate` CLI flag（默认 `all`）。
  - 新增 `--skip-tracing` CLI flag 与 `pipeline.run_tracing` cfg。
  - 修复 `t1/t2/t3` 未初始化：当 `args.run_from_training` 或 `--stages` 跳过这些阶段时，`time_consumption` 中相应字段填 `0.0`（不抛 `NameError`）。
  - 把 tracing 调用包在 `if 'trace' in stages and cfg.pipeline.run_tracing`。
  - 新增 `--artifact-root` CLI flag（默认 `os.environ.get('ORTHRUS_ARTIFACT_ROOT', './artifacts')`）。
- `src/config.py`
  - 新增 `pipeline` 子树：`run_tracing: bool = False`，`stages: list[str] = ['all']`。
  - 新增 `model_selection` 子树：`method: min_val_mean_edge_loss`（默认）、`legacy_test_selection`（默认关闭）。
  - 新增 `cfg.pipeline.stages` 与 `args.stages` 的解析与合并。
  - 新增 `--stages`、`--skip-tracing`、`--artifact-root` CLI 注册。
  - **不修改** `detection.gnn_testing.threshold_method`（该字段在当前代码路径中从未被读取，属于遗留配置；后续清理中处理）。
- `src/detection/evaluation.py`
  - 将 epoch 选择协议改为 `model_selection.method: min_val_mean_edge_loss`（默认）。
  - 允许备选：`last_epoch`。
  - 新增 `legacy_test_selection` 模式（默认关闭）：保留原有按 test MCC 选 best epoch 的行为，仅用于与官方 ORTHRUS 结果的兼容性分析，不得用于论文主结果。
  - 注释化（不删除）原有 test-based best 选择，便于对账。
- `src/detection/orthrus_gnn_testing.py`
  - 在每个 `model_epoch_*` 入口按**正确 replay 协议**执行：
    1. `model.encoder.reset_state()`（仅 OrthrusEncoder 分支）。
    2. `replay_train_history(model, train_data, cfg)`：直接复用现有 batch loop，`torch.no_grad()` + `eval()`，只构建历史，不计算梯度。
    3. 连续处理 val（不重置历史）。
    4. **不重置历史**。
    5. 连续处理 test。
  - **禁止**在 val 与 test 之间插入 reset/replay。
  - 新增 `replay_train_history` 私有函数。

### 新增文件

- `tests/test_baseline_compatibility.py`
  - 合成 5 节点 + 10 条事件的 `TemporalData`，关闭 MSTC 全部开关后比较：
    - 训练 epoch 末 `state_dict` 哈希与旧实现一致（atol/rtol）。
    - `node_evaluation` 计算的 score 列表与旧实现一致。
- `tests/test_orthrus_pipeline_stages.py`
  - 模拟 `--stages train,test,evaluate`：仅跑训练 + 测试 + 评估，验证不调用 build_graphs / embed。
- `tests/test_run_from_training_no_nameerror.py`
  - 模拟 `--run_from_training` 路径，确认 `time_consumption` 中 skipped 字段为 `0.0`，不抛 `NameError`。

### 单元测试

- `test_baseline_compatibility.py::test_edge_loss_matches_baseline`
- `test_orthrus_pipeline_stages.py::test_skip_preprocess`
- `test_run_from_training_no_nameerror.py::test_time_consumption_keys_present`

### Smoke test

Commit 1 **尚未实现 metadata cache 和 PostgreSQL 解耦**，不得声称真实完整运行 `train,test,evaluate` 可完全离线。Commit 1 的 smoke test 范围为：

```bash
# 阶段开关测试
python src/orthrus.py THEIA_E3 \
  --stages train,test,evaluate \
  --run_from_training \
  --artifact-root /tmp/orthrus_c1 \
  --seed 0 --cpu

# baseline 兼容性测试（合成数据）
pytest -q tests/test_baseline_compatibility.py

# pipeline 开关测试
pytest -q tests/test_orthrus_pipeline_stages.py

# NameError 修复测试
pytest -q tests/test_run_from_training_no_nameerror.py
```
预期（合成数据下）：
- `--stages train,test` 等变体不抛 NameError。
- baseline 数值与原 ORTHRUS 在合成数据上吻合（atol/rtol）。
- 不连 PostgreSQL（测试均使用合成数据）。

**真正的无 PostgreSQL 完整 smoke test 移到 Commit 2**（需要 metadata cache + preprocessed artifacts 全部就绪）。

### 高风险模块

- `src/orthrus.py` 的 `time_consumption` 计算（任务说明 §5.2 已点名）。
- `src/detection/evaluation.py` 改 best epoch 选择策略（与现有 wandb 输出格式兼容性）。

### 与任务说明冲突的现有实现

- `evaluation.py` 用 test MCC 选 best epoch（违规 §2.2）；改为 `min_val_mean_edge_loss`（默认）。
- `gnn_testing.threshold_method="str"` 为遗留无效配置字段（不影响当前运行，应后续清理）。

### 不在本 Commit 做的事

- 不引入 metadata cache，不动 PostgreSQL 回退（属于 Commit 2）。
- 不新增任何 MSTC 模块。

---

## Commit 2 — Colab / 元数据缓存 / 数据库回退

### 目标

让 train / test / evaluate 在已有预处理 artifacts 时不强制要求 PostgreSQL；导出 `metadata/` 缓存。

### 修改文件

- `src/config.py`
  - 新增环境变量读取：`ORTHRUS_ARTIFACT_ROOT / ORTHRUS_DATA_ROOT / ORTHRUS_DB_HOST / ORTHRUS_DB_PORT / ORTHRUS_DB_USER / ORTHRUS_DB_PASSWORD`，默认值保留原硬编码。
  - 在 `get_default_cfg` / `get_yml_cfg` 末尾把这些环境变量同步到 cfg。
  - 新增 `semantic_features` cfg 子树：`corpus_scope`（`official_full_dataset` 或 `train_only`）。
- `src/provnet_utils.py::init_database_connection`
  - 新增「环境变量覆盖」分支（host/port/user/password 优先用 cfg.database.*，再回退到环境变量）。
- `src/labelling.py`
  - 新增 `try_load_metadata_cache(cfg)`：若 `cfg._metadata_dir/node_metadata.pkl` 等文件存在，则跳过 `init_database_connection`。
  - `get_ground_truth / get_GP_of_each_attack / get_t2malicious_node`：先尝试 cache，失败再走 PostgreSQL。
- `src/detection/orthrus_gnn_testing.py`
  - 在 `main` 开头加 `nodeid2msg` 的 cache 优先读取（新增 `cfg._metadata_dir/nodeid2msg.pkl`）。
  - 新增 `cfg.testing.include_node_messages` 开关；关闭时跳过 `srcmsg/dstmsg` 填表。
- `src/detection/evaluation_utils.py::get_node_to_path_and_type`
  - 已有 pkl cache，保留并标注。
- `src/detection/evaluation_utils.py::compute_tw_labels`
  - 删除 `if os.path.exists(out_file): os.remove(out_file)`，尊重 cache。

### 新增文件

- `src/mstc/__init__.py`
- `src/mstc/metadata_cache.py`
  - `class MetadataCache`：统一读写 `node_metadata.pkl / uuid_to_node_id.pkl / node_id_to_uuid.pkl / ground_truth_nodes.pkl / attack_to_nodes.pkl / time_to_malicious_nodes.pkl / relation_mapping.json / dataset_manifest.json`。
  - `dump_from_postgres(cfg)`：从 PostgreSQL 导出上述文件，写入 `cfg._metadata_dir/`。
- `config/experiments/baseline.yml`
  - `model.variant: orthrus_baseline`，其余字段与 `orthrus.yml` 一致；指向现有 artifacts。
- `tests/test_metadata_cache.py`
  - 用临时目录 fake 一个 metadata 目录，确认 `MetadataCache.load_node_metadata` 命中时不再连 PostgreSQL（mock `psycopg2.connect`）。

### 单元测试

- `test_metadata_cache.py::test_load_hits_cache`
- `test_metadata_cache.py::test_env_var_overrides_db`

### Smoke test

```bash
python src/orthrus.py THEIA_E3 \
  --stages train,test,evaluate \
  --run_from_training \
  --artifact-root /tmp/orthrus_cache \
  --seed 0 --cpu
```
预期：完全离线运行（mock 已设置 metadata cache + preprocessed artifacts）。

### 高风险模块

- `labelling.py::get_t2malicious_node`（缓存 vs DB 一致性）。
- `orthrus_gnn_testing.py::nodeid2msg` cache 命中（影响 CSV 中 srcmsg/dstmsg 列内容）。

### 与任务说明冲突的现有实现

- `compute_tw_labels` 强制删除 cache。
- `init_database_connection` 不读环境变量。

### 不在本 Commit 做的事

- 不动 word2vec（属于既有 artifacts 的一部分，依赖 dump_from_postgres 时由 dataset 决定）。
- 不新增 MSTC 模型层。

---

## Commit 3 — 时间统计（TimeGap Statistics）

### 目标

在 `TemporalData` 上显式保留 `src_type / dst_type / edge_type_index / event_index`；训练集生成时间分位数与时间桶；不依赖 DB。

### 修改文件

- `src/data_utils.py`
  - 在 `extract_msg_from_data` / `extract_msg_node_type_only` 末尾额外写入：
    - `g.src_type`: `[E]` long（源节点类型索引 0/1/2）。
    - `g.dst_type`: `[E]` long（目的节点类型索引 0/1/2）。
    - `g.edge_type_index`: `[E]` long（来自 `g.edge_type.argmax(-1)`）。
    - `g.local_event_index`: `[E]` long（该时间窗口内的从 0 开始的索引）。
    - `g.window_id`: 标量 int（该窗口在全局 train/val/test 序列中的序号）。
  - 增加断言：`src/dst/t/src_type/dst_type/edge_type_index/local_event_index` 长度一致；`edge_type_index ∈ [0, num_edge_types)`；`src_type/dst_type ∈ [0, num_node_types)`；`g.t` 非递减。
  - `global_event_index`、`split` 在后续 `load_all_datasets` 中拼接生成（Commit 3 仅在 TemporalData 层面补充 local 字段）。
- `src/mstc/time_gap.py`（**唯一实现位置**）
  - **`TimeGapStatistics` 只实现一份，放在 `src/mstc/time_gap.py`**。
  - `src/temporal.py` **不添加** `TimeGapStatistics`，保留原始 `LastNeighborLoader`。
  - `class TimeGapStatistics`（任务说明 §8.1-§8.4）：
    - `fit(train_data_list)`：按时间顺序扫描，统计 `delta_src / delta_dst`（秒 = ns / 1e9），并计算分位数 `[0.2,0.4,0.6,0.8]` 与 `[0.5,0.9,0.99]`。
    - `transform(delta_seconds)` → bucket_id ∈ {0=NO_HISTORY, 1=VERY_SHORT, 2=SHORT, 3=MEDIUM, 4=LONG, 5=VERY_LONG}。
    - `transform_batch(g, last_seen_per_node)`：在 batch 入口更新 last_seen（仅用历史事件，不允许使用当前 batch 的未来事件）。
  - 负时间差抛 `ValueError`；`last_seen` 未见过的节点返回 `NO_HISTORY=0`。
- `src/graph_construction/build_orthrus_graphs.py` / `src/edge_featurization/embed_edges_feature_word2vec.py`
  - **不修改**。它们产出的 `TemporalData` 在 `data_utils.extract_msg_from_data` 中已具备 `src/dst/t/edge_type`，由 data_utils 补充显式字段。

### 新增文件

- `src/mstc/time_gap.py`（**唯一实现位置**）
- `config/experiments/time_only.yml`
- `tests/test_time_gap.py`
  - NO_HISTORY、log1p、负时间差、分位数边界、bucket 划分。
  - 合成 TemporalData：5 个节点、8 条事件，时间戳 10/20/50/100/110，验证 bucket 划分稳定。
- `tests/test_global_event_index.py`
  - `global_event_index` 与 `full_data.msg` 拼接位置严格一致；`split` 字段正确标注 train/val/test；`window_id` 全局唯一。

### 单元测试

- `test_time_gap.py::test_first_seen_is_no_history`
- `test_time_gap.py::test_seconds_conversion`
- `test_time_gap.py::test_negative_delta_raises`
- `test_time_gap.py::test_quantile_boundaries`
- `test_global_event_index.py::test_global_index_matches_full_data`
- `test_global_event_index.py::test_split_field_correct`

### Smoke test

```bash
# Commit 3 只生成时间标签，不修改模型
# smoke test 验证新增字段存在且格式正确
pytest -q tests/test_time_gap.py
pytest -q tests/test_global_event_index.py
```
预期（合成数据下）：
- `g.src_type / g.dst_type / g.edge_type_index / g.local_event_index / g.window_id` 字段存在且形状正确。
- `TimeGapStatistics.fit` 在训练集上统计正确。
- `TimeGapDecoder`（Commit 4 才接入）届时可正常读取时间标签。

### 高风险模块

- `data_utils.extract_msg_from_data`：新增字段会改变 `TemporalData` 的存储尺寸，但只在主流程加载时使用，对老 checkpoint 无影响。
- `temporal.TimeGapStatistics.fit`：必须严格按训练集时间顺序扫描；不许使用 val/test。

### 与任务说明冲突的现有实现

- 现有 `TemporalData` 不含 `src_type / dst_type / edge_type_index / event_index`，需要后续模块显式读取。

### 不在本 Commit 做的事

- 不实现 `TimeGapDecoder`（属于 Commit 4）。
- 不修改 `Orthrus.forward` 返回结构。

---

## Commit 4 — 双任务一致性学习（Time Task）

### 目标

在保留 `EdgeTypeDecoder` 的同时新增 `TimeGapDecoder`；联合 loss；测试输出含 `loss_time_* / src_time_target / dst_time_prediction` 等详细字段。

### 修改文件

- `src/mstc/time_gap.py`
  - 在 `OrthrusEncoder.forward` 阶段注入 `time_targets`：
    - 由 `TimeGapStatistics` 维护 `last_seen`；每个 batch 入口调用 `transform_batch` 计算 `src_time_target / dst_time_target`。
    - **`last_seen` 只在 batch 入口更新；不在 batch 中插入当前事件**——这是因果 micro-batch 关键。
- `src/model.py`
  - **不直接修改原始 `Orthrus` 类**。
  - 新增 `class MSTCOrthrus(nn.Module)`（或 `MultiTaskOrthrus`）：
    - 接收 `encoder`（可为 `OrthrusEncoder` 或 `MultiScaleOrthrusEncoder`）。
    - 返回字典：
      - 训练：`{"loss": scalar, "loss_type": scalar, "loss_time": scalar, "loss_time_src": scalar, "loss_time_dst": scalar}`。
      - 推理：`{"score_raw": [E], "loss_type": [E], "loss_time": [E], "loss_time_src": [E], "loss_time_dst": [E], "edge_logits": [E, R], "src_time_logits": [E, 6], "dst_time_logits": [E, 6]}`。
    - `Orthrus` 类保持不变；`model_factory` 根据 `cfg.model.variant` 选择 `Orthrus` 或 `MSTCOrthrus`。
- `src/decoders.py`
  - 保留原始 `EdgeTypeDecoder` 不变。
  - 新增 `class TimeGapDecoder(nn.Module)`：
    - 结构：`shared_repr = SharedMLP(concat(h_src, h_dst))` → `src_logits = src_head(shared_repr)`，`dst_logits = dst_head(shared_repr)`。
    - 其中 `SharedMLP` 输出 `hidden_dim`；`src_head = Linear(hidden_dim, 6)`；`dst_head = Linear(hidden_dim, 6)`。
    - **不得先把共享层输出降成 6 维，再接两个分类头**。
    - `forward(h_src, h_dst) -> src_logits, dst_logits`。
    - `loss(src_logits, src_target, dst_logits, dst_target, reduction) -> loss_time`（取均值 0.5*(src+dst)）。
- `src/factory.py::decoder_factory`
  - 新增 `cfg.detection.gnn_training.decoder.time_gap.enabled` 开关；启用时 append `TimeGapDecoder`。
- `src/detection/orthrus_gnn_training.py::train`
  - 当 `cfg.model.variant == "mstc"` 时使用 `MSTCOrthrus`；其他情况使用 `Orthrus`。
  - 使用 `MSTCOrthrus` 时：`outputs = model(batch, full_data)`；`outputs["loss"].backward()`。
  - 增加 `cfg.detection.gnn_training.decoder.time_gap.lambda_time` 加权。
- `src/detection/orthrus_gnn_testing.py::test`
  - 当使用 `MSTCOrthrus` 时：读取 `outputs["score_raw"]`、`outputs["loss_type"]`、`outputs["loss_time_*"]` 等；CSV 增加详细列。

### 新增文件

- `config/experiments/time_joint.yml`、`time_type_only.yml`、`time_time_only.yml`
- `tests/test_time_gap_decoder.py`
  - 验证 `logits/loss` 接口拆分；`forward` 输出形状 `[B,6]`；loss 形状 reduction 切换正确。
- `tests/test_no_future_leakage.py`
  - 验证 batch 入口 transform 时 last_seen 仍未更新；batch 末尾才更新。
  - 验证「批次内同一节点 reference_time = min(current_event_times_of_node)」（任务说明 §10.3）。

### 单元测试

- `test_time_gap_decoder.py::test_logits_loss_split`
- `test_no_future_leakage.py::test_no_leak_in_batch`

### Smoke test

```bash
python src/orthrus.py THEIA_E3 \
  --stages train,test,evaluate \
  --run_from_training \
  --enable-time-task true \
  --lambda-time 0.3 \
  --artifact-root /tmp/orthrus_t4 \
  --seed 0 --cpu
```
预期：训练 loss 包含 time 任务；`event_predictions.csv` 列齐全。

### 高风险模块

- `Orthrus.forward` 返回值结构变化：会破坏所有调用者（training/testing/evaluation），必须同步修改 `OrthrusEncoder.forward` / `model.py` / `factory.py`。
- `lambda_time` 默认为 `0.3`，必须可配置且不破坏 baseline（关闭时为 0）。

### 与任务说明冲突的现有实现

- `EdgeTypeDecoder.forward` 直接返回 loss（违规 §9.1）。
- `Orthrus.forward` 返回 scalar（违规 §9.3）。
- `orthrus_gnn_testing.py::test()` 输出 CSV 列不齐（违规 §9.4）。

### 不在本 Commit 做的事

- 不动邻居采样（属于 Commit 5）。
- 不动校准与节点聚合（属于 Commit 6）。

---

## Commit 5 — 多尺度采样与编码（Multi-Scale）

### 目标

新增 `MultiScaleNeighborLoader` + `MultiScaleOrthrusEncoder`，不破坏 `LastNeighborLoader` / `OrthrusEncoder`；门控融合 + 历史 replay。

### 修改文件

- `src/mstc/history_store.py`（新增）
  - `class HistoryStore`：
    - **只保存必要的历史索引**：`neighbor_id`（long）、`event_id`（long）、`timestamp_ns`（**必须为 int64，纳秒时间戳**）、`direction`（可选，int8）。
    - **不得使用 int32 存储纳秒时间戳**（int32 上限约 68 年，溢出风险）。
    - **不得把 edge_type one-hot 存入历史存储**。
    - **不得把 src_emb/dst_emb 存入历史存储**。
    - 边特征和节点特征通过 `event_id` 从 `full_data` 读取。
    - `node_id` 和 `event_id` 只有在确认数值范围安全后才可考虑 int32。
    - `insert(batch)` / `query(nodes, ref_times)` → 返回 short/medium/long 三组 `(node_id, event_id, timestamp_ns, direction)` 元组。
- `src/mstc/multiscale_sampler.py`（新增）
  - `class MultiScaleNeighborLoader(nn.Module)`：
    - `__call__(nodes, ref_times)` → 三个尺度的子图（与 `LastNeighborLoader` 同接口风格）。
    - `reset_state()` / `state_dict()` / `load_state_dict()`。
    - 内部使用 `HistoryStore`。
- `src/mstc/multiscale_encoder.py`（新增）
  - `class MultiScaleOrthrusEncoder(nn.Module)`：
    - `self.shared_graph_encoder`：单一 `GraphTransformer` 实例，三个尺度共享。
    - 三个尺度分别运行 `self.shared_graph_encoder`，可选 `scale_embedding[SHORT/MEDIUM/LONG]` 加到 `x_proj`。
    - 门控 MLP：`gate_input = cat(z_short, z_medium, z_long, current_src_proj, current_dst_proj)` → `gate_logits` → `masked_softmax(non_empty_mask)` → `beta`（`beta.sum == 1`，空尺度权重为 0）。
    - 融合：`h_src = sum(beta_s * h_src_s)`；h_dst 同理。三个尺度全空时 fallback 到 `current_src_proj / current_dst_proj`。
    - `reset_state` 透传到内部 `MultiScaleNeighborLoader`。
- `src/encoders.py`
  - **不修改** `OrthrusEncoder`。
- `src/factory.py::encoder_factory`
  - 新增 `cfg.detection.gnn_training.encoder.context.mode ∈ {'recent', 'multiscale'}`。
  - 当 `mode == 'multiscale'` 时构造 `MultiScaleOrthrusEncoder`；否则保持 `OrthrusEncoder`。
  - 共享 `GraphTransformer` 由 `cfg.encoder.share_encoder` 控制（默认 True）：三个尺度在同一实例上 forward。
- `src/detection/orthrus_gnn_testing.py::main`
  - 在每个 model_epoch_* 入口按正确 replay 协议执行：`reset_state()` → `replay_train_history()` → val → test（val 与 test 之间不重置）。
  - Commit 1 已建立 replay 钩子，此处只需确认 multiscale encoder 也支持。

### 新增文件

- `src/mstc/__init__.py`（更新 export）
- `config/experiments/multiscale_recent20.yml`、`multiscale_recent24.yml`、`multiscale_single_window.yml`、`multiscale_equal.yml`、`multiscale_gate.yml`
- `tests/test_multiscale_sampler.py`
  - 合成 4 个事件（t=10/20/50/100），current=110，验证 short/medium/long 分配准确。
- `tests/test_multiscale_encoder.py`
  - 验证 `short_encoder is medium_encoder is long_encoder`；门控权重和为 1；空尺度权重为 0；全空时 fallback；无 NaN。
- `tests/test_history_state.py`
  - `state_dict` / `load_state_dict` 一致；`replay` 重建后历史与原始一致。

### 单元测试

- `test_multiscale_sampler.py::test_short_medium_long_split`
- `test_multiscale_encoder.py::test_shared_encoder`
- `test_multiscale_encoder.py::test_gate_sums_to_one`
- `test_history_state.py::test_replay_equivalence`

### Smoke test

```bash
python src/orthrus.py THEIA_E3 \
  --stages train,test,evaluate \
  --run_from_training \
  --enable-multiscale true \
  --multiscale-fusion gated \
  --history-checkpoint-mode replay \
  --artifact-root /tmp/orthrus_t5 \
  --seed 0 --cpu
```
预期：训练时间显著增加（3 个子图共享编码器）；`event_predictions.csv` 增加 `short_gate_weight / medium_gate_weight / long_gate_weight / calibration_level` 列（calibration_level 由 Commit 6 写）。

### 高风险模块

- 共享 `GraphTransformer`：三个尺度在同一模块 forward 时需保证 batch 维正确串联；PyG 的 `TransformerConv` 在不同子图上调用是允许的。
- `HistoryStore` timestamp_ns 必须用 int64；溢出检查需在单元测试中覆盖。

### 与任务说明冲突的现有实现

- `OrthrusEncoder` 仅 1 跳。
- 测试时未 `reset_state` + replay（任务说明 §10.8）。

### 不在本 Commit 做的事

- 不动 decoder（Commit 4 已就绪）。
- 不动校准 / 节点聚合（Commit 6）。

---

## Commit 6 — 条件校准 + Top-k 节点聚合

### 目标

实现 `HierarchicalRelationCalibrator`（任务说明 §11）+ `NodeScoreAggregator`（§12）+ `DatasetViews`（§15）；保留旧 threshold/kmeans 对照路径。校准作为独立后处理阶段（不在 GNN testing 循环内拟合）。

### 修改文件

- `src/detection/orthrus_gnn_testing.py::main`
  - **GNN testing 阶段只负责输出完整 raw event scores**：写入 CSV 包含 `score_raw / loss_type / loss_time_*` 等原始输出。
  - **不得在逐 batch 循环里拟合校准器**。
- `src/mstc/calibration_runner.py`（新增，统一唯一位置）
  - 独立后处理 runner，负责：
    1. 读取完整 val raw scores（从 CSV 或从 checkpoint 重新跑 val）。
    2. `HierarchicalRelationCalibrator.fit(val)`。
    3. `transform_val_with_loo(val)` → leave-one-out 转换 val scores。
    4. `calibrate(test)` → 测试集校准分数 + calibration_level。
    5. 写入 `calibrator.pkl` 和 `calibration_summary.json`。
- `src/mstc/calibration.py`
  - **只放校准算法类**，不负责 I/O、不读取 CSV、不保存结果。
  - `class HierarchicalRelationCalibrator`：
    - `fit(val_event_records)`：按 `(src_type, edge_type, dst_type)` 与 `(src_type, dst_type)` 与 `GLOBAL` 三组桶分别排序，使用 `numpy.searchsorted`。
    - `calibrate(test_event_records)`：返回 `score_calibrated = -log(max(p, epsilon))` + `calibration_level ∈ {triplet, type_pair, global}`。
    - `transform_val_with_loo(val_event_records)`：leave-one-out 经验 p 值。
- `src/mstc/aggregation.py`
  - `class NodeScoreAggregator`：
    - `aggregate(node_to_event_scores, method, topk, include_dst)`：`topk_mean / max / mean / topk_sum`。
- `src/mstc/dataset_views.py`
  - `apply_dataset_view(data, mode)`：在 `load_data_set` 之后、`full_data` 拼接之前应用。
  - 合法模式：`host_only`、`host_network_structure`、`host_network_full`。
  - 过滤后必须重新生成：`global_event_index`、`full_data`、`历史事件编号`、`事件数量统计`。
- `src/detection/node_evaluation.py::get_node_predictions`
  - 用 `NodeScoreAggregator` 替换 `reduce_losses_to_score`（保留旧函数作对照）。
  - `use_kmeans` 分支保持原行为，新增 `node_threshold.method ∈ {validation_quantile, max_validation, kmeans}` 分支。
- `src/detection/evaluation_utils.py`
  - 新增 `calibration_summary.json` 与 `calibrator.pkl` 保存逻辑。
  - `get_threshold` 增加 `validation_quantile` 与 `kmeans`（旧 max/mean 保留作对照）。
- `src/config.py`
  - 新增 `calibration / node_aggregation / node_threshold / dataset_view` cfg 子树默认值。

### 新增文件

- `src/mstc/calibration_runner.py`（唯一实现位置；`calibration.py` 仅放算法类）
- `config/experiments/calibration_max.yml`、`calibration_quantile.yml`、`calibration_kmeans.yml`、`calibration_global_p.yml`、`calibration_relation.yml`、`calibration_hierarchical.yml`、`host_only.yml`、`host_network_structure.yml`、`host_network_full.yml`
- `tests/test_calibration.py`
  - `monotonic_p`、`< 1`、add-one、leave-one-out、空组回退、epsilon 不为 0。
- `tests/test_aggregation.py`
  - `less_than_k → mean(all)`、`more_than_k → top_k_mean`、`include_dst` 切换。
- `tests/test_dataset_views.py`
  - `host_only` 过滤 netflow 节点；`host_network_structure` 置零 netflow 语义；`host_network_full` 保留全部。

### 单元测试

- `test_calibration.py::test_p_monotonic`
- `test_calibration.py::test_loo_p`
- `test_aggregation.py::test_topk_mean`
- `test_dataset_views.py::test_host_only_filter`
- `test_dataset_views.py::test_global_event_index_regenerated`

### Smoke test

```bash
python src/orthrus.py THEIA_E3 \
  --stages train,test,evaluate \
  --run_from_training \
  --calibration-method hierarchical_relation \
  --node-aggregation-method topk_mean \
  --node-threshold-method validation_quantile \
  --node-threshold-quantile 0.999 \
  --artifact-root /tmp/orthrus_t6 \
  --seed 0 --cpu
```
预期：`calibration_runner.py` 成功运行；`metrics.json` 含 `FP_per_million / Attack_Detection_Rate / MCC / F1 / AUPRC / AUROC`；`node_predictions.csv` / `event_predictions.csv` 完备；`calibrator.pkl` / `calibration_summary.json` 已保存。Attack Detection Rate 的攻击数量动态读取 `len(cfg.dataset.ground_truth_relative_path)`。

### 高风险模块

- `HierarchicalRelationCalibrator.fit`：必须仅使用正常 val 集；不许读 test。
- `evaluation.main`：`best epoch` 选择必须使用 `min_val_mean_edge_loss`（默认）或 `last_epoch`；`legacy_test_selection`（按 test MCC）仅作官方兼容分析且默认关闭；不得使用 val MCC 或 test MCC 选论文主模型。

### 与任务说明冲突的现有实现

- `reduce_losses_to_score` 只支持 `np.max / np.mean`（缺 topk_mean）。
- `compute_kmeans_labels` 仅看尾部 K 个节点（缺 FP/M 指标）。

### 不在本 Commit 做的事

- 不动 encoder / decoder。
- 不引入 GraphSAGE / MLP（Commit 7）。

---

## Commit 7 — 跨骨干：GraphSAGE 与 Semantic MLP

### 目标

新增 `GraphSAGEBackbone` 与 `SemanticMLPBackbone`，使 MSTC-PIDS 与基线均可跨骨干验证。Semantic MLP 是独立简单非图基线，**不进入 `MultiScaleOrthrusEncoder` 的门控逻辑**。

### 修改文件

- `src/encoders.py`
  - 新增 `class GraphSAGEBackbone(nn.Module)`：
    - `SAGEConv(in_dim → hid_dim) → SAGEConv(hid_dim → out_dim)`，无 edge_dim。
    - `forward(x, edge_index, **kwargs)`：与 `GraphTransformer` 同接口。
  - 新增 `class SemanticMLPBackbone(nn.Module)`：
    - 输入 `concat(x_src, x_dst)`；两层 MLP；**不读 edge_index**；不使用多尺度；不使用门控。
    - 这是一条独立的非图基线路径，与 `MultiScaleOrthrusEncoder` 互不干扰。
- `src/factory.py::encoder_factory`
  - 新增 `cfg.detection.gnn_training.encoder.backbone ∈ {graph_transformer, graphsage, semantic_mlp}`。
  - 当 `backbone == 'semantic_mlp'` 时：跳过 `MultiScaleOrthrusEncoder`，直接使用 `SemanticMLPBackbone` + 对应 Decoder。
  - `semantic_mlp` 模式强制 `context.mode = 'none'`（不读取历史邻居）。
- `src/detection/orthrus_gnn_testing.py::test`
  - 当 `backbone=semantic_mlp` 时：不输出 `short_gate_weight / medium_gate_weight / long_gate_weight` 列。

### 新增文件

- `config/experiments/backbone_graphtransformer.yml`、`backbone_graphsage.yml`、`backbone_mlp.yml`
- `tests/test_backbones.py`
  - 三个 backbone forward 形状一致；共享 encoder 时仍共享参数；MLP 无 edge_index 输入。

### 单元测试

- `test_backbones.py::test_output_shapes_match`
- `test_backbones.py::test_mlp_ignores_edge_index`

### Smoke test

```bash
python src/orthrus.py THEIA_E3 \
  --stages train,test,evaluate \
  --run_from_training \
  --encoder-backbone graphsage \
  --enable-multiscale true \
  --enable-time-task true \
  --calibration-method hierarchical_relation \
  --artifact-root /tmp/orthrus_t7 \
  --seed 0 --cpu
```
预期：`metrics.json` 中 `parameter_count` 显著低于 GraphTransformer。

### 高风险模块

- `GraphSAGEBackbone` 不支持 edge_dim；`OrthrusEncoder` 与 `MultiScaleOrthrusEncoder` 必须兼容 `edge_dim=None`。
- `SemanticMLPBackbone` 不读 edge_index；多尺度编码器需把 `gate` 输入维度相应调整。

### 与任务说明冲突的现有实现

- `encoder_factory` 硬编码 `GraphTransformer`。

### 不在本 Commit 做的事

- 不动校准 / 节点聚合（已在 Commit 6 完成）。

---

## Commit 8 — 实验矩阵 + Colab Notebooks

### 目标

提供 `run_experiment.py / run_matrix.py / collect_results.py / export_tables.py`、Colab 笔记本、汇总 CSV。

### 修改文件

- `src/experiments/__init__.py`（新增）
- `src/experiments/run_experiment.py`（新增）
  - 解析 `--dataset / --config / --seed / --artifact-root / --stages / --checkpoint`；转发给 `src.orthrus.main`。
- `src/experiments/run_matrix.py`（新增）
  - 解析 `--datasets / --configs / --seeds`，按 (dataset × config × seed) 调度 `run_experiment.main`。
- `src/experiments/collect_results.py`（新增）
  - 扫描 `--artifact-root` 下所有 `runs/<model>/seed_*/metrics.json`，输出 `all_runs.csv`。
- `src/experiments/export_tables.py`（新增）
  - 按 main / ablation / calibration / efficiency 维度分组，生成 `mean / std / best / median / successful_seeds / failed_seeds`。
- `src/mstc/experiment_utils.py`（新增）
  - `dump_environment(cfg, out_dir)`：写 `environment.json`、`runtime.json`、`config_resolved.yml`、git commit。
- `src/mstc/metrics.py`（新增）
  - `compute_classification_metrics / compute_fp_per_million / compute_attack_detection_rate`。
- `src/detection/node_evaluation.py::main`
  - 在 `stats` 中追加 `fp_per_million / attack_detection_rate / events_per_second / peak_gpu_memory_mb / peak_cpu_memory_mb`。
- `src/detection/orthrus_gnn_training.py`
  - 记录 `train_seconds_per_epoch / total_train_seconds / peak_gpu_memory_mb / events_per_second`，写到 `runtime.json`。
  - **完整 checkpoint 保存**：至少保存 `model_state_dict`、`optimizer_state_dict`、`epoch`、`Python random state`、`NumPy random state`、`Torch CPU RNG state`、`Torch CUDA RNG states`、`config hash`。如果使用 scheduler，也保存 `scheduler_state_dict`。**只保存模型权重不称为完整断点续训**。
- `src/detection/orthrus_gnn_testing.py::main`
  - 记录 `test_seconds / events_per_second`。

### 新增文件

- `config/experiments/baseline.yml`、`mstc_full.yml` 以及 §4 列出的全部消融 yml。
- `notebooks/00_colab_environment.ipynb`、`01_preprocess_theia.ipynb`、`02_baseline_smoke_test.ipynb`、`03_train_main_models.ipynb`、`04_run_ablations.ipynb`、`05_collect_results.ipynb`
- `tests/test_checkpoint_resume.py`
  - 训练 1 epoch → 保存 → 重新加载 → 训练 1 epoch → 验证参数一致。
- `tests/test_metrics.py`
  - FP/M、Attack Detection Rate、events_per_second 等计算正确。
- `tests/test_environment_dump.py`
  - `dump_environment` 写出的 `config_resolved.yml` 包含 `git commit` / `python / pytorch / pyg / cuda` 版本。

### 单元测试

- `test_checkpoint_resume.py::test_resume_continues`
- `test_metrics.py::test_fp_per_million`
- `test_metrics.py::test_attack_detection_rate`
- `test_environment_dump.py::test_environment_keys`

### Smoke test

```bash
python src/experiments/run_matrix.py \
  --datasets THEIA_E3 \
  --configs config/experiments/baseline.yml,config/experiments/mstc_full.yml \
  --seeds 0 \
  --artifact-root /tmp/orthrus_t8
python src/experiments/collect_results.py --artifact-root /tmp/orthrus_t8
python src/experiments/export_tables.py --artifact-root /tmp/orthrus_t8
```
预期：生成 `results/all_runs.csv`、`results/main_results.csv` 等。

### 高风险模块

- `run_matrix.py` 的 OOM 处理：
  - **正式实验中不得自动降低 `candidate_capacity`、`neighbor budgets`、模型维度**——这些改变会改变实验语义，导致不同 seed 不可直接汇总。
  - **只允许在数学语义保持一致时调整 `batch_size`**（语义等价）。
  - 发生其他 OOM 时：当前 run 失败，输出失败原因，人工制定统一资源配置，用同一配置重新运行所有对照模型。
- `collect_results.py` 必须容忍部分 seed 失败；统计 `successful_seeds / failed_seeds`。

### 与任务说明冲突的现有实现

- 现有 `wandb.finish()` 硬编码为 `online`/`disabled` 二值（任务说明 §5.4 要求 `offline`）。
- 没有 `environment.json` 持久化。

### 不在本 Commit 做的事

- 不再添加新模型 / 校准方法（前面 7 个 Commit 已就绪）。

---

## 跨 Commit 依赖与风险表

| Commit | 高风险模块 | 失败后果 | 缓解措施 |
|--------|-----------|----------|----------|
| C1 | `src/orthrus.py::time_consumption`（NameError） | `--run_from_training` 必崩 | 5 行修复 |
| C1 | epoch 选择改 `min_val_mean_edge_loss` | `legacy_test_selection` 与新协议冲突 | 注释旧代码，默认关闭 |
| C1 | replay 协议（reset→train→val→test，无中间重置） | val/test 间重置导致 cur_e_id 错位 | 每个 checkpoint 只 replay 一次 |
| C2 | `labelling.py::get_t2malicious_node` cache | 离线模式下评估失败 | pkl 与 DB 双路径 |
| C3 | `data_utils.extract_msg_from_data` 新增字段 | 老 checkpoint 加载报错 | 字段可选，缺省时回退 |
| C3 | `global_event_index` 与 `full_data` 拼接严格对齐 | 校准器 LOO 和节点聚合出错 | 多尺度采样（Commit 5）依赖此一致性 |
| C4 | 新增 `MSTCOrthrus` | 原始 `Orthrus` 接口被意外污染 | 通过 `model.variant` 选择，原始类保持不变 |
| C4 | `TimeGapDecoder` 结构（SharedMLP→hidden→6） | 结构错误无法收敛 | 单元测试覆盖 logits/loss 形状 |
| C5 | `HistoryStore` timestamp_ns int64 | int32 溢出（68年限制） | 单元测试覆盖大时间戳 |
| C5 | 共享 `GraphTransformer` 多尺度 | 多尺度引入额外显存 | 共享 encoder 是任务强制要求 |
| C6 | 校准为独立后处理（非 GNN 测试循环内） | 与当前 `orthrus_gnn_testing.py` 架构冲突 | 新增 `calibration_runner.py` |
| C6 | `HierarchicalRelationCalibrator.fit` 只用 val | 与 `legacy_test_selection` 冲突 | C1 已将 `legacy_test_selection` 改为默认关闭 |
| C7 | `SemanticMLPBackbone` 独立路径 | 与 `MultiScaleOrthrusEncoder` 混用 | MLP 是独立 backbone，强制 `context.mode='none'` |
| C8 | OOM 处理不改变实验语义 | 不同 seed 无法汇总 | 仅允许调 batch_size，其他 OOM 失败并记录原因 |
| C8 | `run_matrix.py` seed 失败容错 | 结果不完整 | 统计 `successful_seeds / failed_seeds` |

### 8 个 Commit 之间的关键依赖关系

- **C1 不依赖 C2 的数据库解耦**：C1 只做 pipeline 开关 + replay 协议 + epoch 选择协议。
- **C3 只生成时间标签，不修改模型**：生成 `local_event_index / global_event_index / split / window_id` 字段体系，供 C5/C6 使用。
- **C4 使用新的 `MSTCOrthrus`（不改变 `Orthrus`）**：两条路径通过 `model.variant` 选择。
- **C5 的多尺度历史 `event_id` 与 `global_event_index` 严格一致**：`HistoryStore` 只存 `event_id`，边特征通过 `event_id` 从 `full_data` 读取。
- **C6 校准与模型测试解耦**：GNN testing 只输出 raw scores，校准器作为独立后处理运行。
- **C7 的 Semantic MLP 不进入多尺度模块**：MLP 是独立 encoder backbone，与 `MultiScaleOrthrusEncoder` 互不干扰。
- **C8 不得自动改变实验语义**：正式实验中 OOM 只允许调 batch_size；其他配置变化必须人工决策。

---

## 验收对照（任务说明 §23）

| # | 验收项 | 由哪个 Commit 覆盖 |
|---|--------|--------------------|
| 1 | 原 ORTHRUS 基线仍可运行 | C1, C2 |
| 2 | THEIA_E3 完成预处理、训练、测试和评估 | C1-C8 |
| 3 | THEIA_E5 完成相同流程 | C1-C8 |
| 4 | 已有 artifacts 时不需要 PostgreSQL | C2 |
| 5 | 每个创新模块可独立开关 | C3-C7 |
| 6 | 完整模型可在 GraphTransformer 上运行 | C5, C6 |
| 7 | 完整模块可在 GraphSAGE 上运行 | C7 |
| 8 | 所有测试通过 | 全部 Commit |
| 9 | 测试集标签仅在最终评估函数中读取 | C1, C2 |
| 10 | 时间分位数仅由训练集生成 | C3 |
| 11 | 校准分布仅由正常验证集生成 | C6 |
| 12 | 输出事件级和节点级结果 | C4, C6 |
| 13 | 输出完整配置和环境信息 | C8 |
| 14 | Colab 断线后可从 checkpoint 继续 | C8 |
| 15 | 所有实验可由配置文件复现 | C8 |
| 16 | 不运行攻击重建也不会报错 | C1 |
| 17 | `--run_from_training` 不会报未定义变量 | C1 |
| 18 | 结果收集脚本能汇总多个 seed | C8 |
| 19 | 代码有类型提示和必要注释 | 全部 Commit（增量） |
| 20 | README 新增完整使用说明 | C8 顺带更新 README |

---

## 最小改动原则

- 不重写 `OrthrusEncoder / LastNeighborLoader / EdgeTypeDecoder` 的对外行为。
- 所有新模块通过 cfg 开关启用；关闭时数值结果与原 ORTHRUS 一致（atol=1e-6, rtol=1e-5）。
- 新增模块集中在 `src/mstc/` 与 `src/experiments/`，不污染原包结构。
- 现有测试目录 `tests/` 不存在；本规划在 C1 中开始建立。

---

## 已知限制（与任务说明原文的差距）

- THEIA_E3 攻击场景为 2 个、THEIA_E5 为 1 个；Attack Detection Rate 仅作辅助指标（任务说明 §13 已说明）；攻击数量必须动态读取 `len(cfg.dataset.ground_truth_relative_path)`。
- Word2Vec 当前为 transductive feature preprocessing（使用整库节点语料）；本期新增 `semantic_features.corpus_scope` 配置，官方复现用 `official_full_dataset`，论文主实验用 `train_only`。
- GraphSAGE 无边特征（任务说明 §14.2 已说明）。
- Semantic MLP 是独立非图基线，不使用历史邻居、不使用多尺度、不使用门控。