# C8 Sparse Node Sidecar 与 True Warm-Load 修复报告

## 结论

本轮同时修复了三个相互关联但不能混为一谈的问题：

1. `nodes.pt` 用 `[max_node_id + 1, feature_dim]` 分配两张 dense 浮点表，内存由最大 ID 而非活跃节点数决定；
2. persistent sidecar 命中前仍先扫描并反序列化全部 source `TemporalData`；
3. 旧实现命中 sidecar 后把已经发生的 source load 计数清零，因而 warm telemetry 不可信。

新实现使用按角色分离、ID 严格有序的 active-node sparse sidecar，并让 schema-v4 manifest 足以在 tensor load 前恢复 window metadata。有效 warm path 只对 source 路径做发现与 `stat`/fingerprint，不反序列化任何 source `TemporalData`。

## 已确认的 Colab 故障事实

这不是“疑似 OOM”。`failure.json` 对 full baseline 和 `mstc_full` 都记录了 `returncode=-9`、`signal=9`、`signal_name=SIGKILL`；Linux kernel 还给出了 `/jupyter-children` memory cgroup OOM 证据：

| 运行 | kernel 记录的 anon RSS |
| --- | ---: |
| full baseline | 10,669,212 kB |
| `mstc_full` | 10,667,316 kB |

根 cgroup 的 `memory.events=0` 不能反驳子 cgroup `/jupyter-children` 的直接 OOM 记录。

## 旧 dense 设计及精确成本

真实 THEIA_E3 manifest 为：

- `total_events=46,794`
- `node_table_dim=131`
- `node_table_max_node=1,236,201`
- `edge_type_num_types=10`

旧 sidecar 的 `x_src` 和 `x_dst` 各占：

```text
(1,236,201 + 1) * 131 * 4 = 647,769,848 bytes
```

两张表合计 `1,295,539,696 bytes`，即 `1,235.523 MiB` / `1.207 GiB`。设计错误在于把 ID 空间跨度当成节点基数；稀疏 ID 即使只有少量活跃节点，也会为所有缺失 ID 分配特征行。

## schema-v4 sparse 架构

由于现有语义只能证明同一角色内、同一 node ID 的特征不变，不能证明 `x_src` 与 `x_dst` 跨角色必然相同，本轮没有擅自合并两套表。新 `nodes` sidecar 保存：

- `src_node_ids: int64[U_src]`
- `src_features: float32[U_src, D]`
- `dst_node_ids: int64[U_dst]`
- `dst_features: float32[U_dst, D]`

两组 ID 均严格递增。事件查询先从 compact event tensor 取得 src/dst node ID，再用 `torch.searchsorted` 定位，并做 exact equality 检查；不存在的 ID 会报错，不会通过 clamp 映射到错误行。构建时保留 duplicate-feature invariance validation，覆盖同一 window 内重复和跨 window 重复。

`msg` 仍由对应角色的 sparse node feature 与 compact `edge_type` 重建；compact `src/dst/t/src_type/dst_type/edge_type_index/split` 和 global event ordering 不变。

## 内存预算

对真实 manifest，最保守地假设每个 event 都产生一个独立 src 和一个独立 dst 活跃节点：

```text
feature data = 2 * 46,794 * 131 * 4 = 49,040,112 bytes
node ID index = 2 * 46,794 * 8       =    748,704 bytes
total                                      49,788,816 bytes
```

即特征 `46.768 MiB`、ID 索引 `0.714 MiB`，合计 `47.482 MiB` / `0.046 GiB`。与旧 `1,295,539,696 bytes` 相比，最坏上界仍缩小 `26.021x`，减少 `96.157%`。这只是由真实 `46,794 events` 推出的保守上界；实际 unique role-node 数更小时会继续下降。

synthetic 验收使用 100 个事件、最大 node ID `90,000,000`、`D=5`，src/dst 各 3 个活跃节点。测试实测的张量存储为：

- feature：`(3 + 3) * 5 * 4 = 120 bytes`
- ID index：`(3 + 3) * 8 = 48 bytes`
- 合计：`168 bytes`

该值与 `90,000,000` 的最大 ID 无线性关系。这里报告的是张量 storage bytes，不把 `torch.save` 容器头和 metadata 序列化开销冒充模型数据大小。

## True warm-load 路径

schema-v4 manifest 新增/固化了恢复 lazy collection 所需的 ordered window specs，包括 `split`、`split_window_index`、`global_window_id`、`global_offset`、`num_events`、`max_node`、field metadata 和 `total_events`。调用顺序改为：

```text
resolve source paths
-> os.stat / source fingerprint
-> validate manifest and semantic fingerprint
-> restore window specs from manifest
-> load compact tensor sidecar
-> load sparse node tensor sidecar
-> construct BoundedFullData
```

有效 warm sidecar 上，source `TemporalData` load 是严格 `0`；只有两个 sidecar tensor load。关键测试在建立 sidecar 后 monkeypatch source artifact loader，使任何 source load 立即抛错，第二次 `load_all_datasets()` 仍成功，并验证 train/val/test specs、window ordering、global offsets 完全一致。

cold path 为控制改动风险，仍保留有界 two-pass：一次 metadata scan、一次 compact/sparse build。因此真实 THEIA_E3 的 430 个 source windows 在 cold build 中是 `430 + 430 = 860` 次 source `TemporalData` load；5-window synthetic 测试严格验证为 `5 + 5 = 10` 次。warm valid sidecar 为 `0` 次。

## schema、发布与失效

- `PERSISTENT_SCHEMA_VERSION` 从 3 bump 到 4；目录可以复用，但 schema 3 manifest 会自动 miss 并重建，不要求用户手工清理。
- manifest 校验 source path、mtime、size fingerprint 和 semantic config fingerprint；任一变化都会 cold rebuild。
- compact/nodes tensor 使用 generation-specific 文件名，先发布 tensor，最后原子替换 manifest；`completed=false` 或缺文件的 manifest 不会被当成完整 sidecar。
- manifest 记录 `num_src_active_nodes`、`num_dst_active_nodes`、`node_feature_storage_bytes`，加载时会重新核对 dtype、shape、严格排序、计数、字节数和最大 ID。

## Telemetry 与 RAM phase

counter 只累加，不在 cache hit 后清零。新增并验证：

- `sidecar_manifest_hit`
- `sidecar_tensor_load_count`
- `warm_source_temporaldata_load_count`
- `cold_source_temporaldata_load_count`
- `num_src_active_nodes` / `num_dst_active_nodes`
- `node_feature_storage_bytes` / `node_id_index_bytes`

原有 `metadata_source_load_count`、`compact_build_source_load_count`、`total_source_artifact_load_count` 保留其真实含义。warm 验收值分别为 0、0、0；`sidecar_tensor_load_count=2`。

RAM telemetry 同时记录 current RSS 和 kernel process peak RSS，并覆盖这些关键阶段：

- `before sidecar load`
- `before source metadata scan`
- `after source metadata scan`
- `after compact sidecar load`
- `after sparse node sidecar load`
- `after BoundedFullData construction`
- `before model construction`

## 语义与回归覆盖

新增 sparse-ID 测试覆盖 100 events、重复 ID、乱序事件、src-only、dst-only、跨多个 window 重复、invariance violation、empty events，以及 `x_src`、`x_dst`、`msg`、`edge_type` 相对 eager reference 的逐值相等。既有回归继续覆盖：

- baseline lazy/eager exact event-field equality
- MSTC consumer 与 multiscale encoder
- global event index 和 window/global offsets
- replay protocol
- no-future-leakage
- semantic-config/source-identity sidecar invalidation
- notebook static contracts
- experiment matrix 与 runtime telemetry JSON serialization

具体测试命令和最终结果以本分支最终验收报告为准。

## 下一次 Colab 验收步骤

1. 从 `fix/c8-sparse-node-sidecar` 获取 notebook 固定的 40 位 production+tests commit，并确认 notebook checkout 后 `git rev-parse HEAD` 精确匹配。
2. 清空本次实验输出目录；旧 schema-v3 sidecar 不必手工删除，首次运行应自动 miss/rebuild 为 v4。
3. 运行一次 full baseline cold build，核对 manifest schema 4、active-node 数、feature/index bytes、cold load 计数，以及每个 RSS phase 的 current/peak 值。
4. 不修改 source artifact 和 semantic config，第二次运行相同配置；必须看到 `sidecar_manifest_hit=true`、`sidecar_tensor_load_count=2`、三项 source load 计数均为 0。
5. 再运行 `mstc_full`，确认 baseline/MSTC 都越过 sidecar 与 `BoundedFullData` 构造阶段，并观察 `before model construction` 之后的内存峰值。
6. 如仍发生 SIGKILL，以最后成功持久化的 phase 和 `/jupyter-children` kernel OOM 记录定位下一处峰值，不能用根 cgroup `memory.events=0` 排除 OOM。
