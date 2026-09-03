# MAGIC 外部基线统一协议接入审计：持久化进度

- 【任务状态】COMPLETE
- 【当前阶段】Phase B4：最终静态验收完成
- 【当前分支】`audit/magic-unified-baseline`
- 【起始 HEAD】`9f8607d20e0cb3f3c37dbbf76b2479e12ffb4548`
- 【已审计内容】Phase A 冻结门禁；MAGIC 审计分支存在性；官方论文/README/源码/许可证/依赖；数据输入、特征、训练目标、阈值、checkpoint、ground truth、粒度、指标、seed、资源；当前项目 split、ground truth、runner、matrix、collector、exporter、canonical artifacts 和 evaluator
- 【已创建文档】`docs/MAGIC_BASELINE_AUDIT_PROGRESS.md`；`docs/MAGIC_BASELINE_INTEGRATION_AUDIT.md`；`docs/MAGIC_BASELINE_IMPLEMENTATION_PLAN.md`
- 【已修改文件】仅上述三份新增 Markdown 文档
- 【已完成事项】Phase A 轻量验收与冻结提交；从冻结提交创建独立审计分支；定位并只读审计官方 MAGIC；完成当前项目接入点审计；完成集成审计报告与 M1–M10 实施计划；完成 Markdown 结构、尾随空白与 Git 状态静态验收
- 【未完成事项】无（后续实现前仍需用户确认两项协议选择，不属于本轮审计未完成项）
- 【blockers】官方代码使用 test labels 选择 threshold；没有 THEIA_E5 loader；whole-test-graph 与跨 split 特征维度存在 transductive/future-information 风险；官方 seed 控制不完整；当前环境无 DGL 且核心版本与官方固定版本不同
- 【外部源码是否成功定位】是；官方仓库 `https://github.com/FDUDSDE/MAGIC`，审计 commit `aa0b647eea74b6faa0e52eb444370c4411a32cbe`，无 tag；sparse shallow clone 位于 `/tmp/magic-audit.8K6zpm/MAGIC`，未获取 data/checkpoints/eval_result
- 【是否运行任何训练】否
- 【是否安装任何依赖】否
- 【当前 git status】三份新增审计 Markdown 均未跟踪；无其他改动
- 【当前 git diff --stat】普通 tracked diff 为空；未跟踪文件不会出现在该命令输出中

## 阶段记录

### Phase A：实验矩阵修订冻结

已完成。分支 `docs/reduce-paper-experiment-matrix` 在基准 `16e85bff1f0481a213181663a15a197cdbd9d454` 上通过 `git diff --check` 与针对性轻量测试（41 passed，14 条既有 Notebook cell-id warning），无删除、无 YAML/MSTC 核心差异。冻结提交为 `9f8607d20e0cb3f3c37dbbf76b2479e12ffb4548`，未 push、未创建 PR。

### Phase B0：审计分支建立

已完成。目标分支此前不存在；已从冻结提交创建 `audit/magic-unified-baseline`。本阶段未运行训练、未安装依赖、未下载数据。

### Phase B1：官方来源与方法审计

已完成。USENIX Security 2024 正式论文脚注与官方仓库 README 相互印证仓库身份；当前 `main`/HEAD 为 `aa0b647eea74b6faa0e52eb444370c4411a32cbe`（2024-10-24），仓库无 tag，MIT License（版权行为 `Copyright (c) 2023 Jimmyokok`）。官方代码只原生列出 E3-Trace/E3-THEIA/E3-CADETS，没有 THEIA_E5。

已确认 entity-level 核心为 node/edge type one-hot（论文表述为 label lookup）、GAT masked graph autoencoder、SCE masked-feature reconstruction + sampled-structure BCE、benign-train node embeddings 上的 KNN 和归一化 mean-distance node anomaly score。未使用 Word2Vec；文本名称不进入模型；timestamp 仅用于排序，未作为模型特征。

已确认最高风险项：`model/eval.py` 使用 `precision_recall_curve(y_test, score)`，并用 THEIA 的硬编码 test recall 目标选择 `best_thres`，属于确定的 test-label threshold selection。官方 entity-level 训练固定 50 epochs、只保存最后 checkpoint，没有 validation、early stopping 或 test-based checkpoint selection。源码还从 train+test 共同确定 one-hot 维度，并在完整 test graph 上编码，存在 test-feature/transductive 与未来拓扑边界风险。官方 seed 固定为 0，未提供 CLI seed；未设置 DGL seed，且 `torch.backends.cudnn.determinstic` 拼写错误。

### Phase B2：当前项目接入点审计

已完成。当前项目 THEIA_E3 split 为 train `graph_2..5`、val `graph_9`、test `graph_10/12/13`；THEIA_E5 为 train `graph_8..10`、val `graph_11`、test `graph_14/15`。ground truth 来自当前 attack-specific CSV 的精确 UUID→node-id 集合，不应替换成 MAGIC/ThreaTrace 标签。

推荐将 MAGIC 作为 external baseline backend 接在 `run_matrix.py` 调度与 canonical artifact contract 之间，不进入 `factory.py`/MSTC model factory。输入 adapter 只消费 canonical `src/dst/t/src_type/dst_type/edge_type_index/split/global_event_index`，忽略 ORTHRUS Word2Vec；输出 adapter 保留 raw MAGIC score，并生成 `node_predictions.csv`、`metrics.json`、`runtime.json`。当前 `collect_results.py`/`export_tables.py` 已能在 artifact 合同满足后收集并分类 `magic`。

### Phase B3：审计与实施计划文档

已完成 `MAGIC_BASELINE_INTEGRATION_AUDIT.md` 的 25 个要求章节，结论为 **CONDITIONAL GO**；已完成 `MAGIC_BASELINE_IMPLEMENTATION_PLAN.md` 的 M1–M10 分阶段计划。计划坚持 external backend、原始 MAGIC GMAE/KNN score 与外围统一协议分离；本轮没有创建 adapter、config YAML、容器或结果 artifact。

### Phase B4：最终静态验收

已执行 `git diff --check`；并单独扫描三份未跟踪 Markdown 的行尾空白，因为普通 `git diff` 不包含未跟踪文件。审计报告包含 25 个要求章节，实施计划包含 M1–M10。阶段 B 只有三份新增 Markdown，无 tracked diff、无代码/YAML/Notebook/结果修改；未运行训练、未安装依赖、未下载数据，且未 commit、未 push、未创建 PR。
