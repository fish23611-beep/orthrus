# 实验矩阵最终减负修订：持久化进度报告

- 【任务状态】COMPLETE
- 【当前阶段】Phase 5 已完成
- 【当前分支】`docs/reduce-paper-experiment-matrix`
- 【起始 HEAD】`16e85bff1f0481a213181663a15a197cdbd9d454`
- 【起始工作区】干净（`git status --porcelain=v1` 无输出）
- 【已检查文件】任务附件；两份正式规范全文；`src/experiments/run_matrix.py`；`src/experiments/collect_results.py`；`src/experiments/export_tables.py`；全部 `config/experiments/*.yml` 的关键语义；唯一正式 Colab Notebook；相关 exporter/matrix/Notebook/config 测试
- 【已修改文件】`docs/ORTHRUS-MSTC-PIDS_完整改造任务说明.md`；`docs/IMPLEMENTATION_PLAN.md`；`notebooks/ORTHRUS_MSTC_PIDS_AllInOne_Colab.ipynb`；`src/experiments/export_tables.py`；`tests/test_export_tables.py`；`tests/test_notebooks.py`；本进度报告
- 【已完成事项】起始 Git 核验；两份规范审计与同步修订；runner/config/Notebook/export 审计与必要最小修订；默认 export 收缩为三类 paper_core；Semantic MLP/w/o Gate optional exporter 支持保留；全文关键词、MAGIC 状态、文件存在性和时间语义审计；文档静态合同、Notebook JSON、Python 语法、YAML/分组单测和 `git diff --check` 全部通过
- 【尚未完成事项】无；仅待用户人工复核并自行决定是否提交
- 【已发现但未处理的问题】(1) 仓库当前没有从 E3 Full artifacts 统一生成 w/o Calibration / Global Calibration / w/o Top-k 的复用 orchestration；Notebook 已阻止把三者送入训练矩阵，后续需单独实现/审计。(2) MAGIC 未接入。(3) `src/experiments/run_topk_sensitivity.py` 与 `config/analysis/topk_sensitivity.yml` 实际不存在，文档已纠正为 archived_optional“已设计未实现”，本轮未重建。(4) nbformat 对旧 Notebook cell 缺少 id 给出 14 条 warning，本轮不做无关的大规模 Notebook normalization
- 【是否修改代码】是，仅修改结果导出的分组/选择层及对应测试；未修改 runner、训练框架或模型实现
- 【是否删除文件】否
- 【是否运行真实实验】否
- 【当前 git diff --stat】6 个已跟踪文件：304 insertions(+), 189 deletions(-)；未跟踪进度文件不计入普通 `git diff --stat`
- 【当前 git status -sb】分支 `docs/reduce-paper-experiment-matrix`；两份正式规范、唯一正式 Notebook、exporter 与两个对应测试已修改；本进度报告为新增未跟踪文件；无删除
- 【是否安全继续】YES

## 阶段记录

### Phase 1：只读审计

已完成。除本进度报告外未修改项目文件。全文审计确认两份规范彼此采用同一套旧矩阵，但均与本轮最终减负要求冲突，具体见顶部“已发现但未处理的问题”。

### Phase 2：两份正式规范修订

已完成首轮修订：统一四类状态、24-run 主实验、六项核心消融、最小跨骨干、训练/后处理复用、MAGIC pending 状态及公平协议、Notebook/导出目标和当前论文边界。两份规范均明确禁止把 MAGIC 规划误写为已接入或已有结果。

### Phase 3：配置、runner、Notebook 与结果导出检查

已完成。`run_matrix.py` 的 datasets/configs/seeds 都是必填项，不存在隐藏默认大矩阵；`collect_results.py` 不触发实验，因此均无需修改。`export_tables.py` 默认只导出 main/ablation/backbone_generalization，archived 表需显式开关。Notebook 默认重任务仍全部关闭；启用后主矩阵只跑仓库内 18 个 run，核心消融/跨骨干训练组仅包含真正新增训练项，archived 组需要二次显式授权。配置文件全部保留。

测试：首次执行 exporter + Notebook + config 静态测试共 40 passed；增加分组回归测试后重跑 exporter + Notebook 共 25 passed。仅有既存 nbformat MissingIDFieldWarning，无失败。

### Phase 4：全文一致性审计

已完成。按任务关键词对两份正式规范全文审计，并增加静态合同断言。确认主实验、核心消融、跨骨干、训练复用、导出分组、MAGIC pending、Top-k 缺失状态和时间边界表述一致；未发现残余 A0--A6 / Semantic MLP 主表 / w/o Gate 正式消融表述。最终针对性检查为 41 passed，14 warnings（旧 cell id），无失败。

### Phase 5：Git 验收

已完成。最终分支仍为 `docs/reduce-paper-experiment-matrix`，HEAD 仍为起始值 `16e85bff1f0481a213181663a15a197cdbd9d454`。`git diff --diff-filter=D --name-status`、YAML diff 和核心 MSTC/时间实现 diff 均为空；Notebook 仅 24 insertions / 12 deletions，未大规模重写。最终测试为 41 passed，14 个既存 cell-id warnings，无失败。未 commit、push 或 merge。

## 最终删除与执行检查

- 删除 YAML：否
- 删除 Python：否
- 删除 Notebook：否
- 删除 tests：否
- 删除 artifacts：否
- 删除 Semantic MLP：否
- 删除 w/o Gate：否
- 删除 Top-k sensitivity：否；但专用 runner/config 在本轮开始前即不存在，已按“设计合同保留、实现缺失”记录
- 运行真实实验：否
- 下载/接入/运行 MAGIC：否
- 修改 MSTC 最终时间语义：否
