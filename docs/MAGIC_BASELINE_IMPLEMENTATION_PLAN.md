# MAGIC 外部基线统一协议实施计划

状态：**PLAN ONLY / NOT IMPLEMENTED；M1/M2 CONTRACT FROZEN**
前置审计：`docs/MAGIC_BASELINE_INTEGRATION_AUDIT.md`
审计结论：**CONDITIONAL GO**
锁定 upstream 候选：`FDUDSDE/MAGIC@aa0b647eea74b6faa0e52eb444370c4411a32cbe`
M1/M2 冻结合同：`docs/MAGIC_BASELINE_ENVIRONMENT_CONTRACT.md`

## 0. 范围与硬边界

本计划描述后续实施，不代表 MAGIC 已接入、已测试或已有结果。后续实现必须遵守：

- MAGIC 是 `external baseline backend`，不是 MSTC/ORTHRUS model variant；
- 不进入 `src/factory.py`，不使用 MSTC calibration、MSTC Top-k、ORTHRUS Word2Vec 或 ORTHRUS checkpoint；
- 保留官方 GAT/GMAE、mask、reconstruction loss、KNN 与 raw anomaly-score 核心；
- 统一 dataset、split、信息边界、ground truth、threshold 选择、metrics、seeds 和 artifacts；
- test 只做冻结模型/阈值下的最终评价；
- 不把 upstream 源码、数据或 checkpoint vendor 到本项目；
- 每个高成本阶段都必须显式授权，M8 通过前不运行 pilot，M9 通过前不运行正式 6-run 矩阵。

依据：审计文档；这些是实施门禁。

## 1. 已冻结的两个协议选择

以下选择已由 `docs/MAGIC_BASELINE_ENVIRONMENT_CONTRACT.md` 正式冻结；后续实现不得依据 test 结果改动：

1. **Threshold**：`validation_quantile(q=0.999)`，只用 normal validation node scores，test 不参与；比较规则复用当前统一 threshold helper。
2. **Causal graph/output**：以 canonical time window end 为推理 cutoff，snapshot 只含该 cutoff 以前允许的信息；每个 `(node_id, window)` 保存 raw MAGIC score；同一 node 跨窗口的正式 node score 用预注册、无标签的 `max` 归并。`max` 表示“一次显著异常即把该 entity 列为可疑”，但它是外围 output policy，不是 MAGIC 模型核心。

如果要求严格到每个 event 的即时因果边界，M3/M4 应改为 event-prefix snapshot；该方案成本显著更高，必须先在 M8 测量，不能在运行中静默切换。依据：审计 `inference`。

## 2. 推荐目录与改动面

后续优先新增：

```text
src/baselines/
  magic/
    __init__.py
    source_manifest.py
    input_adapter.py
    protocol.py
    backend.py
    seed.py
    raw_output.py
    output_adapter.py
    runner.py
tests/baselines/magic/
config/experiments/magic.yml        # M4 才创建
containers/magic/                   # 或独立外部 image recipe
```

现有文件只做最小接线：

- `src/experiments/run_experiment.py`：按显式 `experiment_backend=magic_external` 分发，默认 ORTHRUS 路径完全不变；
- `src/experiments/run_matrix.py`：原则上保持 scheduler 与 config-id 逻辑不变；仅在确有必要时补 backend metadata；
- `src/experiments/collect_results.py`：artifact schema 已兼容，优先不改；
- `src/experiments/export_tables.py`：已识别 config stem `magic`，优先不改；
- Notebook：M8 通过后才添加独立显式开关，不把 MAGIC 混入当前 in-repo runner。

不得修改：`src/model.py`、MSTC encoder/decoder/calibration/aggregation 数学实现、现有正式结果。依据：current project + audit architecture。

## 3. 数据与输出架构

```text
canonical split artifacts
  [src,dst,t,src_type,dst_type,edge_type_index,global_event_index]
                │
                ▼
        MAGIC input adapter
  [frozen type schema, causal DGL snapshots, global↔local map]
                │
                ▼
 pinned original MAGIC GMAE + KNN backend
                │
                ▼
   raw_magic per-window node scores
                │
                ▼
 validation-only threshold + label-free node merge
                │
                ▼
 canonical evaluator attaches test ground truth
```

目标 artifact：

```text
<artifact_root>/
  matrix_artifacts/
    <magic-config-id>/
      <THEIA_E3-or-THEIA_E5>/
        runs/
          magic/
            seed_<seed>/
              config_resolved.yml
              environment.json
              runtime.json
              checkpoints/
                magic_final.pt
              raw_magic/
                source_manifest.json
                graph_manifest.json
                local_global_node_map.csv
                validation_node_window_scores.csv
                test_node_window_scores.csv
                knn_reference_manifest.json
                threshold.json
              node_scores/
                metrics.json
                node_predictions.csv
```

`node_predictions.csv` 至少包含 `node_id,score,y_hat,y_true`；`score` 必须是原始归一化 KNN distance 经预注册 node merge 后的值。raw per-window scores 不得删除。`environment.json` 还应记录 upstream URL/SHA/license、wrapper SHA、container digest、Python/Torch/DGL/sklearn/CUDA、seed 与 deterministic flags。依据：current artifact contract + audit。

## 4. Config identity

后续创建独立 `config/experiments/magic.yml`，建议至少包含：

```yaml
experiment_backend: magic_external
model:
  variant: magic
experiment_identity:
  semantics_version: magic_unified_v1
  upstream_commit: aa0b647eea74b6faa0e52eb444370c4411a32cbe
magic:
  graph_protocol: canonical_window_causal_v1
  node_merge: max
  threshold_method: validation_quantile
  threshold_quantile: 0.999
```

该 YAML 不能伪装成 `baseline.yml`，不能继承出一个实际上由 ORTHRUS model factory 运行的模型。config-id 必须覆盖 upstream SHA、graph/input semantics、threshold、node merge 与重要 MAGIC hyperparameters；更改其中任一项应生成不同 artifact identity。上述只是计划示意，本轮未创建 YAML。依据：current `_config_id` contract + audit。

## Phase M1 — source freeze（合同已冻结；实现未开始）

**冻结状态**：M1 source contract 已冻结，见 `docs/MAGIC_BASELINE_ENVIRONMENT_CONTRACT.md`。这表示 official repository、完整 SHA、MIT license、非 paper-exact 边界和禁止浮动 upstream ref 已定稿；不表示 machine-readable manifest 或 backend 校验已经实现。

- **修改什么**：实现 source manifest 校验；锁定 repository URL、full SHA、MIT license hash；构建阶段只接受该 SHA。
- **不修改什么**：不 fork、不改 upstream GAT/GMAE/KNN 代码，不 vendor data/checkpoints。
- **输入**：官方 URL、`aa0b647...`、LICENSE、README、requirements。
- **输出**：machine-readable `source_manifest.json`；可复现的 read-only checkout/image layer。
- **测试**：错误 URL/SHA/license hash fail-fast；dirty upstream checkout fail-fast；无 tag 时不得退回 `main` 浮动 ref。
- **风险**：当前 SHA 晚于会议且无 paper tag；只能称“audited official repository snapshot”，不能称 paper-exact snapshot。
- **完成条件**：离线查看 artifact 即能确认 upstream 与 wrapper 身份，且项目工作树不含第三方源码。

## Phase M2 — environment isolation（合同已冻结；环境未创建）

**冻结状态**：M2 environment/runtime contract 已冻结，见 `docs/MAGIC_BASELINE_ENVIRONMENT_CONTRACT.md`。本地仅用于 development/static/synthetic/CPU/artifact-contract tests；正式 MAGIC 目标为 user-owned Linux GPU server 上的 isolated Docker。当前未创建 Python 3.8 环境、未安装依赖、未构建容器，也未完成 import/GPU smoke。

- **修改什么**：在 user-owned Linux GPU server 上建立独立 Docker container；锁 Python 3.8、Torch 1.12.1+cu116、DGL 1.0.0、sklearn 1.2.2，并补齐 NumPy/NetworkX/tqdm 的实际可运行锁版本。
- **不修改什么**：不升级/降级 `/home` 当前 Python、Torch、PyG、sklearn；不改变 ORTHRUS image。
- **输入**：M1 source manifest、upstream requirements、可用 CUDA/CPU runtime。
- **输出**：image digest/lockfile、dependency inventory、import-only health report。
- **测试**：CPU import smoke；若用 GPU，做 CUDA/DGL import 与 tiny tensor smoke；不加载 THEIA、不训练。
- **风险**：旧 cu116 wheel 与 host driver、DGL wheel availability；requirements 漏依赖。
- **完成条件**：独立 Docker 环境可导入所有 upstream modules，退出后当前 ORTHRUS/MSTC 环境版本完全未变。

## Phase M3 — THEIA input adapter

- **修改什么**：从 canonical artifacts 读取 `src/dst/t/src_type/dst_type/edge_type_index/split/global_event_index`；实现官方方向/去重语义、frozen type schema、DGL graph 与 global↔local mapping；同时覆盖 E3/E5。
- **不修改什么**：不读取 ORTHRUS Word2Vec `x_src/x_dst/msg`；不读取 ground truth；不下载/重新解析 DARPA raw logs；不训练。
- **输入**：当前 canonical train/val/test artifacts 与 dataset config。
- **输出**：确定性 graph manifests、causal snapshots、node mapping sidecars；不产生性能结果。
- **测试**：split isolation、timestamp/global_event_index 顺序、方向规则、simple-edge policy、mapping round-trip、未知 type fail/UNK、E3/E5 synthetic fixtures；monkeypatch ground-truth loader 使任何提前访问立即失败。
- **风险**：paper 与源码的 multi-edge 处理不一致；snapshot 粒度影响语义/成本；同 node 跨窗口重复。
- **完成条件**：相同输入/seed 的 graph hash 稳定，val/test 不改变 train graph/schema，未来事件不进入较早 snapshot。

## Phase M4 — unified split/protocol adapter

- **修改什么**：创建独立 `magic.yml`；实现 train-only GMAE/KNN fit、validation-only threshold、test-evaluation-only 门禁；冻结 fixed 50 epochs/final checkpoint；实现 explicit backend dispatch。
- **不修改什么**：不启用官方 test-label threshold；不让 validation 进入 GMAE/KNN；不启用 test-time model adaptation；不改 model factory。
- **输入**：M3 snapshots、用户确认的 threshold/causal/node-merge 规则、seed。
- **输出**：resolved protocol config、frozen checkpoint identity、threshold metadata。
- **测试**：test labels/test scores 传给 selector 必须在类型/签名上不可达；交换 test labels 不改变 checkpoint/threshold；val 变化可改变 threshold，但不能改变训练 checkpoint；ORTHRUS dispatch regression。
- **风险**：不恰当的 generic dispatcher 可能影响现有 runner；fixed epoch 在新 E5 上可能不是最优，但禁止 test tuning。
- **完成条件**：数据使用矩阵可由自动测试证明为 `train→fit, val→select, test→evaluate`，无隐式 fallback 到官方 `eval.py`。

## Phase M5 — raw prediction export

- **修改什么**：包装 upstream `embed`/KNN 路径，输出 train reference manifest、validation/test per-window raw scores 与 node maps；所有 cache 加 config/checkpoint/input fingerprint。
- **不修改什么**：不 threshold、不附 test label、不计算正式指标；不复用 upstream 无 fingerprint 的 `distance_save_*.pkl`。
- **输入**：M4 final checkpoint、train reference、val/test snapshots。
- **输出**：`raw_magic/*.csv|json` 和 raw score checksum。
- **测试**：score 公式与 upstream tiny fixture 一致；row count/map/checksum；跨 seed/config cache 拒绝；raw 输出无 `y_true`。
- **风险**：KNN 内存/时间，重复 node、空 graph、零 train std/zero mean distance。
- **完成条件**：每个 canonical node/window 都能追溯到输入 snapshot 与原始 score，且无标签参与。

## Phase M6 — unified evaluation adapter

- **修改什么**：在 validation raw score 上算冻结 threshold；按预注册规则归并 test node score；预测固定后才附当前 ground truth；调用 canonical metrics 写 artifacts。
- **不修改什么**：不调用 `precision_recall_curve(y_test, score)` 选阈值；不改 MAGIC loss/score；不使用 ThreaTrace/MAGIC labels。
- **输入**：M5 raw outputs、current validation/test ground truth（分阶段权限）、threshold config。
- **输出**：`threshold.json`、`node_scores/node_predictions.csv`、`metrics.json`。
- **测试**：Recall/MCC/AUPRC/AUROC/FPR fixture；test label permutation 不改变 score/y_hat/threshold；validation threshold exactness；重复 node merge；undefined metric 为 NaN 的 canonical behavior。
- **风险**：validation 无/少正常节点；`>` 与 `>=` 边界；node universe 不一致。
- **完成条件**：canonical collector 能无特判读取一条 synthetic MAGIC run，且 threshold provenance 明确为 validation-only。

## Phase M7 — seed control

- **修改什么**：CLI/config 接收 seed；设置 Python、NumPy、Torch CPU/CUDA、DGL、DataLoader/worker generator；正确设置 deterministic flags 并记录环境。
- **不修改什么**：不把一个 checkpoint 复制成三个 seed；不把官方 100 次 evaluation shuffle 当 3 个独立训练 seed。
- **输入**：seed `0,1,2` 与 deterministic policy。
- **输出**：每 seed 独立 checkpoint/KNN/threshold/raw/canonical artifacts；RNG manifest。
- **测试**：同 seed tiny run graph/mask/sample/score checksum 重复；不同 seed至少一个 stochastic trace不同；缺少 DGL seed hook fail-fast或显式降级记录。
- **风险**：DGL/GPU kernels 不保证 bitwise deterministic；多线程 KNN顺序差异。
- **完成条件**：可复现等级被记录且没有硬编码 seed 0 路径。

## Phase M8 — smoke tests

- **修改什么**：增加纯 synthetic tiny provenance fixtures、bounded end-to-end smoke 与 Notebook/CLI 显式开关。
- **不修改什么**：不访问真实 THEIA；不跑 E3/E5；不自动下载数据/依赖；不生成论文结果。
- **输入**：M1–M7 完整 adapter、tiny train/val/test fixtures。
- **输出**：一套非正式 `is_smoke=true` canonical artifacts 与测试报告。
- **测试**：CPU E2E；可选 GPU E2E；no-future/no-label leakage；artifact schema；resume/cache identity；current ORTHRUS targeted regressions。
- **风险**：synthetic graph 太小无法覆盖 KNN k=10、negative sampling 与空 type；需专门构造边界 fixture。
- **完成条件**：所有 adapter tests 通过；smoke 被 collector 默认排除；工作树无真实结果/第三方数据。

## Phase M9 — E3 seed0 pilot

- **修改什么**：在用户单独授权、环境和数据就绪后，只跑 `THEIA_E3 × MAGIC × seed0`；记录真实资源与协议 trace。
- **不修改什么**：不跑 E5，不跑 seeds 1/2，不在看见 test 指标后改 threshold/hyperparameter/node merge；不覆盖历史 artifacts。
- **输入**：冻结 M8 release、canonical E3 artifacts、隔离 image、明确资源上限。
- **输出**：一个 pilot run 的完整 raw/canonical artifacts、资源报告与 GO/NO-GO gate；结果标为 pilot。
- **测试**：artifact validator；train/val/test access log；checkpoint/threshold provenance；重复 seed0 的小样本一致性；collector/export dry run。
- **风险**：OOM、KNN CPU瓶颈、官方 simple graph 语义在 canonical windows 下退化、node coverage不足。
- **完成条件**：无泄漏、无 OOM/静默降级、指标/资源可解释；用户复核后才可进入 M10。

## Phase M10 — formal E3/E5 3-seed runs

- **修改什么**：仅使用 M9 已冻结的 code/config/image，调度 E3/E5 × seeds 0/1/2 共 6 个 MAGIC runs；收集 mean±std。
- **不修改什么**：不在 run 间改协议；不挑 best seed；不把失败 run 排除后伪称 3-seed 完整；不顺手跑消融/跨骨干。
- **输入**：M9 approved release、两数据集 canonical artifacts、正式资源预算。
- **输出**：6 个独立 canonical run、run-status、`all_runs.csv`、`main_results.csv` 中真实 MAGIC 行。
- **测试**：每 run source/config/data hash；三 seed完整性；失败可恢复；export mean/std 与原始 rows 对账；test labels 只在 evaluator access log 出现。
- **风险**：E5 是新适配而非官方支持；跨 seed资源波动；部分 run失败导致不完整统计。
- **完成条件**：6/6 状态明确、所有 artifacts/schema/provenance 检查通过、无 test tuning，并经人工审阅后才可用于论文。

## 5. 阶段依赖与停止条件

```text
M1 → M2
  ↘
   M3 → M4 → M5 → M6 → M7 → M8 → [人工批准] → M9 → [人工批准] → M10
```

任一阶段发现必须修改 MAGIC core、无法实现 causal boundary、无法在 E5 构造等价输入、或无法隔离环境，应停止并把结论降为 NO-GO；不得通过使用 test labels、放宽 split 或更换 ground truth 来“跑通”。

## 6. 实施验收总表

| 验收项 | 必须满足 |
|---|---|
| source | full SHA 锁定、license记录、无浮动 main |
| architecture | original GAT/GMAE/mask/loss/KNN score |
| data | canonical E3/E5 split；train fit only |
| validation | threshold/diagnostics only |
| test | frozen inference/evaluation only |
| threshold | no test labels；provenance可追溯 |
| ground truth | current strict node-level set；预测后附加 |
| time | future event不改变较早 snapshot/score |
| seed | 0,1,2 独立训练和 artifacts |
| environment | isolated，不改变当前 ORTHRUS 环境 |
| outputs | raw outputs保留 + canonical artifacts齐全 |
| reporting | mean±std；不挑 best seed；失败显式 |

## 7. 当前最近下一步

M1/M2 合同已冻结为 `validation_quantile(q=0.999)`、`canonical window-end causal snapshot + max node merge`，以及独立 Docker runtime 合同。经本分支人工审计后，下一步新建 `feat/magic-unified-adapter`，只进入 M3-M8 的本地适配实现；不得直接跳到 M9/M10。M3-M10 当前均未标记为完成。
