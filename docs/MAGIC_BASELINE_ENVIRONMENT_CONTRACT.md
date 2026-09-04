# MAGIC 外部基线 M1/M2 来源与环境合同

状态：**M1/M2 CONTRACT FROZEN**
冻结日期：2026-09-04
适用名称：**MAGIC (Unified Protocol)**
依据：`docs/MAGIC_BASELINE_INTEGRATION_AUDIT.md`、`docs/MAGIC_BASELINE_IMPLEMENTATION_PLAN.md`

## 1. Purpose

本文档冻结 MAGIC 外部基线的 M1 source contract 与 M2 environment/runtime contract，并明确后续统一论文协议的边界。本次冻结不代表 MAGIC adapter 已实现、隔离容器已构建、依赖已安装或正式实验已运行。

本文档中的三类陈述必须区分：

- **已审计事实**：来自已完成的官方仓库、源码、许可证或当前项目审计；
- **已冻结决策**：本文为后续实现预先固定的统一协议与部署合同；
- **待实施/验证事项**：必须在后续阶段完成，不能因合同已冻结而宣称已经可运行。

## 2. Source Freeze

MAGIC source 正式冻结如下：

| 字段 | 冻结值 |
|---|---|
| Official repository | `https://github.com/FDUDSDE/MAGIC` |
| Pinned SHA | `aa0b647eea74b6faa0e52eb444370c4411a32cbe` |
| License | MIT |
| Source identity | 已审计的官方 repository snapshot |

上述 SHA 是已经审计的官方仓库快照。此前审计未发现与 USENIX Security 2024 论文对应的正式 tag/release，且该提交晚于会议时间，因此不得将它称为 paper-exact tag、paper-exact release 或会议提交时的精确源码快照。

所有正式实验必须记录：repository URL、完整 pinned SHA、license 和 adapter revision。构建或运行流程不得自动跟随 upstream `HEAD`、`main` 或其他浮动引用；若 source identity 不匹配，必须 fail-fast。

## 3. MAGIC Baseline Identity

### 3.1 保留的 MAGIC 原方法

以下属于 MAGIC 方法核心，统一接入不得用 ORTHRUS/MSTC 组件替换：

- GAT / GMAE architecture；
- node / edge type representation；
- masking；
- scaled-cosine feature reconstruction；
- sampled-structure BCE；
- benign-train KNN reference；
- normalized KNN anomaly score；
- method-specific core hyperparameters。

### 3.2 本论文的 Unified Protocol adapter

以下属于本文外围适配，不是 MAGIC 官方原始行为：

- canonical THEIA split；
- strict information boundary；
- validation-only threshold；
- causal snapshot inference boundary；
- node score merge；
- canonical artifact export；
- unified evaluator；
- seed control。

论文、配置和结果中应使用名称 **MAGIC (Unified Protocol)**。不得把 3.2 中的行为描述为 MAGIC official inference/evaluation protocol。

## 4. Local Development Boundary

当前本地 ORTHRUS/MSTC 开发环境的已确认实例信息为：

| 项目 | 值 |
|---|---|
| Environment | `pids` |
| Python | 3.9.25 |
| PyTorch | 1.13.1+cu117 |
| GPU | NVIDIA RTX 5070 Laptop GPU |

此前已确认，当前旧 PyTorch 不支持该 GPU 的 `sm_120` 正式 CUDA kernel。因此本地环境只承担：

- adapter development；
- protocol development；
- static checks；
- synthetic tests；
- 条件允许时的 CPU-level tests；
- artifact contract tests。

本地 `pids` 环境不是 MAGIC formal GPU experiment environment。不得为了 MAGIC 升级、降级或安装包到 `pids`，也不得用本地 GPU smoke 结果替代正式环境证据。

## 5. MSTC Formal Runtime

已经实际运行 MSTC-PIDS 的当前 formal experiment environment 实例如下：

| 项目 | 值 |
|---|---|
| Platform | Alibaba Cloud PAI-DSW |
| Python | 3.10.19 |
| PyTorch | 2.4.1+cu124 |
| PyG | 2.8.0.post1 |
| GPU | NVIDIA A10 |
| CUDA runtime | 12.4 |

这是已经实际运行 MSTC-PIDS 的正式环境实例，不表示 MSTC 在方法上强制要求 Python 3.10，也不构成 MAGIC 的运行环境。

## 6. MAGIC Formal Runtime Contract

MAGIC formal runtime contract 冻结为：

| 项目 | 冻结值 |
|---|---|
| Python | 3.8 |
| PyTorch | 1.12.1+cu116 |
| DGL | 1.0.0 |
| Runtime isolation | independent Docker container |
| Target | user-owned Linux GPU server |

MAGIC 不与 MSTC 共用 Python environment。不得在 MSTC 的 Python 3.10 / Torch 2.4.1 / PyG 2.8 环境中直接安装 MAGIC 的旧依赖。

这里冻结的是目标合同，而不是“环境已经构建并验证”的事实。Python 3.8 与旧版 Torch/DGL 带来 wheel 可得性、宿主驱动兼容性和安全维护风险；后续部署必须记录容器镜像 digest、完整依赖 inventory、宿主 NVIDIA driver、CUDA/DGL import 与最小 GPU smoke 结果。任何不兼容都必须显式失败或另行修订合同，不得静默改用较新依赖后仍声称遵循本合同。

## 7. Docker Isolation Model

推荐且已冻结的服务器结构为：

```text
Linux GPU server
├── NVIDIA Driver
├── Docker
├── NVIDIA Container Toolkit
├── MSTC container
│   ├── Python 3.10
│   ├── Torch 2.4.1
│   └── PyG 2.8
└── MAGIC container
    ├── Python 3.8
    ├── Torch 1.12.1+cu116
    └── DGL 1.0.0
```

两个容器只通过受控的 dataset/artifact volume 交换 canonical files；Python 环境、Torch 与 graph framework 完全隔离。共享 volume 不允许变成跨 split、跨 seed 或跨配置复用无 fingerprint cache 的旁路。

本轮不创建 Dockerfile、镜像、Python 3.8 环境或依赖 lock；这些均不属于本次合同冻结的已完成事实。

## 8. Unified Train/Validation/Test Boundary

统一信息边界固定为：

| Split | 允许行为 | 禁止行为 |
|---|---|---|
| Train | 训练 GMAE；构建 benign-train KNN reference；冻结 train-only schema | 选择 threshold；读取 test labels；让 validation/test 扩展 type vocabulary |
| Validation | 在冻结 checkpoint 与 train-only KNN reference 下生成 raw node scores；仅按固定协议计算 threshold | 进入 GMAE 参数训练或 KNN reference；使用 test 信息调参 |
| Test | 加载冻结模型、KNN reference、schema 与 threshold；生成最终 raw scores/predictions | 选择 checkpoint、threshold、hyperparameter 或改变 test-time model behavior |

canonical THEIA split、strict information boundary 与以上阶段权限属于 Unified Protocol adapter，不是 MAGIC 官方原始 split/validation 行为。

## 9. Threshold Contract

阈值合同正式固定为：

```yaml
method: validation_quantile
q: 0.999
```

具体语义如下：

- Train 不得选择 threshold；
- Validation 仅使用 validation normal-node score distribution 计算 `q=0.999` threshold；
- Test 只加载并应用已经冻结的 threshold；
- test labels、test metrics、test recall 和 test score 的评价结果不得参与 threshold selection。

这里的 normal-node universe 由预先冻结的 canonical benign validation split 定义，不得通过读取 validation ground-truth 标签或按标签过滤节点来构造。若未来数据审计发现 validation split 并非 benign-only，必须先停止并修订统一协议，不能临场使用标签筛选后继续声称满足本合同。

`q=0.999` 是预先固定的论文协议 operating point，不是根据 E3/E5 test 性能选择的最优值。正式 artifact 必须记录 method、q、参与计算的 validation score universe、比较规则与 threshold 数值。

## 10. Causal Snapshot Contract

正式推理采用 **canonical window-end causal snapshots**。对任一 snapshot，其模型输入只能包含该 canonical window end 时刻及以前按协议允许的信息，不得使用未来事件或未来拓扑。

同一 node 在多个有效 causal snapshot 中出现时，固定采用：

```yaml
node_merge: max
```

定义为：

```text
final_score(node)
  = max over all valid causal snapshot scores of that node
```

该规则必须以相同方式用于 validation 与 test，并且与标签无关。canonical window-end snapshot 与 `max` merge 是本文 Unified Protocol 对 MAGIC 的外围 inference adaptation，不是 MAGIC official inference protocol。

## 11. Ground Truth Contract

四个主实验模型统一使用当前项目相同的 strict node-level ground truth。MAGIC 官方标签、ThreaTrace 标签或其他 baseline 自带标签不得替代 canonical ground truth。

Ground truth 只能在 score、threshold 和 prediction 已经冻结后进入最终 unified evaluator。它不得进入：

- training；
- KNN reference construction；
- type/schema construction；
- threshold selection；
- test-time model behavior。

若发现 canonical train split 污染或标签/节点 universe 不一致，必须 fail-fast 并单独处理数据协议，不能使用 test ground truth 静默过滤训练图。

## 12. Artifact Contract

后续 MAGIC 每个正式 run 至少必须保留：

- `config_resolved.yml`；
- `environment.json`；
- `runtime.json`；
- raw node scores（包括可追溯的 causal snapshot/node score）；
- `node_predictions.csv`；
- `metrics.json`。

Raw anomaly scores 是强制 artifact，不能只保存 `0/1` prediction。`node_predictions.csv` 必须同时保留 canonical node identity、合并后的连续 score 与冻结 threshold 下的 prediction；ground truth 仅由最终 evaluator 附加。

目标数据流冻结为：

```text
canonical artifacts
  → MAGIC adapter
  → pinned MAGIC backend
  → raw anomaly score
  → unified protocol
  → canonical artifacts
```

`environment.json` 还必须记录 source URL、完整 pinned SHA、MIT license、adapter revision、容器 digest、Python/Torch/DGL/CUDA 与 seed/determinism 信息。cache 或中间结果必须绑定 source、adapter、config、checkpoint、input 和 seed identity。

## 13. Forbidden Behaviors

后续实施与正式运行禁止：

- 自动跟随 MAGIC upstream `HEAD` / `main`；
- 把 pinned snapshot 称为 paper-exact tag/release；
- 把 Unified Protocol adapter 行为称为 MAGIC 官方行为；
- 在当前 `pids` 或 MSTC formal environment 中安装 MAGIC 老依赖；
- 让 MAGIC 与 MSTC 共用 Python/Torch/graph framework environment；
- 用 train 选择 threshold，或用 test labels、test metrics、test recall 调参；
- 使用 whole-test future topology 破坏 causal snapshot boundary；
- 用 MAGIC/ThreaTrace labels 替换 canonical ground truth；
- 只导出二值预测而丢弃 raw anomaly scores；
- 跨 run 复用无完整 identity/fingerprint 的 cache；
- 在合同未修订时静默改变 frozen source、runtime、threshold 或 node merge。

## 14. M1 Acceptance Criteria

本轮 M1 source freeze 在满足以下条件后记为 **frozen**：

- official repository、完整 SHA 与 MIT license 已明确记录；
- source 被准确表述为“已审计的官方 repository snapshot”；
- 无正式 paper tag/release 的边界已明确；
- future formal run 的 repository URL、pinned SHA、license、adapter revision 记录义务已冻结；
- 禁止浮动 upstream ref 已写入合同。

本轮未实现 machine-readable source manifest 或 backend checkout 校验；它们属于后续 adapter/runner 实施验收，不影响 M1 合同冻结状态，也不得被误报为已完成实现。

## 15. M2 Acceptance Criteria

本轮 M2 environment/runtime contract 在满足以下条件后记为 **frozen**：

- local development boundary 已明确，且 `pids` 保持不变；
- 已实际运行 MSTC-PIDS 的 DSW formal environment 实例已准确记录；
- MAGIC Python 3.8 / Torch 1.12.1+cu116 / DGL 1.0.0 合同已固定；
- user-owned Linux GPU server + independent Docker container 隔离模型已固定；
- MSTC/MAGIC 只共享受控 dataset/artifact volume 的边界已固定；
- threshold、causal snapshot、node merge、ground truth 与 artifact 合同已冻结。

本轮未创建或验证 MAGIC container，未安装依赖，也未运行 import/GPU smoke；因此只能声明 M2 合同已冻结，不能声明 MAGIC formal runtime 已部署或 operational。

## 16. Next Stage Boundary

M1/M2 合同冻结后，M3-M10 仍未完成。本轮不得进入 M3-M8，不得编写 MAGIC adapter、创建 Dockerfile、安装依赖或运行真实实验。

下一阶段的唯一建议是：从当前经人工审计的状态新建 `feat/magic-unified-adapter`，再进入 M3-M8 的本地 adapter/protocol/static/synthetic/CPU/artifact-contract 实现。M9 pilot 与 M10 formal runs 仍需各自独立门禁和人工批准。
