# MAGIC M5/M6 实现进度

- **分支**: feat/magic-score-evaluator
- **Base HEAD**: 16f1adf (M3/M4 frozen)
- **本轮起始时间**: 2026-09-06
- **目标**: M5 (MAGIC Raw Score Export) + M6 (MAGIC Unified Evaluator Integration)
- **完成状态**: **M5/M6 LOCAL PROTOCOL IMPLEMENTATION COMPLETE**

---

## A. Git 状态

```
branch: feat/magic-score-evaluator
HEAD: 16f1adfa64fbad9030497bfdd05cee26fd8d6b07
modified: src/baselines/magic/__init__.py
untracked: src/baselines/magic/scoring.py
untracked: src/baselines/magic/evaluator.py
untracked: tests/test_magic/test_magic_scoring.py
untracked: tests/test_magic/test_magic_evaluator.py
```

---

## B. 上一轮 WIP 审计

| 文件 | 状态 |
|---|---|
| `src/baselines/magic/__init__.py` | 已修改 - 新增 M5/M6 导出 |
| `src/baselines/magic/scoring.py` | 完整 - 实现 MAGICEntityScorer 等 |
| `src/baselines/magic/evaluator.py` | 完整 - 实现 MagicEvaluator 等 |
| `tests/test_magic/test_magic_scoring.py` | 完整 - 28 tests |

中断点: 创建 `test_magic_evaluator.py` 时被人工停止。
**所有 WIP 实现完整且正确，不重新生成。**

---

## C. M5 实现

### API

```python
class MAGICEntityScorer:
    def __init__(self, k=10, seed=0, max_train_reference_samples=50000)
    def fit(train_embeddings, train_node_ids) -> self
    def score(target_embeddings, target_node_ids) -> List[NodeScoreRecord]
    def score_snapshot(snapshot, embeddings, node_ids) -> List[NodeScoreRecord]
```

### 核心特性

1. **train-only fit**: fit() 只使用 train_embeddings
2. **THEIA k=10**: 默认 k=10 (MAGIC_K_NEIGHBORS)
3. **max 50k samples**: 最多采样 50,000 个 train embedding 计算 reference distance
4. **KNN raw score**: score_raw = dist_target / dist_reference
5. **canonical_node_id 保留**: NodeScoreRecord 只暴露 canonical_node_id
6. **deterministic**: 使用 np.random.seed 冻结
7. **non-finite fail fast**: NaN/inf 输入报错
8. **zero variance fail fast**: 常数 embedding 报错
9. **non-positive reference distance fail fast**: 零距离报错

### 不接受

- labels
- malicious nodes
- ground truth
- test data in fit()

---

## D. M5 Tests (28 tests)

```
tests/test_magic/test_magic_scoring.py
```

| 测试类 | 覆盖 |
|---|---|
| TestTrainOnlyFit (4) | fit 创建状态, reference distance, 空输入 |
| TestValidationIsolation (2) | validation 不改变 state |
| TestTestIsolation (1) | test 不改变 state |
| TestDeterminism (2) | deterministic, seed |
| TestScoreOrdering (1) | 高距离节点高分 |
| TestCanonicalIdentity (1) | canonical_node_id 保留 |
| TestIdentityMapping (2) | local index 不泄漏 |
| TestNonFiniteHandling (3) | NaN/inf 报错 |
| TestReferenceDenominator (2) | zero variance / zero reference 报错 |
| TestSnapshotBoundary (1) | snapshot 边界 |
| TestFutureIsolation (1) | future node 不可见 |
| TestMaxMerge (1) | max merge |
| TestLabelRejection (2) | API 无 labels |
| TestKNeighbors (2) | k 参数 |
| TestNodeScoreRecord (3) | 有限性校验 |

---

## E. M6 实现

### API

```python
class MagicEvaluator:
    def __init__(self, k=10, threshold_method="validation_quantile", threshold_quantile=0.999)
    def fit_threshold(validation_node_scores) -> ThresholdConfig
    def predict(test_node_scores) -> Dict[str, int]
    def evaluate(ground_truth, predictions, scores) -> Dict[str, Any]
```

### 三阶段分离

| 阶段 | 输入 | 输出 | 是否接收 labels |
|---|---|---|---|
| Stage A: fit_threshold | validation_scores | ThresholdConfig | ❌ |
| Stage B: predict | test_scores, threshold | predictions | ❌ |
| Stage C: evaluate | ground_truth, predictions, scores | metrics | ✅ |

### 复用项目指标

```python
from src.mstc.metrics import compute_classification_metrics
```

输出: tp, fp, tn, fn, precision, recall, f1, mcc, auprc, auroc, fpr

---

## F. Threshold 数据边界

```python
# Stage A: only validation scores
config = fit_magic_threshold(validation_node_scores, q=0.999)

# Stage B: test scores + frozen threshold
predictions = predict_magic(test_node_scores, config)
```

**禁止**:
- 接收 test labels
- 接收 test scores 用于 threshold fit
- 使用 MAGIC upstream `precision_recall_curve(y_test, score)` 行为

---

## G. Ground-truth 数据边界

- **Stage A** (fit_threshold): 不接收 ground truth
- **Stage B** (predict): 不接收 ground truth
- **Stage C** (evaluate): 唯一接收 ground truth 的阶段

```python
metrics = compute_magic_metrics(
    ground_truth=ground_truth,  # ONLY here
    predictions=predictions,
    scores=test_node_scores,
)
```

---

## H. Canonical Node Identity

```python
@dataclass
class NodeIdentityMap:
    _canonical_to_local: Dict[str, int]
    _local_to_canonical: Dict[int, str]
```

- DGL backend 使用 local contiguous index
- 统一 evaluator 使用 canonical_node_id
- 通过 `NodeIdentityMap` 提供可逆映射

**禁止**: 直接用 DGL local index 与 ground truth join

---

## I. Test Population

**必须**: canonical test split 中所有应评估节点都保留

```python
# 全部 test node 都进入预测
predictions = evaluator.predict(test_node_scores)
# predictions 包含所有 test_node_scores 中的节点
```

**禁止**:
- 根据 y_test 筛节点
- 根据 malicious flag 保留节点
- 根据异常分数提前裁剪

---

## J. Metrics

复用 `src/mstc/metrics.py::compute_classification_metrics`：
- TP, FP, TN, FN
- Precision, Recall, F1
- MCC, AUPRC, AUROC, FPR

AUROC/AUPRC 使用 **raw scores** 而非 binary predictions。

---

## K. Validation Benign-only 核实

依据 `src/config.py::DATASET_DEFAULT_CONFIG`:
- THEIA_E3: val = graph_9, date = 2018-04-09 → benign_only
- THEIA_E5: val = graph_11, date = 2019-05-11 → benign_only

**合同状态**: 由 canonical split 定义为 benign-only validation day。
运行时 evaluator 不读取 labels 来挑 benign nodes。

---

## L. 新增/修改文件

| 文件 | 状态 |
|---|---|
| `src/baselines/magic/__init__.py` | modified (新增 M5/M6 导出) |
| `src/baselines/magic/scoring.py` | new (M5) |
| `src/baselines/magic/evaluator.py` | new (M6) |
| `tests/test_magic/test_magic_scoring.py` | new (28 tests) |
| `tests/test_magic/test_magic_evaluator.py` | new (33 tests) |

---

## M. 测试结果

```
tests/test_magic/  →  124 passed in 4.05s
- M3 input_adapter: 32 tests
- M4 protocol: 31 tests
- M5 scoring: 28 tests
- M6 evaluator: 33 tests
```

---

## N. Adversarial Leakage Tests

通过 `verify_label_independence()` 验证:
- `threshold_a == threshold_b`
- `scores_a == scores_b`
- `predictions_a == predictions_b`
- `metrics_a != metrics_b`

测试场景:
1. 全部 0 vs 全部 1
2. 任意翻转
3. 不同 ground truth 不影响 score/prediction

---

## O. MAGIC Upstream SHA256 (前后一致)

| 文件 | SHA256 |
|---|---|
| `train.py` | `9003a2ec...` |
| `eval.py` | `299352ea...` |
| `model/eval.py` | `4761d648...` |
| `utils/loaddata.py` | `57a84d42...` |
| `requirements.txt` | `acbec74f...` |

**结论**: /opt/magic-upstream 未被修改。

---

## P. 依赖安装

**否**。本轮不安装任何新依赖（仅使用已存在的 numpy + sklearn）。

---

## Q. 真实 E3/E5 运行

**否**。本轮仅 synthetic 测试。

---

## R. MAGIC Training 运行

**否**。本轮不运行 MAGIC training。

---

## S. Blockers

| Blocker | 状态 |
|---|---|
| Validation benign-only | ✅ 已核实 |
| /opt/magic-upstream 完整性 | ✅ 未修改 |
| M3/M4 regression | ✅ 63 tests 继续通过 |
| DGL/Torch 1.12 安装 | ✅ 未安装 |

---

## T. 当前准确状态

> M5/M6 local protocol implementation complete.
> M7-M10 pending.

**不是**:
- ❌ MAGIC reproduction complete
- ❌ MAGIC end-to-end integrated
- ❌ MAGIC verified on E3/E5

---

## U. M7 下一步建议

**M7 — Seed Control**:
- Python / NumPy / Torch CPU/CUDA seed 控制
- DataLoader generator / worker seed
- Deterministic flags
- 每 seed 独立 artifacts

不要:
- 安装 DGL/Torch 1.12
- 运行真实 E3/E5
- 声称 MAGIC verified

---

## 禁止事项确认

- ✅ 不实现 MAGIC GAT/GMAE 模型
- ✅ 不安装 DGL/Torch 1.12
- ✅ 不修改 /opt/magic-upstream
- ✅ 不读取 ground truth labels (仅 Stage C)
- ✅ 不实现 M7-M10
- ✅ 不运行真实 E3/E5
- ✅ 不运行真实 MAGIC training
- ✅ 不 git commit/push (等待人工审计)
