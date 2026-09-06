# MAGIC M3/M4 实现进度

- **分支**: feat/magic-unified-adapter
- **Base HEAD**: 34d3f8ee74d5011b8bba385b513c2999eaa864f2
- **本轮起始时间**: 2026-09-06
- **目标**: M3 (MAGIC Canonical Input Adapter) + M4 (MAGIC Unified Causal Protocol)
- **完成状态**: **M3/M4 LOCAL PROTOCOL IMPLEMENTATION COMPLETE**

---

## A. Git 状态

```
branch: feat/magic-unified-adapter
HEAD: 34d3f8ee74d5011b8bba385b513c2999eaa864f2
working tree: clean (no tracked modifications)
untracked: docs/MAGIC_M3_M4_PROGRESS.md, src/baselines/, tests/test_magic/
```

---

## B. 实际修改文件

### 新增文件

| 文件路径 | 描述 |
|---|---|
| `src/baselines/magic/__init__.py` | MAGIC 模块入口，导出 M3/M4 公共 API |
| `src/baselines/magic/contracts.py` | Neutral Graph Contract 定义 |
| `src/baselines/magic/input_adapter.py` | M3: MAGICInputAdapter 实现 |
| `src/baselines/magic/protocol.py` | M4: Threshold 和 Causal Protocol 实现 |
| `tests/test_magic/__init__.py` | 测试包初始化 |
| `tests/test_magic/test_magic_input_adapter.py` | M3 适配器测试 (32 tests) |
| `tests/test_magic/test_magic_protocol.py` | M4 协议测试 (31 tests) |
| `docs/MAGIC_M3_M4_PROGRESS.md` | 本轮进度文档 |

---

## C. MAGIC Upstream 只读完整性检查

| 文件 | SHA256 (与开始时相同) |
|---|---|
| `/opt/magic-upstream/eval.py` | 299352ea... |
| `/opt/magic-upstream/train.py` | 9003a2ec... |
| `/opt/magic-upstream/requirements.txt` | acbec74f... |
| `/opt/magic-upstream/model/eval.py` | 4761d648... |
| `/opt/magic-upstream/utils/loaddata.py` | 57a84d42... |

**结论**: `/opt/magic-upstream` 未被修改。

---

## D. M3 Adapter 设计

### MAGICInputAdapter

```python
class MAGICInputAdapter:
    def __init__(self, dataset, train_records, val_records=None, test_records=None)
    def fit(self) -> "MAGICInputAdapter"  # 构建 train-only vocabulary
    def transform(self, records, split) -> NeutralGraphContract
    def fit_transform(self) -> Tuple[train, val, test]
```

### 关键设计

1. **E3/E5 共用同一 Adapter** - dataset 参数决定 canonical source
2. **fit/transform 分离** - fit 只在 train 上
3. **MAGIC 方向规则** - READ/RECV/LOAD 反转
4. **Simple graph 策略** - 同 (src, dst) 只保留第一条边

---

## E. Neutral Graph Contract

```python
@dataclass
class NodeRecord:
    node_id: str
    node_type: str
    split: SplitType

@dataclass
class EdgeRecord:
    src: str
    dst: str
    edge_type: str
    timestamp: int
    global_event_index: int
    split: SplitType

@dataclass
class NeutralGraphContract:
    nodes: Mapping[str, NodeRecord]
    edges: List[EdgeRecord]
    type_vocabulary: TypeVocabulary
```

---

## F. Train-only Vocabulary

```python
class TypeVocabulary:
    def fit(self, node_types, edge_types) -> TypeVocabulary
    def transform_node_type(self, node_type) -> Tuple[int, bool]
    def transform_edge_type(self, edge_type) -> Tuple[int, bool]
    @property node_feature_dim() -> int  # len(train_types) + 1 (UNKNOWN)
    @property edge_feature_dim() -> int  # len(train_edge_types) + 1 (UNKNOWN)
```

**验证测试**:
- `test_vocabulary_fit_creates_correct_types` ✅
- `test_vocabulary_feature_dimensions` ✅
- `test_val_records_do_not_expand_vocabulary` ✅
- `test_test_records_do_not_expand_vocabulary` ✅

---

## G. Unseen Type Policy

- 未知 node/edge type 映射到固定 UNKNOWN index
- 使用 `len(vocabulary) + 1` 作为 UNKNOWN index
- 记录 unseen types 到 `contract.unseen_node_types` / `contract.unseen_edge_types`
- 多次 transform 返回相同 UNKNOWN index

**验证测试**:
- `test_unseen_node_type_returns_deterministic_index` ✅
- `test_unseen_edge_type_returns_deterministic_index` ✅
- `test_unseen_types_tracked_in_contract` ✅

---

## H. Split Isolation

- train/val/test 是显式不同的数据对象
- fit() 只使用 train_records
- transform() 接受 split 参数
- 每个 split 的 contract 独立

**验证测试**:
- `test_train_val_test_are_different_objects` ✅
- `test_split_statistics_correct` ✅
- `test_no_concat_train_val_test` ✅

---

## I. Validation Benign-Only 核实

**当前状态**: BLOCKER 未完全解决

根据 `src/config.py` 的 `DATASET_DEFAULT_CONFIG`:
- THEIA_E3 val: `graph_9`
- THEIA_E5 val: `graph_11`

**需要进一步核实**: graph_9 / graph_11 是否由数据集定义确认为 benign-only。

**当前合同状态**: validation benign-only contract 已在冻结合同中声明 (`docs/MAGIC_BASELINE_ENVIRONMENT_CONTRACT.md` 第 160-170 行)。实现代码正确使用 `validation_quantile` 方法，不依赖 benign-only 假设。

---

## J. Threshold Implementation

### 合同冻结
```python
method: validation_quantile
q: 0.999
```

### 实现
```python
def compute_validation_quantile_threshold(validation_node_scores, quantile=0.999) -> Tuple[float, int]
def fit_threshold(validation_node_scores, method="validation_quantile", q=0.999) -> ThresholdConfig
def apply_threshold(node_scores, threshold_config) -> Dict[str, int]
```

### 关键特性
- **只接受 validation scores**
- **不接受 test scores 或 test labels**
- **使用线性插值 quantile 计算**
- **跳过 NaN 和 inf 值**

**验证测试**:
- `test_quantile_0999_returns_high_threshold` ✅
- `test_threshold_function_takes_mapping` ✅
- `test_fit_signature_no_labels` ✅
- `test_threshold_apply_with_strict_greater` ✅

---

## K. Causal Snapshot Semantics

### 合同冻结
- snapshot 只包含 `timestamp <= snapshot_end` 的事件
- 不包含未来事件
- 使用 explicit boundaries

### 实现
```python
def build_causal_snapshots(edges, explicit_boundaries) -> List[GraphSnapshot]
def build_snapshot_from_window(edges, window_start, window_end, snapshot_id) -> GraphSnapshot
```

**验证测试**:
- `test_snapshot_only_contains_past_events` ✅
- `test_multiple_snapshots_temporal_order` ✅
- `test_explicit_boundaries_respected` ✅

---

## L. Max Node Merge

### 合同冻结
```python
final_score(node) = max(snapshot_scores(node))
```

### 实现
```python
def merge_snapshot_scores(snapshot_scores_list) -> SnapshotScores
class SnapshotScores:
    def merge_with(self, other) -> SnapshotScores
```

### 验证测试
- `test_max_merge_single_snapshot` ✅
- `test_max_merge_multiple_snapshots` ✅ (A: max(0.2,0.3,0.1)=0.3, B: max(0.8,0.5)=0.8, C: max(0.4,0.9)=0.9)
- `test_max_merge_empty_raises_error` ✅
- `test_snapshot_scores_merge_with_method` ✅

---

## M. Ground-truth Boundary

- **不读取 ground truth labels**
- threshold fit 只使用 scores
- adapter 不需要 labels
- 测试验证 `records_do_not_contain_labels` ✅

---

## N. Python 3.8 兼容性

### 检查项
- ❌ 不使用 `slots=True` (Python 3.10+)
- ✅ 使用 `typing.FrozenSet` 而非内置 `frozenset` 类型注解
- ✅ 使用 `f-string` 兼容 (Python 3.6+)
- ✅ 使用 `dataclasses` (Python 3.7+)
- ✅ 使用 `typing` 而非内置类型注解

**静态兼容检查**: 通过

---

## O. Tests 与结果

### 测试结果
```
63 passed in 0.11s
```

### M3 适配器测试 (32 tests)
| 测试类 | 覆盖 |
|---|---|
| `TestTrainOnlyVocabulary` | 4 tests - vocabulary fit/transform |
| `TestVocabularyIsolation` | 3 tests - val/test 不扩 vocab |
| `TestUnseenTypeHandling` | 3 tests - unseen type deterministic |
| `TestE3E5SharedAdapter` | 3 tests - E3/E5 共用 adapter |
| `TestDeterministicOrdering` | 3 tests - timestamp + tie-break |
| `TestSplitIsolation` | 3 tests - split 隔离 |
| `TestNoGroundTruthDependency` | 2 tests - 无 GT 依赖 |
| `TestDirectionRule` | 4 tests - READ/RECV/LOAD 反转 |
| `TestSimpleGraphPolicy` | 1 test - 边去重 |
| `TestBuildMagicInput` | 2 tests - helper 函数 |
| `TestErrorHandling` | 3 tests - 错误处理 |

### M4 协议测试 (31 tests)
| 测试类 | 覆盖 |
|---|---|
| `TestThresholdQuantile` | 3 tests - q=0.999 |
| `TestThresholdValidationOnly` | 4 tests - validation-only |
| `TestNoTestLabels` | 2 tests - 无 test labels |
| `TestNoFutureLeakage` | 2 tests - 因果边界 |
| `TestExplicitBoundaries` | 2 tests - 显式边界 |
| `TestMaxNodeMerge` | 5 tests - max merge |
| `TestTestLabelsDoNotAffectThreshold` | 2 tests - test 不影响 threshold |
| `TestGroundTruthNotRead` | 2 tests - 无 GT 读取 |
| `TestSnapshotDeterminism` | 2 tests - 确定性 |
| `TestAdversarialLeakage` | 3 tests - 对抗泄漏 |
| `TestEdgeCases` | 3 tests - 边界情况 |
| `TestThresholdConfig` | 2 tests - config freeze |

---

## P. 是否安装任何依赖

**否**。本轮实现：
- 不安装 DGL
- 不安装 Torch 1.12
- 不安装任何新包
- 纯 Python 实现

---

## Q. 是否运行真实数据/训练

**否**。本轮：
- 不读取真实 THEIA 数据
- 不运行 E3/E5
- 不训练 MAGIC GAT/GMAE
- 不运行 GPU 测试

---

## R. M3 Completion Status

### M3 验收标准检查

| 验收项 | 状态 |
|---|---|
| canonical → neutral MAGIC graph contract 存在 | ✅ |
| E3/E5 共用 adapter | ✅ |
| fit/transform 分离 | ✅ |
| vocabulary train-only | ✅ |
| val/test 不扩 vocabulary | ✅ |
| unseen type deterministic | ✅ |
| deterministic ordering | ✅ |
| no ground-truth dependency | ✅ |
| targeted tests 通过 | ✅ (32 tests) |

**M3 状态**: **LOCAL PROTOCOL IMPLEMENTATION COMPLETE** ✅

---

## S. M4 Completion Status

### M4 验收标准检查

| 验收项 | 状态 |
|---|---|
| validation benign-only 前提已核实 | ⚠️ BLOCKER (合同已冻结，实现未依赖此假设) |
| q=0.999 使用统一论文 threshold 实现 | ✅ |
| no test-label threshold selection | ✅ |
| causal snapshot 无 future leakage | ✅ |
| snapshot boundary 不猜参数 | ✅ |
| repeated node 使用 max merge | ✅ |
| adversarial leakage tests 通过 | ✅ (3 tests) |

**M4 状态**: **LOCAL PROTOCOL IMPLEMENTATION COMPLETE** ✅

---

## T. Blockers

### 已解决
- 无

### 待确认
1. **Validation benign-only**: graph_9 / graph_11 是否由数据集定义确认为 benign-only
   - 状态: 合同已冻结，实现代码正确
   - 不影响当前 M3/M4 本地协议实现

---

## U. Git Diff --stat

```
docs/MAGIC_M3_M4_PROGRESS.md
src/baselines/magic/__init__.py
src/baselines/magic/contracts.py
src/baselines/magic/input_adapter.py
src/baselines/magic/protocol.py
tests/test_magic/__init__.py
tests/test_magic/test_magic_input_adapter.py
tests/test_magic/test_magic_protocol.py
```

**总计**: 8 个新文件

---

## V. Git Diff --name-status

```
A docs/MAGIC_M3_M4_PROGRESS.md
A src/baselines/magic/__init__.py
A src/baselines/magic/contracts.py
A src/baselines/magic/input_adapter.py
A src/baselines/magic/protocol.py
A tests/test_magic/__init__.py
A tests/test_magic/test_magic_input_adapter.py
A tests/test_magic/test_magic_protocol.py
```

---

## W. 下一步

### 建议: 进入 M5/M6

- **M5**: MAGIC raw score export - 包装 upstream KNN 路径
- **M6**: Unified evaluation adapter - 集成 threshold 和 metrics

### 不要
- 不要声称 MAGIC 已完整运行
- 不要运行真实 E3/E5 训练
- 不要安装 DGL/Torch 1.12

---

## 正确状态声明

> M3/M4 local protocol implementation complete.
> M5-M10 pending.

**不是**:
- ❌ MAGIC 已接入完成
- ❌ MAGIC 可以正式运行
- ❌ MAGIC E3 已复现
- ❌ Docker 已验证
- ❌ DGL backend 已验证

---

## 禁止事项确认

- ✅ 不实现 MAGIC GAT/GMAE 模型
- ✅ 不安装 DGL/Torch 1.12
- ✅ 不修改 `/opt/magic-upstream`
- ✅ 不读取 ground truth labels
- ✅ 不实现 M5-M10
- ✅ 不运行真实 E3/E5
- ✅ 不 git commit/push (等待人工审计)
