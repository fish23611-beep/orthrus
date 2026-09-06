# MAGIC M7 Progress — Seed Control / Reproducibility Contract

**状态**: M7 local seed-control implementation complete

**日期**: 2026-09-06

---

## 1. M7 Seed Contract

M7 实现了统一的 RNG 控制层，确保 MAGIC 基线在正式运行中：

- 使用显式传入的 seed，不依赖隐式默认值
- 所有 RNG（Python/NumPy/Torch/DGL）都从同一个 base seed 派生
- DataLoader generator 和 workers 有确定性 seed 配置
- deterministic flags 正确设置
- Machine-readable manifest 记录所有 RNG 状态

### 正式 Seeds

| Seed | 用途 |
|------|------|
| 0 | 第一个独立 stochastic run |
| 1 | 第二个独立 stochastic run |
| 2 | 第三个独立 stochastic run |

⚠️ **禁止行为**:
- 把同一个 checkpoint 复制成三个 seed
- 把 upstream 的 100 次 evaluation shuffle 当成 3 个独立训练 seed
- 在代码路径里偷偷硬编码 seed=0
- 使用 test 结果选择 seed

---

## 2. Python / NumPy / Torch / CUDA / DGL Coverage

### Python random
```python
from src.baselines.magic.seed import set_python_random_seed
set_python_random_seed(0)  # 设置 Python random 模块
```

### NumPy
```python
from src.baselines.magic.seed import set_numpy_seed
set_numpy_seed(0)  # 设置 NumPy 全局 RNG

# 或创建独立 RNG
from src.baselines.magic.seed import create_numpy_generator
rng = create_numpy_generator(0)  # 不会影响全局 RNG
```

### Torch CPU/CUDA
```python
from src.baselines.magic.seed import set_torch_seed
set_torch_seed(0, set_cuda=True)  # 设置 Torch 和 CUDA RNG
```

### Deterministic Flags
```python
from src.baselines.magic.seed import set_deterministic_flags
set_deterministic_flags(deterministic=True, warn_only=True)
# 设置:
# - torch.backends.cudnn.deterministic = True
# - torch.backends.cudnn.benchmark = False
# - torch.use_deterministic_algorithms(..., warn_only=True)
```

### DGL Hook
DGL 是可选的，延迟导入：
```python
from src.baselines.magic.seed import set_dgl_seed

# 如果 DGL 不可用且 require_dgl=True，抛出 ImportError
set_dgl_seed(0, require_dgl=True)  # 正式环境需要

# 如果 DGL 不可用且 require_dgl=False，返回降级状态
result = set_dgl_seed(0, require_dgl=False)
# result = {"dgl_available": False, "degraded": True, ...}
```

---

## 3. DataLoader Generator/Worker Policy

### DataLoader Generator
```python
from src.baselines.magic.seed import build_magic_generator

gen = build_magic_generator(0)
# 返回 torch.Generator() 且 generator.manual_seed(0)
# 如果 torch 不可用，返回 None
```

### Worker Init Function
```python
from src.baselines.magic.seed import build_worker_init_fn

worker_fn = build_worker_init_fn(base_seed=0)
# 每个 worker 得到 seed = base_seed + worker_id
# Worker 0: seed=0, Worker 1: seed=1, Worker 2: seed=2, ...

# 使用方式
from torch.utils.data import DataLoader
loader = DataLoader(dataset, num_workers=3, worker_init_fn=worker_fn)
```

### Worker Seeds Utility
```python
from src.baselines.magic.seed import build_worker_seeds

seeds = build_worker_seeds(base_seed=42, num_workers=4)
# 返回 [42, 43, 44, 45]
```

---

## 4. Deterministic Policy

### 兼容目标
- Python 3.8
- Torch 1.12.1

### API 检测
- `torch.backends.cudnn.deterministic` 存在
- `torch.backends.cudnn.benchmark` 存在
- `torch.use_deterministic_algorithms` 存在
- `warn_only` 参数: Torch 1.12+ 支持，版本检测后使用

### 降级记录
如果某 deterministic API 不可用：
- 不能静默忽略
- 必须在 `ReproducibilityManifest.warnings` 中记录降级

---

## 5. RNG Manifest Schema

```python
from src.baselines.magic.seed import ReproducibilityManifest

manifest = ReproducibilityManifest(seed=42)
# 字段:
# - seed: int
# - require_dgl: bool
# - python_seed_set: bool
# - numpy_seed_set: bool
# - torch_available: bool
# - torch_cuda_available: bool
# - torch_seed_set: bool
# - torch_cuda_seed_set: bool
# - torch_cuda_seed_all_set: bool
# - deterministic_requested: bool
# - cudnn_deterministic: Optional[bool]
# - cudnn_benchmark: Optional[bool]
# - torch_deterministic_algorithms: Optional[bool]
# - deterministic_algorithms_warn_only: bool
# - dgl_available: bool
# - dgl_version: Optional[str]
# - dgl_seed_set: bool
# - dgl_random_seed_set: bool
# - dataloader_generator_seeded: bool
# - worker_seeds_configured: bool
# - reproducibility_level: str  # "full" | "partial" | "degraded" | "unknown"
# - warnings: list[str]

# 导出为 dict
manifest_dict = manifest.to_dict()
```

### Reproducibility Level
- `full`: 所有关键 RNG 都已设置
- `partial`: 部分 RNG 不可用
- `degraded`: CUDA 或必需 DGL 不可用
- `unknown`: 初始状态

---

## 6. Same-seed vs Different-seed Synthetic Evidence

### Same-seed Deterministic
```python
# 同一 seed 产生相同随机序列
set_python_random_seed(42)
seq1 = [random.random() for _ in range(10)]
set_python_random_seed(42)
seq2 = [random.random() for _ in range(10)]
assert seq1 == seq2  # ✓ 通过
```

### Different-seed Changes Trace
```python
# 不同 seed 产生不同随机序列
set_python_random_seed(0)
seq0 = [random.random() for _ in range(10)]
set_python_random_seed(1)
seq1 = [random.random() for _ in range(10)]
assert seq0 != seq1  # ✓ 通过
```

---

## 7. Local 环境边界

本轮实现**仅使用**:
- ✅ `src/baselines/magic/seed.py` - 新增 seed module
- ✅ `src/baselines/magic/scoring.py` - 修改 scorer RNG 逻辑
- ✅ `tests/test_magic/test_magic_seed.py` - 新增 M7 测试
- ✅ synthetic tests
- ✅ mock/fake DGL hooks

**未执行**:
- ❌ 真实 THEIA_E3
- ❌ 真实 THEIA_E5
- ❌ MAGIC training
- ❌ DGL 真实安装
- ❌ Torch 1.12 安装
- ❌ 正式 Docker 构建

---

## 8. DGL/CUDA 能力边界

### DGL
- **当前环境**: DGL 不可用（未安装）
- **实现方式**: 延迟导入，不在 module import 时强制 `import dgl`
- **测试方式**: 使用 `FakeDGLModule` mock
- **正式环境**: 需要 DGL 1.0.0+，使用 `set_dgl_seed(0, require_dgl=True)`

### CUDA
- **当前环境**: CUDA 可能可用（取决于 torch 安装）
- **实现方式**: 检测 `torch.cuda.is_available()`
- **降级**: 如果 CUDA 不可用，跳过 CUDA seed 设置，记录在 manifest 中

---

## 9. Upstream Hardcoded Seed 0 隔离

### Upstream 硬编码 Seed 0
| 文件 | 行 | 行为 | 影响 |
|------|-----|------|------|
| `train.py` | 39 | `set_random_seed(0)` | 不影响我们的 wrapper |
| `eval.py` | 23 | `set_random_seed(0)` | 不影响我们的 wrapper |
| `model/eval.py` | 59 | `set_random_seed(s)` | 不影响我们的 wrapper |
| `model/eval.py` | 116 | `set_random_seed(0)` | 不影响我们的 wrapper |

### 隔离策略
- Upstream 是外部 baseline，不被 wrapper 调用
- 我们的 wrapper 使用独立的 seed module
- 每个 wrapper run 可以使用不同的 seed

### Upstream SHA256
```
9003a2ec19632073d55a1d689b828272bc53496685175fa4c88dbb63fdf028ad  train.py
299352ea0387237fd0fcefe806024e1ecf18bd0d82ebdbdf12d31b4fbeed2208  eval.py
4761d6481bd29ddc2789eebf2d4dbb04a76176f23223e0690a1f0a0f5f2d75dc  model/eval.py
57a84d42d6fa3f13a98e7ef606e9c3087574868bea3d70d62adaeca2bf155924  utils/loaddata.py
9e350ba709815437434c27aa4983e28785b4ab673ea3bf75dff6094984757fec  utils/utils.py
```

---

## 10. M8-M10 Pending

| Phase | 状态 | 说明 |
|-------|------|------|
| M1 | CONTRACT FROZEN | 未实现 |
| M2 | CONTRACT FROZEN | 未实现 |
| M3 | ✅ DONE | Canonical Input Adapter |
| M4 | ✅ DONE | Unified Causal Protocol |
| M5 | ✅ DONE | Raw Score local protocol |
| M6 | ✅ DONE | Unified Evaluator local protocol |
| **M7** | **✅ DONE** | **Seed Control (本轮)** |
| M8 | PENDING | Smoke tests |
| M9 | PENDING | E3 seed0 pilot |
| M10 | PENDING | Formal E3/E5 3-seed runs |

---

## 11. 新增/修改文件

### 新增
- `src/baselines/magic/seed.py` - M7 seed module
- `tests/test_magic/test_magic_seed.py` - M7 测试 (27 tests)
- `docs/MAGIC_M7_PROGRESS.md` - 本文档

### 修改
- `src/baselines/magic/scoring.py` - scorer RNG 逻辑增强

---

## 12. 测试结果

### M7 Targeted Tests
```
27 passed, 1 skipped in 0.94s
```

### M3-M6 Regression Tests
```
151 passed, 1 skipped in 4.46s
```

---

## 13. API 使用示例

### 基础用法
```python
from src.baselines.magic.seed import set_magic_seed, MagicSeedController

# 方式 1: 便捷函数
manifest = set_magic_seed(0)
print(manifest.to_dict())

# 方式 2: 控制器（推荐用于正式 runs）
controller = MagicSeedController(
    seed=0,
    set_deterministic=True,
    set_cuda=True,
    require_dgl=False,  # 或 True 在正式环境
)
manifest = controller.set_all_seeds()

# 获取 DataLoader generator
gen = controller.build_generator()
worker_fn = controller.build_worker_init_fn()
```

### 在 Scorer 中使用
```python
from src.baselines.magic.seed import set_magic_seed
from src.baselines.magic.scoring import MAGICEntityScorer

# 先设置全局 RNG
set_magic_seed(0)

# scorer 可以使用局部 RNG 避免污染全局
scorer = MAGICEntityScorer(
    k=10,
    seed=0,
    use_local_rng=True,  # 推荐
)
```

---

## 14. 准确状态

✅ **M7 local seed-control implementation complete**

⚠️ 以下**未完成**（不在 M7 范围内）:
- ❌ MAGIC reproducibility fully verified on GPU
- ❌ MAGIC three-seed experiments complete
- ❌ MAGIC reproduction complete
- ❌ DGL 真实安装
- ❌ 正式 THEIA runs

---

## 15. M7 Pre-commit 修复（第二轮）

### 修复问题
1. **M5 RNG 语义不能因 M7 改变**：scorer 改用 `np.random.RandomState(seed)`
   而非 `np.random.default_rng(seed)`，保持与 legacy M5 `np.random.seed(s);
   np.random.permutation(n)` 完全一致的 sampling sequence。
2. **Hard-coded seed=0 清理**：
   - `MAGICEntityScorer(seed=None)` 默认改为 None，fit() 时 fail-fast
   - `ReproducibilityManifest(seed=None)` 表达 unknown（JSON 输出 `"seed": null`）
   - `MagicSeedController(seed=...)` 无默认值，必须显式传入
   - `get_current_manifest()` 内部使用 `seed=None`
3. **Pytest skip 清理**：使用 `monkeypatch` 模拟 torch/DGL unavailable，
   不依赖当前机器是否安装 torch/DGL。

### Scorer RNG 行为
- `use_local_rng=True`：使用独立 `np.random.RandomState(seed)`
- `use_local_rng=False`：同样使用 `np.random.RandomState(seed)`（M7 修复后默认安全）
- **永远不调用 `np.random.seed()` 全局 RNG**
- **永远不调用 `np.random.default_rng()`**

### Global RNG Isolation 证据
- `test_scorer_fit_does_not_call_global_numpy_seed`: 通过 monkeypatch
  `np.random.seed` spy，确认 `scorer.fit()` 不调用全局 seed
- `test_scorer_fit_default_path_does_not_pollute_global_rng`: 默认路径
  下不污染全局 RNG
- `test_scorer_does_not_pollute_global_rng` (M7 旧测试): 旧路径同样安全

### Legacy-equivalence 证据
- `test_legacy_global_vs_local_randomstate_equivalence`: 验证
  `np.random.RandomState(s).permutation(n)` 与
  `np.random.seed(s); np.random.permutation(n)` 完全相同
- `test_scorer_randomstate_matches_legacy_m5_sampling`: 验证 scorer's
  local RandomState 与 M5 legacy 序列一致

### Unknown Seed 表达
- 字段类型: `Optional[int]`，默认 None
- JSON 输出: `"seed": null`（不是 0）
- Seed 0 仍是合法实验 seed，整数 0

### Skip 清理
- 删除 2 处运行时 `pytest.skip(...)`
- 删除 5 处 `@pytest.mark.skipif(not _TORCH_AVAILABLE)` 装饰器
- 改为 `monkeypatch.setattr` 模拟 unavailable path
- 结果: M7 targeted tests = 0 skipped

### 测试结果（M7 Pre-commit 修复后）
- M7 targeted: 38 passed, 0 skipped
- tests/test_magic 全量: 162 passed, 0 skipped
- py_compile: OK
- upstream SHA256: 与之前完全一致
- 0 hard-coded seed=0 in runtime wrapper
