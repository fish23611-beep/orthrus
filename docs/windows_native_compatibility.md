# Windows 原生兼容性与验收记录

本次修复基于 `fix/windows-native-compat`，冻结基线为
`027962c372df3ad2d3e5ffd0a09351e06c75e409`。所有操作使用原生
PowerShell 和 `F:\venvs\mstc-pids\Scripts\python.exe`，未使用 WSL，未执行全量训练。
本次相关验收结果与完整命令位于
`artifacts/windows-compat/20261019/acceptance-final/commands.json` 和同目录 JUnit XML。

最终状态：**PASS（本次 Windows 兼容性验收范围）**。四组共 **878 passed，0 failed，
0 errors，0 skipped**；统一启动器退出码为 **0**。

| 最终测试组 | 通过 | 失败/错误/跳过 | 退出码 |
| --- | ---: | --- | ---: |
| affected：内存、profiler、数据加载、预处理与合成训练 | 206 | 0 / 0 / 0 | 0 |
| mstc：数值稳定性、checkpoint、多尺度、TimeGap、校准与 Top-K | 306 | 0 / 0 / 0 | 0 |
| io：图构建、时间顺序、序列化和评估 | 188 | 0 / 0 / 0 | 0 |
| orchestration：启动入口、运行时基准、事件索引与恢复 | 178 | 0 / 0 / 0 | 0 |

上述组内包含 P4-NS **10/10**、Checkpoint Policy **16/16**、内存测试 **26/26**。
另有 16 个真实模块导入、原生内存/GPU smoke、4096 事件 RSS 基准、语法检查、
`pip check`、`pip freeze`、`git diff --check`，各命令退出码均为 0。

最终内存对照中，分配前 RSS/峰值为 **723.84765625 MiB**；触碰 32 MiB 分配后
为 **755.8515625 MiB**；释放后 RSS 回落到 **723.84765625 MiB**，峰值仍为
**755.8515625 MiB**。三个阶段 helper、psutil 与 Windows
`GetProcessMemoryInfo` 读数一致，证明峰值没有被当前 RSS 替代。
CUDA 矩阵和 TransformerConv/SAGEConv 的前后向通过，实测探针峰值 GPU
分配仅 **17.38916015625 MiB**。

## 环境与依赖

| 项目 | 实测版本 |
| --- | --- |
| Python | 3.10.11 / Windows AMD64 |
| PyTorch | 2.9.0+cu128 |
| PyTorch Geometric | 2.8.0.post1 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| psutil | 7.2.2 |
| pytest | 9.1.1 |
| 新增 igraph | 0.11.5 |
| 新增 texttable | 1.7.2 |
| 新增 pytest-mock | 3.15.1 |

仅安装了表中三项新增包，均来自官方 PyPI wheel。igraph 使用
`igraph-0.11.5-cp39-abi3-win_amd64.whl`，版本与冻结 Dockerfile 一致；
ABI3 wheel 已在本机 Python 3.10 实际导入验证。原有数值依赖均保留。
`requirements-devcontainer.txt` 未改；其中指定的 pytest 版本与本机不同，未因此降级本机 pytest。

增量安装方法如下，需先具备上述 CUDA 环境，不能用这份增量清单创建完整环境：

```powershell
& 'F:\venvs\mstc-pids\Scripts\python.exe' -m pip install --only-binary=:all: --index-url https://pypi.org/simple -c constraints-windows-compat.txt -r requirements-windows-compat.txt
```

NLTK 的 Python 包已安装，但包安装不包含分词资源。本次从 NLTK 官方数据仓库手动安装
`punkt`、`punkt_tab` 到 `artifacts/nltk_data/tokenizers`，逐项对照官方索引校验 SHA-256：

| 资源 | 压缩包字节数 | SHA-256 |
| --- | ---: | --- |
| punkt | 13905355 | `51c3078994aeaf650bfc8e028be4fb42b4a0d177d41c012b6a983979653660ec` |
| punkt_tab | 4319076 | `e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106` |

常规安装可用 `python -m nltk.downloader -d artifacts/nltk_data punkt punkt_tab`。
本机该下载器因网络地址解析到 `198.18.1.22` 被其安全检查拒绝，因此采用官方文档支持的手动安装：
从 `https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/<资源>.zip`
下载，校验上述摘要后解压到 `artifacts/nltk_data/tokenizers`。
未关闭 NLTK 安全检查，未替换分词器。下载来源和摘要记录在 `nltk-resources.json`。

小型 ground truth 标注子模块原先未初始化，已按父仓库引用初始化到
`012f321f46137650496e639b0ad7e0a66db07a73`。它与 THEIA 原始数据、预处理产物是不同输入。

## 根因与文件变更

| 文件 | 问题与修复 |
| --- | --- |
| `src/runtime/memory_metrics.py` | 新增统一、尽力而为的内存读取；Windows 使用 RSS 和真实峰值工作集，缺失或异常返回 `None`；支持 `peak_wset` 和新版 `peak_rss` 字段。 |
| `src/data_utils.py` | 移除 Windows 不存在的顶层 `resource` 导入，复用 helper；保留 Unix 数据加载器原有峰值回退选项。 |
| `src/runtime/runtime_profiler.py` | 移除顶层 `resource` 导入，复用 helper；原报告字段和采样接口保持。 |
| `src/mstc/experiment_utils.py` | `peak_cpu_memory_mb()` 复用 helper，保持公开接口。 |
| `scripts/benchmark_preprocess_memory.py` | 复用真实进程峰值读取；缺失峰值时明确失败，避免输出伪造基准值。 |
| `src/artifact_names.py` | Windows 写入时间窗口文件时将 `:` 编码为 `%3A`；读取时可逆解码；Unix 文件名、时间字符串和数值不变。 |
| `src/graph_construction/build_orthrus_graphs.py` | 仅保存图的文件名使用 Windows 编码，图构建、事件筛选与窗口划分不变。 |
| `src/edge_featurization/embed_edges_feature_word2vec.py` | 用 `os.path.basename` 提取本地文件名，避免 Windows 反斜杠路径被当作完整输出文件名。 |
| `src/detection/orthrus_gnn_testing.py` | 仅 CSV 输出文件名编码，推理与评分逻辑不变。 |
| `src/detection/evaluation_utils.py` | 提取本地文件名并解码后，继续使用原来的时间解析与标签计算。 |
| `tests/test_memory_metrics.py` | 覆盖 Windows 字段、单位、异常、缺失峰值、Unix 模拟、消费者异常降级及原生读取。 |
| `tests/test_artifact_names.py` | 验证 Windows 名称可写、往返与顺序，以及 Unix 名称保持。 |
| `tests/conftest.py` | Windows 没有符号链接权限时使用真实目录 junction，继续验证生产 realpath 解析。临时目录所在卷需支持 junction。 |
| `tests/test_runtime_benchmark.py` | 使用上述路径别名夹具，原重叠保护断言不变。 |
| `tests/test_c8_3b_empty_day_semantics.py` | 使用上述路径别名夹具，原 PostgreSQL 版本和解析路径断言不变。 |
| `tests/test_preprocess_memory_streaming.py` | 测试产物名称适配 Windows；显式反序列化测试自己生成的图和 PyG 对象。 |
| `tests/test_embed_edges_temporal_order.py` | 对测试自己生成的 PyG 对象显式指定 `weights_only=False`，保持顺序/对齐断言。 |
| `tests/test_global_event_index.py` | 同上，保持事件 split/index 断言。 |
| `tests/test_pytorch26_serialization_compat.py` | 旧版加载器模拟显式模拟其原来的 `weights_only=False` 默认值，避免误用宿主 PyTorch 2.9 默认值。 |
| `scripts/check_windows_native.py` | 可重跑的 16 模块导入、Windows API/psutil 对照、32 MiB 分配释放和小型 CUDA/PyG 前后向检查。 |
| `scripts/validate_windows_compat.py` | 分进程执行相关测试和检查，保存完整参数、环境、日志、JUnit 与退出码。 |
| `requirements-windows-compat.txt` | Windows 增量依赖，不改 Linux 容器清单。 |
| `constraints-windows-compat.txt` | 固定已验证的关键依赖，避免增量安装变更数值栈。 |
| 本文档 | 记录修复边界、复现步骤、证据与未完成实验。 |

没有删除测试、改变断言预期或增加 skip/xfail。已有六个测试文件的全部 `assert` AST
与冻结基线一致。生产序列化信任边界未改；只有测试自己创建的对象使用显式完整反序列化。

## 复现与证据

在项目根目录运行：

```powershell
$env:PYTHONUTF8 = '1'
& 'F:\venvs\mstc-pids\Scripts\python.exe' scripts/validate_windows_compat.py --output-dir artifacts/windows-compat/20261019/recheck
```

启动器为子进程设置 `WANDB_MODE=disabled`、`PYTHONUTF8=1`，将 `src` 加入
`PYTHONPATH`，将项目 NLTK 资源目录放在 `NLTK_DATA` 首位并保留原搜索路径。
UTF-8 模式解决部分旧测试按 GBK 读取 UTF-8 源码的问题；不应误判为模型缺陷。
启动器使用调用它的同一个解释器执行所有 pip 和 pytest 命令。

启动器包含四组独立测试进程，列表写在 `GROUPS` 中，执行形式为
`python -m pytest -q <该组全部测试文件> --junitxml=<输出目录>/<组名>.xml`，
逐个完整命令与退出码见 `commands.json`，并额外执行：

```text
python scripts/check_windows_native.py --output <输出目录>/native-smoke.json
python scripts/benchmark_preprocess_memory.py --events 4096 --fetch-size 128
python -m compileall -q src scripts tests
python -m pip check
python -m pip freeze
git diff --check
```

所有原始审计及失败/复跑日志保存在 `artifacts/windows-compat/20261019`，未把先前失败算作通过。
`environment.before.txt`、`environment.after.txt` 和 `dependency-audit.json` 保存版本变化；
安装报告保存 wheel URL 与哈希。`pre-existing.patch`、`memory_metrics.before.py` 保存进入任务时的修改。

`semantic-audit.json` 记录 15 个冻结核心文件逐字节相同，覆盖采样/融合、GNN、TimeGap、
HistoryStore、RNG、checkpoint、校准、Top-K、阈值和序列化。其余四个源文件已用 AST
验证除新增导入与指定文件名赋值外完全相同。冻结分支仍指向原基线。

## 尚未完成的验证

Linux 和 macOS 只做了读取路径/单位/回退的模拟测试，尚未在原生系统重新跑回归或 G3。
Linux 仍使用 `/proc/self/statm` 的驻留页乘 `SC_PAGE_SIZE`；峰值仍使用
`getrusage(RUSAGE_SELF).ru_maxrss`，Linux 按 KiB、macOS 按字节转换为 MiB。
这些代码与模拟证据不能代替原生平台验收。

`F:\Paper\Data\MSTC-Data\datasets` 与 `artifacts` 在验收时均为空。仓库虽有预置权重，
仍缺少 THEIA_E3/E5 预处理窗口、相应语义模型和所需 metadata，不能执行真实 2-Window GPU
实验。未下载大型数据、未运行全量 E3/E5。小型 GPU 检查只证明算子兼容性，不能证明真实
模型数值等价性或 8 GB 显存足够容纳真实窗口。

4096 事件 RSS 基准仅用于检查脚本可运行和新旧图语义相符，其内存差异不作为性能收益结论。
PyTorch TF32 与 profiler 的弃用提示保留，未为消除提示改变冻结推理设置。

官方依据：[igraph 安装说明](https://python.igraph.org/en/0.10.2/install.html)、
[psutil 内存 API](https://psutil.io/api/)、[NLTK 资源安装](https://www.nltk.org/data.html)、
[NLTK 官方索引](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml)。
