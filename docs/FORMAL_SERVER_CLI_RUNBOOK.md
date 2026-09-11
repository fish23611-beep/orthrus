# Formal Server CLI Runbook（Host environment candidate）

## 状态与边界

本文描述 ORTHRUS/MSTC-PIDS 的纯命令行候选部署路径。它是 **Host environment candidate / experimental deployment path**，尚未通过正式 GPU 验收；当前 Docker runtime contract 与冻结 fallback 仍然有效。

- Docker fallback commit：`1ce790a40741ad38372a7b4ad4f00f5384c8dbbc`
- Docker fallback tag：`formal-f1-docker-fallback-v1`
- MSTC 与 MAGIC 必须使用两个独立 Python 环境，不能混装依赖。
- Notebook 仅保留作历史、Colab 或 DSW 工具；服务器正式流程直接调用 CLI，不依赖 Jupyter。
- 本文不改变数据、split、seed、模型、阈值、评分或评估协议。

建议的服务器目录结构如下，路径只是部署示例，不是 Python 代码中的固定值：

```text
/home/yby/
├── orthrus-formal-host/
├── magic-upstream/
├── envs/
│   ├── mstc-formal/
│   └── magic-formal/
└── experiment-data/
```

开始运行前，应确认仓库 revision、数据目录、输出目录、MAGIC upstream revision 和当前激活的 Python 环境。不要让两个环境共用 `site-packages`，也不要在一次运行中动态安装或升级包。

## MSTC 独立环境

候选环境合同：Python 3.10.19、PyTorch 2.4.1+cu124、PyG 2.8.x。该环境只运行 MSTC，不安装 MAGIC 的 Torch/DGL 依赖。

激活预先创建并固定依赖的环境后，先记录运行时信息并检查 CLI：

```bash
source /home/yby/envs/mstc-formal/bin/activate
cd /home/yby/orthrus-formal-host
python --version
python -c 'import sys, torch, torch_geometric; print(sys.executable); print(torch.__version__, torch.version.cuda); print(torch_geometric.__version__); print(torch.cuda.is_available())'
python src/experiments/run_matrix.py --help
```

使用 matrix runner 调度 dataset × config × seed：

```bash
python src/experiments/run_matrix.py \
  --datasets THEIA_E3,THEIA_E5 \
  --configs config/experiments/mstc_full.yml \
  --seeds 0,1,2 \
  --artifact-root /home/yby/experiment-data/mstc-artifacts
```

如需单次运行，可使用同一套 Python pipeline CLI：

```bash
python src/experiments/run_experiment.py \
  --dataset THEIA_E3 \
  --config config/experiments/mstc_full.yml \
  --seed 0 \
  --artifact-root /home/yby/experiment-data/mstc-single-seed-0 \
  --stages all
```

## MAGIC 独立环境

候选环境合同：Python 3.8、PyTorch 1.12.1+cu116、DGL 1.0.0。该环境只运行 MAGIC，不与 MSTC 环境共享 Torch 或 graph framework。

`/home/yby/magic-upstream` 必须是冻结的 MAGIC snapshot；runner 会继续执行既有 upstream 完整性校验。激活环境后先核对版本、GPU 可见性、upstream SHA 与 CLI：

```bash
source /home/yby/envs/magic-formal/bin/activate
cd /home/yby/orthrus-formal-host
python --version
python -c 'import sys, torch, dgl; print(sys.executable); print(torch.__version__, torch.version.cuda); print(dgl.__version__); print(torch.cuda.is_available())'
git -C /home/yby/magic-upstream rev-parse HEAD
python -m src.baselines.magic.formal_runner --help
```

Host 环境运行时显式传入 upstream 路径：

```bash
python -m src.baselines.magic.formal_runner \
  --dataset THEIA_E3 \
  --seed 0 \
  --upstream-path /home/yby/magic-upstream \
  --device cuda \
  --output-dir /home/yby/experiment-data/magic-artifacts/THEIA_E3/seed_0
```

同一 runner 在 Docker fallback 中不传 `--upstream-path`，会继续使用后端默认的 `/opt/magic-upstream`：

```bash
python -m src.baselines.magic.formal_runner \
  --dataset THEIA_E3 \
  --seed 0 \
  --device cuda \
  --output-dir /artifacts/THEIA_E3/seed_0
```

每次正式候选运行都应归档 `runtime_manifest.json`。其中 Python 可执行文件与 prefix、Torch/DGL/CUDA 实际版本、CUDA 可用性与设备清单，以及 resolved upstream path 用于区分不同宿主环境；这些记录不是正式 GPU 验收本身。

## 验收与回退

Host 候选环境在宣称可用于正式实验前，至少还需分别完成依赖 inventory、CUDA/DGL import、最小 GPU smoke、artifact 完整性和结果可复现性验收。Python 3.8 与旧版 Torch/DGL 的 wheel 可得性、宿主 NVIDIA driver 兼容性及安全维护状态都是部署风险，不能因 CLI 可用而视为已解决。

若 Host 候选路径失败，停止使用该环境，不移动 fallback tag，也不修改冻结合同；从 `formal-f1-docker-fallback-v1` 对应的只读基线恢复现有 Docker 正式方案。回退不应覆盖或删除 Host 尝试产生的日志和 artifacts。
