# 环境、代码组织与项目维护

首次使用说明与常用命令见 [README](../README.md)。本文集中记录环境细节、模块迁移、结果保留和打包方法。

## 环境

项目 `.venv` 是独立创建的 Conda prefix 环境。已验证版本为 Python 3.12.15、PyTorch 2.8.0+cu128、nvcc 12.8.93、flashrnn 1.0.8、xlstm 2.0.3 和 mlstm-kernels 2.0.6。本机 GPU 为 NVIDIA GeForce RTX 4080 Laptop GPU。

`configs/conda-linux-64.lock` 固定 Python、CUDA 12.8.1 开发工具链和宿主编译器的包构建，`configs/requirements-runtime.lock` 固定推理、绘图及测试依赖。安装使用 `--no-deps`，因为 XLSTM 元数据另外声明了训练及 Notebook 工作流依赖；常规 `pip check` 可能报告这些未安装依赖。重建命令见 README。

[`inference/gpu.py`](../src/mytirex/inference/gpu.py) 优先定位当前 prefix 的 nvcc，设置 `CUDA_HOME`，组合 Conda 与 NVIDIA wheel 的头文件和开发库路径。只有 PyTorch CUDA runtime 不足以编译 FlashRNN。显式指定的 `CUDA_HOME` 会被尊重，应指向匹配的 12.8 工具链。

默认 `MAX_JOBS=4`，可通过环境变量调整。编译缓存位于 `.cache/{torch_extensions,triton,inductor}`。首请求可能包含 tracing、缓存加载或内核编译，应与稳定阶段分开记录。

当前模型加载固定为 `compile=False`、`use_flex_attention=False`、`local_files_only=True`，关闭 Hub 联网。数据下载、正式评估和双设备实验由使用者手动运行，规则见 [AGENTS.md](../AGENTS.md)。

## 代码组织

`cli.py`、`__main__.py`、`__init__.py` 和 `common.py` 保留在包根目录。21 个业务模块按责任移动到六个子包，命令名、参数及结果格式保持原样。

| 子包 | 文件与责任 |
|---|---|
| `data/` | `arrival.py` 到达历史；`regression.py` 固定回归数据；`prepared.py` 因果语料；`references.py` 当前版历史参考 |
| `inference/` | `model.py` 模型边界及轨迹；`gpu.py` 环境；`delivery.py` 计算耗时与交付选值 |
| `timing/` | `clock.py` 同步与年龄；`receiver.py` 接收窗口；`transport.py` UDP 与事件重放 |
| `streaming/` | `runtime.py` 异步运行器；`replay.py` 受控回放与评分；`live.py` 实时 UDP 发布 |
| `evaluation/` | `speed.py` 固定延迟回放；`aoi.py` 时间实验；`simulation.py` 私有仿真驱动；`benchmark.py` 单请求耗时；`doctor.py` 诊断 |
| `reporting/` | `metrics.py` 通用指标；`aoi_metrics.py` 年龄指标；`html.py` 图表和报告 |

Python 接口使用新位置，例如：

```python
from mytirex.inference.model import ForecastTrajectory, TiRexModel
from mytirex.timing.receiver import TimestampedReceiver
from mytirex.streaming.runtime import StreamConfig, StreamingRuntime
```

内部导入、延迟导入、测试导入、monkeypatch 路径和保留的结果分析脚本均同步更新。静态语法树对比确认业务代码除导入语句外没有变化。目录变化会改变集成源码 SHA256，旧结果保留原身份；`vendor/tirex/`、权重、数据和运行配置保持原内容。

数据视图只包含已经收到的历史。`ForecastTrajectory` 返回完整 9 分位数轨迹；`value_at()` 用于未来网格点，`value_at_offset()` 用于观测锚点到未来中位数之间的连续插值。时间、跨度和评分隔离规则见 [时间测量指南](aoi.md)。

## 结果保留规则

本次整理保留以下核心结果及其配置、身份、报告和明细：

| 结果 | 目录 |
|---|---|
| 完整时间 validation | `outputs/20261010_190625_942320_aoi_timing/` |
| 完整当前版 GPU validation | `outputs/20261010_234729_859814_aoi_current/` |
| 最新完整固定延迟回放 | `outputs/20261011_200652_924724_speed_current/` |
| 175/300 ms 各 1000 次单请求测量 | `outputs/20261011_142805_811118_benchmark_current/`、`outputs/20261011_142913_491793_benchmark_current/` |
| 计算时延补偿 | `outputs/20261011_144937_463100_delivery_current/` |
| 15 分钟当前版流式实验 | `outputs/20261011_185625_913343_stream_current/` |
| 无模型流式检查 | `outputs/20261011_184534_246487_stream_timing/` |
| 真实 UDP 采集及分析 | `outputs/20261011_141017_948279_aoi_capture/`、对应的 `_review/` |
| 时间与当前版 validation 复核 | `outputs/validation_aoi_review/` |
| 环境与 GPU 诊断 | `outputs/latest_doctor.json` 指向的成功目录 |

两份完整 validation 原逐查询表共约 32.6 GiB，现以 `aoi_queries.csv.gz` 无损保存。每份 `storage.json` 记录原文件大小／SHA256、压缩后大小／SHA256，以及完整解压内容校验结果。原配置、metadata、汇总和分块指标不变，冻结配置中的原 CSV 大小仍是未压缩大小。

Pandas 可以直接读取：

```python
import pandas as pd
rows = pd.read_csv("outputs/20261010_190625_942320_aoi_timing/aoi_queries.csv.gz", nrows=6000)
```

重复、小范围旧验收、失败和中断目录，以及临时安装／开发审计输出已经清理。官方历史输出及旧对比只保留身份，清单为 `outputs/archive/cleanup_manifest.json`，不进入日常配置、分析或成功指针。封存源码及旧迁移清单仍仅用于身份追溯。

`latest_*.json` 继续指向相应完整实验或关键成功结果。本次路径验收短结果集中放在 `outputs/verification/20261011_cleanup/`，不替代已有完整实验。以后新增输出仍按原时间目录写入，清理时应检查状态、配置和实验范围。

## 验收范围

本机回归检查使用：

```bash
.venv/bin/python -m pytest
```

原时间跨度测试样例只将派生步数改为 8，却保留 175 ms 年龄上界，整理前同样失败。现统一为 200 ms 上界和 8 步，计算函数未改。导入扫描改为递归覆盖全部子包。

资产完整性测试只校验当前版源码、权重、数据、当前版历史参考和论文，不依赖封存官方文件。原完整迁移清单继续保留追溯身份。

本次检查 **87 项通过**；**12 个短命令验收通过**。模型流式短检查以 25/100 ms 两种请求间隔共发布 80 个状态，实际 40 Hz；localhost 采集与重放的查询 CSV 完全一致。

命令验收覆盖 GPU 诊断、有限查询的速度回放、少量 benchmark 样本、纯时间及模型模式的 AoI、交付补偿、无模型及模型流式回放，以及 localhost UDP 接收和日志重放。使用现有本地数据和当前版模型，不下载数据，不启动完整正式评估。确切命令、退出状态和结果文件见 `outputs/verification/20261011_cleanup/verification.json`。

`doctor --verify-assets` 是原迁移目录的完整内容身份检查，包含封存资产的文件校验，不加载封存模型。分发后的当前运行资产使用 `doctor --smoke` 检查。

## 打包与迁移

`dist/` 提供 Python wheel、项目归档和 SHA256 清单。wheel 包含全部应用子包及命令入口，使用时通过 `--root` 指向具备配置、当前版 vendor、权重和数据的项目目录。

分发文件为 `mytirex-0.1.0-py3-none-any.whl`、`mytirex-0.1.0-project.tar.gz` 和 `SHA256SUMS`。wheel 已在隔离目录安装，全部 31 个应用模块导入成功，并用显式 `--root` 完成一次无模型时间测量。

项目归档包含源码、配置、文档、测试、当前版 `vendor/tirex/`、本地数据、权重、当前版历史参考和核心报告／指标。不复制 `.venv`、编译缓存或 Git，不加入封存官方源码和预测。两份完整 validation 的压缩逐查询表保留在原工作区，分发包携带汇总、分块指标及 `storage.json` 身份，避免重复打包大型明细。

如需在分发目录重新执行完整 validation 复核，应把这两份 `aoi_queries.csv.gz` 复制到原时间目录，再运行保留的 `review.py`。包内 `PACKAGE_CONTENTS.json` 列明包含的文件及省略的大型明细。

在新目录解压项目归档，按 README 建立 Conda prefix 环境，再运行回归检查、无模型时间短检查和 `doctor --smoke`。路径由 `--root` 或可编辑安装位置解析，无需原 `/home/patton/experiment2` 或封存官方资产。

文档按主题保留四份：本指南、`aoi.md`、`streaming.md`、`design.md`。原计算时延说明和 validation 记录合并进 `aoi.md`，原“总方案”归为 `design.md`；参考论文 `2607.01204v1.pdf` 保留原内容与路径。
