# MyTirex

MyTirex 用于研究车辆数据传输和计算存在延迟时，如何估计车辆在当前时刻的状态。目前使用车辆速度数据：先测量最新收到的数据已经有多旧，再用 TiRex-2 预测这段时间内的变化，最后持续输出与当前时刻对齐的速度。

例如，一条速度记录到达时已经过去 175 ms，预测计算又花了 14 ms，那么结果交付时需要估计的是原记录之后约 189 ms 的速度。

项目包含时间测量、速度预测、计算耗时补偿和连续输出。**时间测量可以独立运行，默认不加载模型。** 开发、测试和日常运行只使用本项目的 `vendor/tirex/` 当前版，具体约束见 [AGENTS.md](AGENTS.md)。

## 能做什么

- **测量数据年龄**：通过源时间戳和两端时钟同步，计算收到的数据距当前时刻已经过去多久，同时给出误差范围和可用状态。
- **补偿传输与计算延迟**：使用已收到的历史预测未来速度，在计算结束后按实际完成时间选择预测值。
- **持续发布状态**：接收数据与后台推理同时进行，默认每 25 ms 发布一次状态；推理较慢时沿用已有预测轨迹。
- **生成可查看的结果**：提供时间误差、速度误差、单请求耗时和发布频率的 CSV、图表及 HTML 报告。

实验主要基于 Comma2k19 的 CAN 车速数据。网络抖动、丢包和时钟误差由受控回放模拟，也可以使用 UDP 在两台设备之间采集真实时间戳。两类结果的含义见 [时间测量指南](docs/aoi.md)。

## 快速开始

准备好 `.venv` Conda 环境、数据和模型权重。在项目根目录直接运行：

```bash
cd /home/patton/MyTirex

# 检查数据年龄计算，不加载模型
.venv/bin/python -m mytirex eval-aoi \
  --max-series 1 --max-queries 16 \
  --profile jitter_loss_tail --scenario timestamp_noise

# 检查连续输出，不加载模型
.venv/bin/python -m mytirex eval-stream --duration-s 1 \
  --profile jitter_loss_tail --scenario timestamp_noise
```

每条命令都会打印结果目录。用浏览器打开其中的 `report.html` 即可查看报告，CSV 用于进一步分析。新机器需要先建立环境并放置本地资产。

## 安装与运行要求

环境固定为 Linux x86_64、Python 3.12、PyTorch 2.8.0 和 CUDA 12.8。模型推理需要 NVIDIA GPU、驱动，以及能编译原生扩展的 CUDA 开发工具链；时间测量入口不要求加载 GPU 模型。

项目使用 **Conda prefix 环境**，不是 `python -m venv`。重新建立环境时：

```bash
cd /path/to/MyTirex
~/anaconda3/bin/conda create -y --copy --prefix "$PWD/.venv" \
  --file configs/conda-linux-64.lock
.venv/bin/python -m pip install --no-deps --only-binary=:all: \
  -r configs/requirements-runtime.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
```

按实际安装位置替换 Conda 路径。数据与权重从本地资产恢复，程序不会自动下载；正式评估由使用者手动运行。环境版本、编译缓存和分发说明见 [维护指南](docs/development.md)。

## 常用命令

下面是确认功能和结果输出的短命令。带模型的命令只加载当前版 TiRex；首次请求可能需要数十秒初始化或编译内核。

| 命令 | 用途 | 主要结果 |
|---|---|---|
| `doctor --smoke` | 检查本地环境、资产和真实 GPU 预测 | 检查报告、预测轨迹 |
| `eval-aoi` | 检查数据年龄计算 | 年龄误差、区间覆盖率 |
| `eval-speed` | 在五条固定速度序列上回放预测 | 预测值、速度误差 |
| `benchmark` | 测量一次请求的实际计算耗时 | 逐请求耗时、分位数 |
| `eval-aoi --with-model --delivery` | 检查计算完成时的预测补偿 | 交付时刻误差、计算耗时 |
| `eval-stream` | 检查接收与连续发布 | 状态流、发布频率 |

```bash
# 当前版 GPU 功能检查
.venv/bin/python -m mytirex doctor --smoke

# 固定延迟回放：每种延迟只评分 8 个查询
.venv/bin/python -m mytirex eval-speed \
  --sequence speed_seg01 --delay-ms 175 --delay-ms 300 --max-queries 8

# 单请求耗时短检查；正式尾延迟测量应手动使用至少 1000 个稳定请求
.venv/bin/python -m mytirex benchmark --batch-size 1 --samples 32 --warmup 5

# 将计算耗时加入预测补偿
.venv/bin/python -m mytirex eval-aoi --with-model --delivery \
  --max-series 1 --max-queries 8 \
  --profile fixed_175 --scenario shared_clock_control

# 用模型更新轨迹，并每 25 ms 发布状态
.venv/bin/python -m mytirex eval-stream --with-model --duration-s 10 \
  --profile fixed_175 --scenario shared_clock_control \
  --inference-period-ms 25 --inference-period-ms 100
```

全部参数见 `.venv/bin/python -m mytirex <命令> --help`。各入口保留原命令名和参数，可从其他工作目录运行，并通过 `--root /path/to/MyTirex` 指定项目目录。

真实网络输入使用 `sync-source`、`sync-capture`、`replay-aoi` 和 `stream-receive`。两端运行及日志重放见 [时间测量指南](docs/aoi.md#真实网络采集与重放)，连续 JSON 输出见 [流式指南](docs/streaming.md#真实-udp-输入与下游输出)。

## 数据、模型与配置

| 位置 | 内容 |
|---|---|
| `data/regression/comma2k19/` | 五条各 180 秒的插值车速序列，用于固定延迟回归检查 |
| `data/prepared/comma2k19/v1/` | 保留原传感器时间的因果车速数据，用于时间与流式实验 |
| `data/raw/comma2k19/` | 已有原始数组和来源清单 |
| `models/tirex2/` | 本地 TiRex-2 权重 |
| `vendor/tirex/` | 当前版 TiRex 源码 |
| `references/current_speed/` | 与固定协议匹配的当前版历史预测 |

两类速度数据都使用 40 Hz 网格，即每 25 ms 一个点，但目标值定义不同，不能混用。模型只接收当时已经到达的数据，评分参考值在预测完成后由评估端读取。模型加载使用本地文件，不访问 Hub。

主要运行配置为 `configs/experiment.json` 和 `configs/tirex_tuning.json`。时间配置为 `configs/aoi.json`，经验证的副本为 `configs/aoi_frozen.json`；数据、源码和权重身份由同目录的索引记录。本次整理只调整代码位置与导入路径，保留原数据划分、预测参数、计算方法和评分协议。

## 项目结构

```text
MyTirex/
├── src/mytirex/
│   ├── cli.py             # 命令入口
│   ├── common.py          # 路径、身份与结果目录
│   ├── data/              # 数据读取、到达历史与历史参考
│   ├── inference/         # 当前版模型、GPU 与交付补偿
│   ├── timing/            # 时钟同步、接收缓冲区与 UDP
│   ├── streaming/         # 异步运行、回放与实时发布
│   ├── evaluation/        # 诊断、时间、速度与性能评估
│   └── reporting/         # 指标、图表与 HTML
├── configs/               # 配置、环境锁与资产身份
├── docs/                  # 专题指南与维护说明
├── tests/                 # 回归和接口检查
├── data/                  # 本地数据
├── models/                # 本地权重
├── vendor/tirex/           # 当前版 TiRex
├── references/            # 历史身份与当前版回归参考
├── outputs/               # 关键结果、验收记录与历史身份索引
└── dist/                  # 分发包与内容校验清单
```

Python 接口位置见 [模块说明](docs/development.md#代码组织)。`python -m mytirex` 和安装后的 `mytirex` 命令保持一致。

## 结果与验证

结果写入独立的 `outputs/<时间>_<任务>/`，成功运行更新对应的 `latest_*.json`。常见文件包括逐查询 CSV、汇总指标、`config.json`、`metadata.json`、图表和 `report.html`；流式入口另有逐行写出的 `states.jsonl`。

已有核心结果包括完整时间验证、当前版模型验证、175/300 ms 各 1000 次单请求测量、交付补偿，以及 15 分钟流式实验。流式实验的实际发布率约为 **39.99 Hz**，使用后台完整推理和轨迹复用。结果及适用范围见 [时间验证记录](docs/aoi.md#已有验证结果)和 [结果清单](docs/development.md#结果保留规则)。这些是已有实验，不代表整理后的代码已重新完成全量评估。

两份完整验证的逐查询表以 `aoi_queries.csv.gz` 无损保存，汇总、分块指标、报告及原配置不变。重复、失败、中断和开发临时输出已移除，清理身份见 `outputs/archive/cleanup_manifest.json`。本次短验收结果集中放在 `outputs/verification/20261011_cleanup/`，原完整实验的成功指针继续保留。

本机运行回归检查：

```bash
.venv/bin/python -m pytest
```

数据、模型、结果、环境和编译缓存均不纳入 Git。Python wheel 与可迁移项目包位于 `dist/`，内容和恢复方法见 [打包说明](docs/development.md#打包与迁移)。

## 进一步阅读

- [时间测量与计算时延补偿](docs/aoi.md)：时间定义、同步方法、双设备命令及已有验证。
- [流式输入与连续输出](docs/streaming.md)：异步调度、轨迹覆盖、状态接口及实验命令。
- [研究设计](docs/design.md)：研究目标、现有边界与后续方向。
- [维护指南](docs/development.md)：环境细节、模块位置、结果清理、验收与打包。
