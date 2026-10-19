# MyTirex

MyTirex 用于研究远程驾驶信号存在传输和计算延迟时，如何估计当前时刻的值：先测量收到的数据已经有多旧，再使用当前版 TiRex-2 预测这段时间内的变化，最后持续发布与实际时刻对齐的信号。

项目支持任意命名数值信号的单变量预测、多目标联合预测、历史协变量和部分已知的未来协变量。同一套接口可接收车辆遥测，也可接收驾驶端的控制信号；已有 Comma2k19 车速回放与评价入口继续保留。

例如，一条速度记录到达时已经过去 175 ms，预测计算又花了 14 ms，那么结果交付时需要估计的是原记录之后约 189 ms 的速度。

**当前优先测量数据年龄（AoI）；时间测量验收默认不加载模型。** 开发、调试、测试和日常评估只使用 `vendor/tirex/` 当前版。官方封存版仅可在用户对具体任务明确手动授权后用于最终对比，历史运行与结果不构成授权，详见 [AGENTS.md](AGENTS.md)。数据下载与正式评估由使用者手动运行。

## 能做什么

- **测量数据年龄**：通过源时间戳、四时间戳同步和接收端单调时钟，估计最新观测的年龄、误差界及可用状态。
- **预测通用信号**：按名称和单位接入控制量、转角、坐标、姿态等数值序列；多个目标在同一个模型输入中联合预测，协变量提供条件。
- **补偿传输与计算延迟**：根据观测年龄、排队和计算耗时确定预测跨度，在交付或发布时刻从轨迹中选值。
- **连续发布状态**：接收与后台 batch=1 推理并行，默认每 25 ms（40 Hz）发布一次；最多一个在途请求，新数据合并进历史，两次推理之间复用轨迹。
- **接入不同进程或设备**：提供 UDP 源适配器、实时接收命令和 Python 回调接口，支持车端与驾驶端分别部署。
- **回放与评价**：提供受控网络／时钟回放、接收日志重放、车速误差评价、单请求耗时测量，以及 CSV、图表和 HTML 报告。

通用信号的名称和单位由调用方定义；采样对齐、角度展开、坐标转换和控制量边界也由调用方处理。已有定量预测实验使用 Comma2k19 CAN 车速，其他信号的预测效果需要相应实验验证。

## 安装与运行要求

已验证环境为 Linux x86_64、Python 3.12、PyTorch 2.8.0 和 CUDA 12.8。模型推理需要 NVIDIA GPU、驱动，以及能编译原生扩展的 CUDA 开发工具链；纯时间测量与无模型流式接收可独立运行。

项目使用 **Conda prefix 环境**，不是 `python -m venv`。重新建立环境时：

```bash
cd /path/to/MyTirex
~/anaconda3/bin/conda create -y --copy --prefix "$PWD/.venv" \
  --file configs/conda-linux-64.lock
.venv/bin/python -m pip install --no-deps --only-binary=:all: \
  -r configs/requirements-runtime.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
```

按实际安装位置替换 Conda 路径。车速回放需要恢复本地数据；模型推理还需要 `vendor/tirex/` 和 `models/tirex2/`。程序不会自动下载数据或权重，模型只从本地加载。环境版本、编译缓存和恢复说明见 [维护指南](docs/development.md)。

以下命令均在项目根目录执行。安装后的 `mytirex` 与 `.venv/bin/python -m mytirex` 等价；CLI 可用 `--root /path/to/MyTirex` 指定项目资产目录。

## 快速开始：独立检查时间测量

准备好环境和 `data/prepared/comma2k19/v1/` 数据后，先运行两个无模型短检查：

```bash
# 检查观测年龄、误差界和可用状态
.venv/bin/python -m mytirex eval-aoi \
  --max-series 1 --max-queries 16 \
  --profile jitter_loss_tail --scenario timestamp_noise

# 检查连续接收与发布
.venv/bin/python -m mytirex eval-stream --duration-s 1 \
  --profile jitter_loss_tail --scenario timestamp_noise
```

命令会打印结果目录，用浏览器打开其中的 `report.html` 查看报告，CSV 用于进一步分析。受控回放中的网络抖动、丢包和时钟误差来自模拟；真实 UDP 测量使用实际收发时间戳，两者的解释见 [时间测量指南](docs/aoi.md)。

## 通用信号预测接口

`TiRexModel.predict_signals()` 接收按信号名索引、位于同一时间网格上的一维数组。一个目标对应单变量预测，多个目标共同组成一组联合预测；`batch_size` 表示独立序列组数。

| 参数 | 内容 | 是否输出预测 |
|---|---|---|
| `targets` | 等长的目标历史，至少 2 个点 | 所有目标 |
| `past_covariates` | 与目标历史对齐的条件信号 | 否 |
| `future_covariates` | 历史加至少整个预测跨度的未来位置；未来未知处可为 `NaN` | 否 |

以下使用合成数组演示接口，运行时会加载当前版本地 GPU 模型：

```python
import numpy as np
from mytirex.inference.model import TiRexModel

model = TiRexModel.from_local(".")  # 也可传入 MyTirex 项目的绝对路径
t = np.arange(256, dtype=np.float32)

# 单变量预测
single = model.predict_signals({"steer": np.sin(t * 0.02)}, prediction_length=12)

# 联合预测速度和位置，油门作为历史协变量
joint = model.predict_signals(
    {"speed": 8 + 0.01 * t, "x_m": 0.2 * t},
    prediction_length=12,
    past_covariates={"throttle": 0.4 + 0.1 * np.sin(t * 0.02)},
    sample_period_us=25_000,
    units={"speed": "m/s", "x_m": "m", "throttle": "1"},
)
assert joint.status == "ok", joint.failure_reason
print(joint.target_names, joint.values.shape)  # ("speed", "x_m"), (2, 9, 12)
print(joint.values_at(joint.source_time_us + 25_000))  # 下一网格点的中位数
```

返回 `SignalForecast`，轨迹形状为 `[目标数, 9 个分位数, 未来步数]`，数值保留输入单位。默认采样周期为 25 ms，也可显式指定其他正整数微秒周期；模型本身不感知采样周期。

未来协变量只能包含预测时已经知道的计划或观测，未知未来用 `NaN`，数组仍须覆盖整个请求跨度。延迟、丢包和跨设备时钟场景使用 `CaptureSession(signal_schema=...)` 与 `StreamingRuntime`；接收端本地协变量可通过 `on_snapshot` 补入。完整示例和因果边界见 [通用信号指南](docs/signals.md)。

## 通用 UDP 接入与连续输出

示例配置 [signals_example.json](configs/signals_example.json) 声明 `throttle`、`brake`、`steer` 三个预测目标和 `speed` 历史协变量。两个终端均从项目根目录运行，即可用合成信号检查传输、时间测量与按名称发布，无需加载模型或读取 CAN 数据。

源端：

```bash
.venv/bin/python examples/signal_source.py \
  --bind 0.0.0.0 --port 5005 --duration-s 30 \
  --signals-config configs/signals_example.json
```

接收端：

```bash
.venv/bin/python -m mytirex stream-receive \
  --host 127.0.0.1 --port 5005 --duration-s 10 \
  --signals-config configs/signals_example.json --stdout
```

跨设备时将 `--host` 替换为源设备 IP。需要启用当前版预测时，手动添加 `--with-model`；可用 `--inference-period-ms 100 --publish-period-ms 25` 分别设置模型更新与发布间隔。首次模型请求可能初始化或编译内核，接收历史不足、时钟无效或轨迹不可用时发布保持值并记录原因。

`--stdout` 将状态逐行输出为 JSON，运行日志写入标准错误；状态同时保存在 `states.jsonl`。主要字段为：

| 字段 | 含义 |
|---|---|
| `prediction` | 按目标信号名索引的本次发布值；回退时为接收保持值 |
| `last_observation` / `delivery_hold` | 最新收到的目标值／最近一次预测交付时选定的目标值 |
| `aoi_us` / `aoi_lower_us` / `aoi_upper_us` | 观测年龄及误差界，单位微秒 |
| `publish_time_us` / `receiver_clock_id` | 实际发布时间及其接收端单调时钟身份 |
| `mode` / `reason` / `timing_valid` | 轨迹或保持值模式、原因及时间测量有效性 |
| `target_names` / `units` | 输出目标及单位 |

实际采集程序通过 `SignalSource.publish()` 发送对齐数据；同进程接入可向 `StreamingRuntime.process()` 提供接收与同步事件，通过 `on_publish` 获取状态。双向接入使用两个源适配器／端口和两个接收会话。CARLA 双向案例位于相邻项目 `carla_experiments`，接入方式见 [通用信号指南](docs/signals.md#carla-接入与验收范围)。

## 常用命令

下面的入口用于原车速实验、时间测量和性能检查。表中命令名均接在 `.venv/bin/python -m mytirex` 之后。

| 命令 | 用途 | 模型加载 |
|---|---|---|
| `doctor --smoke` | 检查本地环境、资产和真实 GPU 预测 | 当前版 |
| `eval-aoi` | 受控回放，检查年龄误差与区间覆盖率 | 默认不加载；`--with-model` 启用 |
| `eval-speed` | 五条固定车速序列的 175/300 ms 延迟回放 | 当前版 |
| `benchmark` | 同步测量 batch=1 单请求耗时与分位数 | 当前版 |
| `eval-aoi --with-model --delivery` | 评价计算完成时的预测补偿 | 当前版 |
| `eval-stream` | 按真实时间推进受控回放，评价发布频率和车速误差 | 默认不加载；`--with-model` 启用 |
| `sync-source` | 回放已有 CAN 数据并响应 UDP 时钟同步 | 不加载 |
| `sync-capture` | 采集真实 UDP 时间戳与观测年龄 | 不加载 |
| `replay-aoi` | 按接收顺序重放已记录的事件日志 | 不加载 |
| `stream-receive` | 接收 UDP 并持续发布；通用信号用 `--signals-config` | 默认不加载；`--with-model` 启用 |

以下为按需手动运行的当前版模型短检查；首次请求可能需要数十秒初始化或编译内核：

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

# 比较 25/100 ms 请求间隔，两者均每 25 ms 发布
.venv/bin/python -m mytirex eval-stream --with-model --duration-s 10 \
  --profile fixed_175 --scenario shared_clock_control \
  --inference-period-ms 25 --inference-period-ms 100
```

全部参数见 `.venv/bin/python -m mytirex <命令> --help`。默认无模型入口只测量时间或发布接收值，加入 `--with-model` 后才启用预测。

已有 CAN 数据的双端采集与日志重放见 [时间测量指南](docs/aoi.md#真实网络采集与重放)，调度、轨迹复用和车速流式实验见 [流式指南](docs/streaming.md)。

## 数据、模型与配置

| 位置 | 内容 |
|---|---|
| `data/regression/comma2k19/` | 五条各 180 秒的插值车速序列，用于固定延迟回归检查 |
| `data/prepared/comma2k19/v1/` | 保留原传感器时间的因果车速数据，用于时间与流式实验 |
| `data/raw/comma2k19/` | 已有原始数组和来源清单 |
| `models/tirex2/` | 本地 TiRex-2 权重 |
| `vendor/tirex/` | 当前版 TiRex 源码 |
| `references/current_speed/` | 与固定协议匹配的当前版历史预测 |

两类速度数据都使用 40 Hz 网格，即每 25 ms 一个点，但目标值定义不同，不能混用。接收与预测只使用当时已经到达的历史和当时已知的协变量；评分参考值由评估端独立读取。

主要运行配置为 `configs/experiment.json` 和 `configs/tirex_tuning.json`，默认历史窗口为 256 点。时间配置为 `configs/aoi.json`，经验证的副本为 `configs/aoi_frozen.json`；流式入口默认使用冻结副本，预测轨迹上限为 320 步。通用信号配置使用 `targets`、`past_covariates`、`future_covariates`、`units` 和 `sample_period_us`，示例见 `configs/signals_example.json`。

数据、当前版源码和权重身份由配置目录内的索引记录。封存源码和官方历史结果仅用于身份追溯，不进入日常运行配置、自动分析或最新结果指针。

## 项目结构

```text
MyTirex/
├── src/mytirex/
│   ├── cli.py             # 命令入口
│   ├── common.py          # 路径、身份与结果目录
│   ├── signals.py         # 通用信号 schema、历史视图和联合预测轨迹
│   ├── data/              # 数据读取、到达历史与历史参考
│   ├── inference/         # 当前版模型、GPU 与交付补偿
│   ├── timing/            # 时钟同步、接收缓冲区与 UDP
│   ├── streaming/         # 源适配器、异步运行、回放与实时发布
│   ├── evaluation/        # 诊断、时间、速度与性能评估
│   └── reporting/         # 指标、图表与 HTML
├── configs/               # 配置、环境锁与资产身份
├── docs/                  # 专题指南与维护说明
├── examples/              # 通用 UDP 源端示例
├── tests/                 # 回归和接口检查
├── data/                  # 本地数据
├── models/                # 本地权重
├── vendor/tirex/           # 当前版 TiRex
├── references/            # 历史身份与当前版回归参考
├── outputs/               # 关键结果、验收记录与历史身份索引
└── dist/                  # 分发包与内容校验清单
```

Python 接口位置见 [模块说明](docs/development.md#代码组织)。

## 结果与验证

结果写入独立的 `outputs/<时间>_<任务>/`，成功运行更新对应的 `latest_*.json`。无模型与模型流式结果使用不同指针。常见文件如下，具体内容随入口而异：

| 文件 | 内容 |
|---|---|
| `config.json` / `metadata.json` | 运行配置、源码／资产身份、状态和适用范围 |
| `report.html` / `plots/` | 可浏览报告与图表 |
| 查询 CSV、`metrics_summary.csv` 等 | 时间误差、车速误差或单请求性能明细与汇总 |
| `states.jsonl` | 连续发布状态，不含评分真值 |
| `publications.csv` / `inferences.csv` | 发布时刻、轨迹选值与后台请求记录 |
| `runtime_metrics.csv` | 发布频率、间隔、跳过时隙和轨迹覆盖等调度指标 |
| `events.jsonl` | 真实 UDP 接收、同步与查询日志，可用 `replay-aoi` 重放 |

真实 UDP 入口记录调度与观测年龄；其现场预测误差需要外部真值。项目自带的发布时刻车速评分使用带参考数据的受控回放。

已有核心结果包括完整时间 validation、当前版模型 validation、175/300 ms 各 1000 次单请求测量、交付补偿，以及 15 分钟流式实验。该流式实验实际发布率约为 **39.99 Hz**，使用后台完整推理和轨迹复用。结果及适用范围见 [时间验证记录](docs/aoi.md#已有验证结果)和 [结果清单](docs/development.md#结果保留规则)。两份完整 validation 的逐查询表以 `aoi_queries.csv.gz` 无损保存。

2026-10-12 的通用信号扩展记录包含 MyTirex 98 项检查、CARLA 双端 4 项接入检查，以及当前版 GPU 的单变量、联合目标、协变量短检查，详见 [通用信号验收记录](docs/development.md#通用信号扩展验收)。这些记录验证接口、因果接收和有限输出；通用信号精度、正式 CARLA 与真实多设备实验仍需使用者手动评估。历史实验绑定各自的源码与配置身份，不代表后续代码已重新完成全量评估。

本机运行回归检查：

```bash
.venv/bin/python -m pytest
```

数据、模型、结果、环境和编译缓存均不纳入 Git。`dist/` 中的 0.1.0 wheel 与项目包是原归档，尚未包含通用信号扩展；使用这些新接口应安装当前工作区源码。归档内容和恢复方法见 [打包说明](docs/development.md#打包与迁移)。

## 进一步阅读

- [通用信号与多端接入](docs/signals.md)：单变量、联合预测、协变量、对称接收接口和 UDP。
- [时间测量与计算时延补偿](docs/aoi.md)：时间定义、同步方法、双设备命令及已有验证。
- [流式输入与连续输出](docs/streaming.md)：异步调度、轨迹覆盖、状态接口及实验命令。
- [研究设计](docs/design.md)：研究目标、现有边界与后续方向。
- [维护指南](docs/development.md)：环境细节、模块位置、结果清理、验收与打包。
