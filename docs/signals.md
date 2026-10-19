# 通用信号预测与多端接入

MyTirex 可将任意对齐的数值信号作为变量接入，包括控制量、转向、车轮转角、位置、姿态及运动状态。核心接口不规定名称、单位或车辆类型。控制量边界、角度展开、坐标系转换和不同采样率的对齐由调用项目处理；原车速实验协议继续保留。

## 信号角色与 TiRex-2 对应关系

| MyTirex 角色 | TiRex-2 输入 | 是否预测输出 |
|---|---|---|
| `targets` | `TimeseriesType.target`，形状 `[V_target, T]` | 所有目标联合输出 |
| `past_covariates` | `past_covariates`，形状 `[V_past, T]` | 仅作为历史条件 |
| `future_covariates` | `future_covariates`，形状 `[V_future, T + H]` | 已知值作为条件；未来未观测位置可为 `NaN` |

多个目标和协变量放入**同一个** `TimeseriesType`，让当前版 TiRex 的 variate mixer 使用同组信号之间的关系。`batch_size` 表示独立时间序列组的数量，不是一个流里的变量数。一个目标就是单变量预测；多个目标就是联合预测。实现沿用 [TiRex-2 论文](2607.01204v1.pdf)及当前版 `vendor/tirex/src/tirex2/model/types.py` 的接口，不修改 checkpoint 或模型结构。

未来协变量可提供预测时已经确定的计划、轨迹或周期量。依据 [TiRex-2](https://arxiv.org/abs/2607.01204) §3.1、§3.2 和附录 F 的 partial future observability，SDK 也允许未来区间仅部分已知：已知位置给出实际值，未知位置给出 `NaN`，直接交给当前模型的原生缺失值掩码。数组长度仍须覆盖整个请求跨度，历史部分须有限，禁止用无穷大表示缺失。

例如车辆状态历史截至 `s`，驾驶端在 `t > s` 已知自己从 `s` 到 `t` 的操作，这段操作可以作为相对 `s` 的未来协变量；`t` 后尚未发生的操作仍是 `NaN`。尚未收到的实际车辆状态也不能作为已知值。系统不会把最后一个操作值外推为假造的未来计划。

## 外部项目直接调用

在 MyTirex 环境中，或安装可编辑包并指定本地资产目录后：

```python
import numpy as np
from mytirex.inference.model import TiRexModel

# 仅当前版本地权重；从 configs 读取已有运行参数，无自动下载。
model = TiRexModel.from_local("/home/patton/MyTirex")

# 演示数组；实际接入时替换为采集的同一时间网格上的历史。
t = np.arange(256, dtype=np.float32)
steer_history = np.sin(t * .02)
single = model.predict_signals({"steer": steer_history}, prediction_length=12)

joint = model.predict_signals(
    {"speed": 8 + .01 * t, "x_m": .2 * t, "yaw_deg": steer_history},
    prediction_length=12,
    past_covariates={"throttle": .4 + .1 * steer_history, "brake": t * 0},
    sample_period_us=25_000,
)
assert joint.status == "ok", joint.failure_reason
# [3 targets, 9 quantiles, 12 steps]，原始输入单位。
print(joint.target_names, joint.values.shape)
print(joint.values_at(joint.source_time_us + 25_000))

# 确有未来计划时，数组包含 T 点历史 + 至少 H 点已知未来。
planned = model.predict_signals(
    {"x_m": .2 * t, "y_m": steer_history}, 12,
    past_covariates={"speed": 8 + .01 * t},
    future_covariates={"planned_steer": np.sin(np.arange(268) * .02)},
)
```

直调接口支持长度至少 2 的历史，按模型 `context_length` 截取最近窗口。未来协变量的历史也同步截取，保留对应的未来部分。`source_time_us` 可指定最近源 tick 的绝对单调时间；未指定时从零构建相对网格。返回 `SignalForecast.values`，形状 `[target, quantile, future_step]`，目标顺序见 `target_names`；应检查 `status` 和 `failure_reason`。

模型不包含采样周期知识；泛化接口允许其他正整数微秒网格，采样周期和单位对实际效果的影响留给调用系统实验。无模型地构建输入可用 `SignalView.from_history(...)`。需要丢包掩码、数据年龄或不同设备时钟时，使用以下接收会话。

## 连续补偿：任意一端都使用同一个接收接口

```python
from mytirex.signals import SignalSchema
from mytirex.timing.clock import ClockConfig
from mytirex.timing.receiver import TimingConfig
from mytirex.timing.transport import CaptureSession
from mytirex.streaming.runtime import StreamConfig, StreamingRuntime

schema = SignalSchema(
    targets=("throttle", "brake", "steer"),
    units={"throttle": "1", "brake": "1", "steer": "1"},
)
session = CaptureSession(
    "vehicle-receiver-clock", ClockConfig(), TimingConfig(),
    signal_schema=schema, sample_period_us=25_000,
    context_length=model.context_length, minimum_observations=32,
)
runtime = StreamingRuntime(
    session, StreamConfig(inference_period_us=100_000), model=model,
    on_publish=lambda state: print(state["prediction"]),
)
# 同一事件循环中：接收回调 runtime.process(packet_or_completed_sync_event)
# 然后 await runtime.run(duration_us=60_000_000)。
```

每个包有 `values: {signal_name: number}`，必须包含 schema 的所有目标与协变量。包内各变量属于同一个源网格 tick；接收缓冲只存实际收到的包。丢失位置是 NaN，推理前逐行做因果前向填充，前导未知位置仍保留 NaN。快照与源数据不共享可写存储。

事件包含 `source_clock_id`、`sequence`、`source_time_us`、`received_at_us`、`event_time_us`，以及可选 `native_source_time_us` 和 `send_time_us`；时间单位都是整数微秒。`source_time_us - sequence × sample_period_us` 在一个会话中必须固定。`received_at_us`、`event_time_us` 和查询时间属于接收端单调时钟。不同设备时钟通过已有四时间戳同步或独立有效映射换算，不能直接相减。

发布状态包含 `prediction`、`last_observation`、`delivery_hold` 三个按**目标信号名**索引的字典，还包含目标／协变量名称、单位、数据年龄、模式和轨迹覆盖范围。协变量不出现在预测输出里。旧车速调用及 `prediction_mps` 等字段继续保留；通用 schema 里的 `speed` 目标也提供这些字段，没有 `speed` 时为空。

连续输出复用原来的时间补偿：一次后台请求预测所有目标，覆盖数据年龄、排队、计算和下一次更新前的时间；每次发布按实际时刻从每个目标轨迹取值。时钟无效、历史不足、计划不够长或轨迹耗尽时发布已收到的保持值并给出原因。schema 在一个会话内固定，变更时应新建接收会话。

同步一次性补偿可用 `mytirex.inference.delivery.forecast_delivery(model, view, estimator)`；输入 `SignalView` 时 `prediction_start` 和 `prediction_delivery` 为目标字典，原 `ReceiverView` 输入仍返回车速标量。

## 不同进程／设备之间的 UDP 接入

外部采集程序可使用 `SignalSource`，源适配器不加载模型：

```python
import asyncio
from mytirex.streaming.source import SignalSource

source = SignalSource(schema, sample_period_us=25_000)
transport, _ = await asyncio.get_running_loop().create_datagram_endpoint(
    lambda: source, local_addr=("0.0.0.0", 5005),
)
# 该事件循环的采集回调中调用；变量来自你的采样／网格对齐层。
source.publish(
    {"throttle": .4, "brake": 0., "steer": .1},
    sequence=sequence, source_time_us=grid_tick_us,
    native_source_time_us=acquired_at_us,
)
```

接收端读取相同 schema 文件，既可以运行在车端，也可以运行在驾驶端：

```bash
.venv/bin/python -m mytirex stream-receive \
  --host SOURCE_DEVICE_IP --port 5005 --duration-s 60 \
  --signals-config configs/signals_example.json --with-model --stdout
```

先去掉 `--with-model` 即可检查链路与时间测量。接收端通过同步请求订阅源。不论信号方向，接收与推理接口相同。双向接入使用两个源适配器／端口和两个接收会话；多个来源不可混进同一个源时钟／网格。一个源可向多个订阅者发布。真实多设备 GPU 部署仍需在每个需要本地推理的接收端满足 README 的 CUDA 环境要求。

[`configs/signals_example.json`](../configs/signals_example.json) 给出三个控制目标和速度协变量。最短传输演示：

```bash
# 第一个终端：合成变量用于接口检查；实际项目替换采集函数。
.venv/bin/python examples/signal_source.py --bind 0.0.0.0 --duration-s 30
# 第二个终端：按名称连续输出，不加载模型。
.venv/bin/python -m mytirex stream-receive \
  --host 127.0.0.1 --duration-s 10 --signals-config configs/signals_example.json --stdout
```

有 `future_covariates` 的实时 schema 要在每个新锚点的包里给出 `future_values`，数组从**下一源网格 tick**开始，历史部分来自包的 `values`。长度必须覆盖该次补偿需要的全部步数；不足时 `future_covariates_horizon_limit` 阻止该次推理。单个 UDP JSON 包上限 8192 字节，过长计划或大量变量可由调用项目选择其他传输协议并调用 SDK。

上述 UDP 包协议仍要求有限计划值。接收端本地拥有协变量时，可通过 `StreamingRuntime.on_snapshot` 在提交推理前补充不可变 `SignalView` 的历史及 `known_future`；未知未来为 `NaN`。`features["covariate_revision"]` 随本地观测变化，用于在车辆历史未更新时也刷新请求。此回调只组装已知数据，实际预测仍调用同一个原生模型。

## CARLA 接入与验收范围

`/home/patton/carla_experiments/scripts/run_signals.sh` 分别接入驾驶端遥测和车端控制。默认驾驶端联合预测速度、前轮转角和位置姿态，车辆实际控制及运动状态作为历史协变量，本地已发生的操作作为部分已知未来协变量；车端联合预测油门、制动和转向。实验命令及控制模式见该项目 `docs/signals.md`。

开发检查验证形状、因果接收、联合分组、计划边界、时间选择、UDP 和旧车速兼容性；当前版 GPU 短检查验证三种输入能返回有限轨迹。本次不训练、不优化特定信号、不评估正式预测精度，不加载官方封存版。
