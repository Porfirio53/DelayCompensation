# 真实时间戳、动态观测年龄与时间测量误差实验

本阶段对应 [研究设计](design.md) 的第一步：把预设的 175/300 ms 延迟扩展为接收端实时测得的观测年龄。实现包括应用层四时间戳同步、两个单调时钟之间的偏移／漂移估计、实际收到的源历史、查询时刻动态 AoI、时间误差区间，以及可独立运行的受控实验和 UDP 实机采集。

代码已做单元测试、localhost UDP 采集／重放和少量真实 GPU 功能检查；记录见 [维护指南的验收记录](development.md#验收范围)。正式数据评估及双设备网络实验由使用者运行。默认速度目标是查询时刻 `q`；第二步的可选 `--delivery` 模式加入计算耗时与交付时刻补偿，运行说明见 [计算时延补偿](#计算时延补偿)。

2026-10-11 已复核使用者完成的时间测量与当前版 GPU 全量 validation；结果、限制和配置记录见 [已有验证结果](#已有验证结果)。需要追加 held-out 时间评估时使用 [aoi_frozen.json](../configs/aoi_frozen.json)，该文件与验证过的时间配置逐字节相同。模型使用限制见 [AGENTS.md](../AGENTS.md)。

## 数据选择与实验边界

采用已有的 `data/prepared/comma2k19/v1/`，即 Comma2k19 原生 CAN speed 经因果 previous-sample ZOH 得到的 40 Hz 数据。保留原传感器时间，按车辆／记录日期隔离的划分为 train 790、validation 147、test 91 个块；原五条插值回归序列相关的旧数据隔离规则保持在该语料身份中。配置索引和准备清单共同校验内容，默认数据身份为 `662e2eef6ef7d84c04f99d95de6bd7fb6627ef66fe704ddd771f6981f4f99828`。

这份数据适合研究“时间确定误差如何改变速度预测的取值时刻”：速度参考值、源时间和原生观测年龄都已有可追溯定义，无需新增下载或重新划分。速度评分目标定义为因果 ZOH CAN 参考值。它没有真实通信到达时间。`eval-aoi` 使用受控到达过程与时钟；`sync-source` 把同一 CAN 块按真实单调时钟发送，`sync-capture` 记录真实 UDP 时序。这两种证据在 metadata 中分别标明。

真实网络的 AoI **估计误差**需要独立校准的同步参考才能评分。只有四时间戳时，固定链路不对称与时钟偏差无法完全区分；采集报告因此给出点估计、误差区间和可用状态，不生成现场 AoI 精度真值。

## 时间域与 AoI

所有输入时间戳使用整数微秒，现场读取 `time.monotonic_ns() // 1000`。每个源端／接收端会话具有不同的 clock ID；源端重启会建立新会话并清空旧的同步与接收历史。墙上时间只用于输出目录命名。

| 字段 | 时钟域 | 含义 |
|---|---|---|
| `native_source_time_us` | 源端 | ZOH 取值对应的原生传感器记录时刻 |
| `source_time_us` | 源端 | 25,000 µs 规则采样网格；TiRex 轨迹锚点 |
| `send_time_us` | 源端 | UDP 负载编码／发送前的用户态时间标记，保存在采集日志中 |
| `received_at_us` | 接收端 | 用户态收到 datagram 后立即记录的时间 |
| `event_time_us` | 接收端 | 完整事件对程序可用的时刻，日志按此时刻递增 |
| `query_time_us` | 接收端 | 当前接收视图与 AoI 的查询时刻 |

网格时刻与实际发送标记分开：网格到发送之间的源端调度开销也属于观测年龄。`received_at_us` 包含内核队列到用户态读取的影响，不等于网卡硬件到达时间。

同步器估计源时间到接收时间的映射：

```text
R(S) = receiver_anchor + offset + rate × (S - source_anchor)
relative_drift_ppm = (rate - 1) × 1,000,000

grid_AoI(q)   = q - R(last_source_time)
native_AoI(q) = q - R(last_native_source_time)
```

使用锚点差值拟合，避免两个设备很大的 boot time 起点相减时丢失微秒精度。最新观测按源序号选择；迟到包可以填补保留窗口的缺口，不能把最新源时刻退回旧包。没有新包时，即使最近的 RTT 很小，AoI 仍随查询时间增长。接收端只知道窗口内哪些位置尚未收到，不把最终丢包结局当成已知特征。

`native_AoI - grid_AoI` 反映 ZOH 已经保持了多久。TiRex 的历史和轨迹以网格为基准，预测跨度使用 grid AoI；原生年龄用于质量诊断，不能把 hold age 再加到网格跨度上。

## 四时间戳同步与误差界

一次交换包含接收端发请求 `r1`、源端收请求 `s2`、源端发响应 `s3`、接收端收响应 `r4`。只有响应实际完成并可用后，它才进入同步器。方法依据 [RFC 5905 的四时间戳定义](https://www.rfc-editor.org/rfc/rfc5905.txt)，使用本项目的 UDP 协议进行应用层测量，不修改操作系统时钟。

名义 RTT 为 `(r4-r1) - (s3-s2)`；考虑相对速率后，处理时间项应为 `rate × (s3-s2)`。实现使用带漂移／时间误差余量的合理性检查，拒绝明显负 RTT、过大 RTT 或逆序可用事件。RTT 用于过滤同步交换，不能替代 AoI。

默认保存最近 64 次已完成交换，在每个 500 ms 桶中选择最低 RTT 的一个样本，再对源／接收中点做加权仿射拟合。至少有 3 次交换后可以确定偏移；拟合漂移还需要足够的桶及至少 2 s 源时间跨度。拟合速率限制在声明的 ±500 ppm 内。超过 3 s 没有新同步，状态转为 stale。

误差区间不依赖“链路对称”假设，也不把拟合残差当成可靠误差界。设源到接收的相对速率始终在 `1±d` 内、每个同步时间戳误差不超过 `e`。任意源时间 `s` 的映射由每次交换约束为：

```text
lower_R(s) = r1 + (s-s2) - d × abs(s-s2) - (2+d) × e
upper_R(s) = r4 + (s-s3) + d × abs(s-s3) + (2+d) × e
```

取所有保留交换的上下界交集，点估计限制在有效区间中，再加入源观测时间戳和查询读时误差，得到 AoI 上下界。若区间不相交、年龄为负、区间总宽超过配置上限，时间记录会明确失效。默认上限是 **区间总宽 100 ms**，不是半宽。

这些是声明条件下的确定性界限，不是统计置信区间。低 RTT 有助于缩小界限，但无法消除固定不对称偏差；现场的时间戳误差界与漂移界应根据设备和测量定义校准。`ClockMapping.known()` 提供独立校准映射的入口，可接入经过验证的外部同步参考。

## 动态预测跨度与对照方法

模型网格位于源时钟域，因此不能直接用接收时钟中的年龄除以 25 ms：

```text
source_offset = estimated_grid_AoI / estimated_rate
point_steps   = ceil(source_offset / 25,000)
upper_steps   = ceil(max(0, upper_grid_AoI) / minimum_rate / 25,000)
```

对于独立校准的 exact mapping，最小／最大速率均取已知速率。当前 checkpoint 最多支持 320 步，即名义源网格 8 s。`TiRexModel.forecast_observation(view)` 默认根据 `upper_steps` 生成覆盖不确定性的轨迹；`conservative=False` 使用点估计跨度。无效时间或越界请求会明确报错，调用者可采用有状态标记的保持值回退。

`ForecastTrajectory.value_at_offset()` 在最后已收到的观测值与未来中位数轨迹之间做线性插值，支持非网格查询和 0 步观测锚点，拒绝范围外取值。原固定协议的 `value_at()` 仍要求未来网格点。

受控评估保留六种年龄确定方法：

| `policy` | 查询时刻的确定方式 |
|---|---|
| `fixed_175` | 假设年龄／源预测跨度恒为 175 ms（7 步） |
| `fixed_300` | 假设恒为 300 ms（12 步） |
| `raw_timestamps` | 直接相减源、接收时间戳，忽略不同 boot time 与漂移 |
| `offset_only` | 在线校正偏移，速率保持为 1 |
| `synchronized` | 在线估计偏移与漂移，并保留误差区间 |
| `oracle` | 评分驱动器的真实时钟与年龄，作为选时误差对照 |

为了隔离年龄误差，`eval-aoi --with-model` 对每份接收历史生成一条**配置上限长度**的轨迹（默认 320 步），各方法只改变该轨迹上的取值时刻。生成长度由配置决定，不使用 oracle 年龄；评分端在模型返回 CPU 轨迹后才读取本次查询的目标真值。实际接入的 `forecast_observation()` 则使用测得的动态跨度。这种实验设计不用于比较短预测请求的推理加速效果。

不同 clock scenario 使用相同的包到达种子和模型随机种子；网络种子还包含冻结 split、series、profile、replicate。模型与时间同步器只接触当前已收到的历史／已完成交换，未来到达计划、最终丢包标记、真实时钟和未来速度留在仿真／评分端。

时间测量不依赖模型。使用 `--with-model` 时只加载当前版 TiRex，默认接收窗口和实际模型历史均为 256 点。需要研究完整长历史时，应另设接收上下文与起评时间，并使用相同的评分查询集。

## 网络与时钟实验配置

[configs/aoi.json](../configs/aoi.json) 默认 validation、256 点历史、至少 32 个实际观测、从源块第 7 s 开始、每 4 个网格点查询一次（名义 10 Hz），每种随机网络 1 个 replicate。`--query-stride 1` 可用全部 40 Hz 查询点。固定 profile 每次只生成一个 replicate。

| 默认网络 profile | 定义 |
|---|---|
| `fixed_175` / `fixed_300` | 固定源网格通信延迟 |
| `correlated_jitter` | 175 ms 基值，AR(1) 抖动，ρ=0.9、创新标准差 35 ms，截到 0–500 ms |
| `jitter_loss_tail` | 同上，另加 2% 丢包、1% 的 500–1500 ms 附加长尾；自然产生乱序 |

profile 参数直接来自冻结准备语料清单。额外的 `uniform_0_500` 可通过 `--profile` 选择。

| 默认时钟 scenario | 主要变化 |
|---|---|
| `shared_clock_control` | 同一起点／速率，对称 1 ms 路径，无同步网络抖动 |
| `clock_offset` | 接收时钟起点偏移 +500 ms |
| `clock_drift` | +500 ms 起点偏移，接收／源相对速率 +120 ppm |
| `timestamp_noise` | 上述漂移，加四个同步时间戳各自均匀 ±5 ms 误差 |
| `asymmetric_path` | 同步请求／响应路径 1/45 ms，各自 ±1 ms 抖动 |
| `underestimated_timestamp_error` | 注入 ±5 ms，估计器却只声明 ±1 ms，用于检验错误误差界 |
| `sync_outage` | 第 20–30 s 中断同步，另有 5% 同步响应丢失；车辆包仍按原 profile 到达 |
| `source_stamp_bias` | 观测网格／native 时间戳都偏 +10 ms，同步时间戳仍正常 |

未单列路径参数的场景默认请求／响应各 5 ms、各自均匀 ±2 ms 抖动。同步名义周期 250 ms，源端处理 20 ms。错误界匹配的误差实验默认把声明的同步时间误差界扩大到注入幅度加 1 µs 量化余量、源时间误差界扩大到源 stamp bias 的绝对值；metadata 保存每个场景的实际声明值。可用场景中的 `timestamp_error_bound_us` / `source_error_bound_us` 显式覆盖，包括故意低估的情况。

噪声作用于同步时间戳测量，源 stamp bias 单独作用于车辆观测时间戳。时钟 drift 会随着时间改变偏移，滚动拟合与过期状态分别覆盖持续同步和同步中断情形。较小的时间误差也可能使 `ceil` 得到相邻步数，实际速度取值使用连续跨度插值；应同时看 AoI 误差、步数覆盖和速度误差。

## 运行命令

在项目根目录运行，输出自动写到独立目录。以下短检查可先确认配置和 GPU 是否可用：

```bash
cd /home/patton/MyTirex
.venv/bin/python -m pytest

# 仅时钟与 AoI，不加载 GPU 模型
.venv/bin/python -m mytirex eval-aoi \
  --max-series 1 --max-queries 16 --profile jitter_loss_tail --scenario timestamp_noise

# 当前版小规模 GPU 轨迹检查，按需运行
.venv/bin/python -m mytirex eval-aoi --with-model --source current \
  --max-series 1 --max-queries 4 --profile jitter_loss_tail --scenario timestamp_noise --batch-size 4
```

正式评估由你手动执行。时间测量验收先在 validation 检查时钟界限与误差，再按需求在冻结配置上运行 test。纯时间命令如下；已完成的 validation 无需重跑：

```bash
# 全 validation；随机网络用 3 次重复
.venv/bin/python -m mytirex eval-aoi --split validation --replicates 3

# 需要追加 held-out 时间评估时，仍不加载模型
.venv/bin/python -m mytirex eval-aoi \
  --aoi-config configs/aoi_frozen.json \
  --split test --replicates 3
```

只有需要评价后续预测器时才加 `--with-model`；该入口仅使用当前版 TiRex，并同时输出时间指标与速度指标。单独验收时间测量时无需运行 GPU 评估。

默认 validation 全矩阵、1 次重复约有 249 万个查询组合、1494 万行 CSV；3 次重复时随机网络部分相应增加。可以按 `--profile` / `--scenario` 分开运行，或先用 `--max-series`、`--max-queries` 限定范围。CSV 流式写入，精确百分位保留数值误差向量，内存仍随查询数增加。`--max-queries` 包含不可用与回退查询；不要用它的短运行替代长尾／中断评估。

## 真实网络采集与重放

源端需要已有准备语料；接收端需要可运行本项目 CLI 的 Python 环境和配置。这两个入口只测时间，不调用 GPU。

源设备：

```bash
.venv/bin/python -m mytirex sync-source \
  --bind 0.0.0.0 --port 5005 --duration-s 90 --split validation
```

接收设备，将 IP 换成源设备地址：

```bash
.venv/bin/python -m mytirex sync-capture \
  --host 192.168.1.10 --port 5005 --duration-s 60 \
  --sync-interval-ms 250 --query-period-ms 25
```

`sync-source` 默认选冻结划分中按 series ID 排序的首个块，可用 `--series-id` 指定。原 CAN 时间在源端重基到本次回放的单调时钟，保留 native 与网格的时间差。源端按自己的 25 ms 网格回放；调度错过的点会跳过并计数，不用突发补发伪造正常节奏。块结束后继续提供同步响应，观测年龄随等待增长；重复发送新块需要新源会话。

接收端查询使用循环实际执行时的单调时间；`--query-period-ms 25` 是计划周期，应从查询 CSV 检查实际间隔。它不代表已实现 40 Hz 模型推理。可在受控网络条件下改变链路抖动、长尾和丢包，再保存各次 trace。

原样重放采集日志：

```bash
.venv/bin/python -m mytirex replay-aoi \
  --trace "outputs/<采集运行目录>/events.jsonl"
```

默认恢复 header 中的时钟／时间误差配置、上下文和最小观测数，使相同事件序列得到相同的 AoI 查询结果。需要研究测量界限的敏感性时，复制并调整 JSON，再显式使用 `--aoi-config configs/aoi_local.json`；输出会标为配置覆盖。采集／重放默认均为时间测量，车辆信号来自准备 CAN 回放。

## 输出与解释

| 文件 | 内容 |
|---|---|
| `aoi_queries.csv` | 全部查询；各 policy 的估计／真实 grid/native AoI、区间、连续跨度、步数、可用状态；GPU 模式另含预测、真值和回退原因 |
| `metrics_summary.csv` | 按源码、网络、时钟、policy 汇总 |
| `metrics_by_series.csv` | 加入 series、车辆／日期 group、route、replicate 的指标 |
| `metrics_timing_reasons.csv` | 全部时间不可用原因及计数 |
| `config.json` / `metadata.json` | 配置／选择、冻结数据身份、代码与模型身份、有效上下文、实际声明的误差界、执行状态 |
| `plots/aoi_trace.png` / `report.html` | 一段 AoI、误差界、动态步数和误差展示，以及汇总表 |
| `events.jsonl` | 实机采集额外保存：配置 header、完成的四时间戳、包与查询事件 |

时间指标包括 AoI MAE/RMSE/Bias/P95/P99（ms）、原生 AoI 误差、grid/native 区间覆盖率、步数吻合和跨度低估次数。`aoi_ms_*` 包含有点估计的不可用查询；`usable_aoi_ms_*` 只统计 `timing_valid=true`。覆盖率以有有限有效区间的查询为分母，并非全部查询的可用率。两者应结合 `query_count`、`timing_usable_count`、不可用原因计数解读。

GPU 模式的速度误差单位为 m/s。同步不可用、历史不足和跨度超限时，采用最后有限已收到值并继续评分；没有任何观测时留下缺失预测并计数。模型失败与选择时刻回退分别保留，初始模型整批失败会使运行失败，失败目录不覆盖成功指针。`ready` 表示历史与时间可用，还需要检查模型跨度上限。

受控评估 CSV 的 `ready` 和 `clock_*` 诊断字段保留共同的同步接收视图状态；各 policy 的可用性由其自身 `timing_valid`、`context_ready` 和 `status` 表达，实际选值跨度使用对应的 `source_offset_us`。

受控实验的 `true_*` 来自私有仿真时钟；实机采集没有这些真值，metadata 明确 `ground_truth_age_available=false`。实机报告的 AoI 分位数描述观测年龄本身，不是年龄估计误差。`elapsed_s` 是 GPU 批量耗时摊销，单请求时延仍以 `benchmark` 为准。

成功运行分别更新 `outputs/latest_aoi.json`、`latest_aoi_capture.json`、`latest_aoi_trace.json`。原固定延迟评估和参考入口独立保留。

## 代码接口

| 模块 | 责任 |
|---|---|
| `timing/clock.py` | 四时间戳、因果时钟映射、误差界、观测年龄与源跨度 |
| `timing/receiver.py` | 实际收到的包、序号窗口、native 时间和不可变接收视图 |
| `timing/transport.py` | 真实 UDP 同步／回放源／采集和事件重放 |
| `evaluation/simulation.py` | 私有仿真时钟、同步扰动及只推进已完成事件的驱动 |
| `evaluation/aoi.py` / `reporting/aoi_metrics.py` | 六种选时对照、模型隔离、流式查询输出与指标 |
| `inference/model.py` | 动态观测跨度接口、完整分位数轨迹和源网格插值 |

接入实际传感器时，应在源端形成规则网格并携带真实 native 时间、网格时间、序号和本次启动的 clock ID；接收端把收到的包交给 `TimestampedReceiver.ingest()`，把完成的同步交换交给 `ClockSynchronizer.update()`。每次查询取当时的 `clock.snapshot(q)` 与 `receiver.snapshot(q, mapping)`，仅把不可变视图传给模型。

交付补偿已在同一个接收单调时钟域记录调用与实际完成时间；异步运行器另记录 snapshot、start、delivery 和发布时刻。相关接口继续使用相同的源网格跨度与插值，见下文及 [流式指南](streaming.md)。

## 计算时延补偿

第二步沿用已有接收历史、时钟映射和轨迹插值，只在当前版 TiRex 外增加单请求耗时估计与完成后选值。入口是 `eval-aoi --with-model --delivery`，固定 batch=1。正式评估由使用者手动运行，功能验收使用有限查询的短命令。

使用者提供的两份 1000 请求报告中，175 ms 场景稳定 P50/P95/P99 为 **13.919/15.507/16.599 ms**，300 ms 场景为 **13.602/14.544/15.746 ms**。因此取向上取整的 **17 ms** 作为计算耗时初值；每次成功推理后更新 `estimate = 0.8 × estimate + 0.2 × actual`。这些测量来自当前版，原始报告保留在 `outputs/20261011_142805_811118_benchmark_current/` 和 `outputs/20261011_142913_491793_benchmark_current/`。

### 处理流程

1. 获取调用时刻 `q` 的接收视图及测得的 grid AoI。只使用已经收到的历史和已完成同步。
2. 以 AoI 上界加预计计算耗时确定预测长度，再留 **一个 25 ms 网格步**的余量。接收域的时间通过同步速率换算到源域；长度不使用真实时钟或未来标签。
3. 同步等待 GPU 完成、CPU 轨迹返回，记录单调时钟测得的 `compute_us`。启动时默认预热 5 次；这些请求不评分，首请求也不进入 EWMA。
4. 用实际完成时间选择轨迹中位数，支持非网格插值：

```text
delivery_time = q + queue_us + compute_us
h_eff         = measured_grid_AoI(q) + queue_us + compute_us
source_offset = measured_source_offset(q) + (queue_us + compute_us) / estimated_rate
```

例如同域 grid AoI 为 175 ms、实测计算耗时为 14 ms 时，交付跨度为 **189 ms**。选值位于 175 与 200 ms 预测点之间，不将计算耗时取整成 25 ms。本入口的回放没有推理任务队列，`queue_us=0`；`forecast_delivery()` 接受调用者提供的实际排队时间。第三步的异步运行器另记录线程调度与结果分发时间，并为两次更新之间的连续发布预留轨迹，见 [流式指南](streaming.md)。

`TiRexModel.forecast_observation()` 新增 `extra_delivery_us` 与 `margin_steps` 参数；默认值保留查询时刻预测。`inference/delivery.py` 的 `forecast_delivery(model, view, estimator, queue_us=0)` 封装计算耗时测量和实际完成时刻选值，不接触评分真值。计时包含跨度规划、上下文处理、模型调用、GPU 完成与 CPU 轨迹取回；接收事件回放及返回后的取值不计入该计算耗时。

### 手动运行

先运行一个小范围检查；首次请求仍可能需要约 24–25 s 的内核编译／初始化：

```bash
cd /home/patton/MyTirex
.venv/bin/python -m mytirex eval-aoi --with-model --delivery \
  --max-series 1 --max-queries 32 \
  --profile fixed_175 --profile fixed_300 --scenario shared_clock_control
```

需要检查动态 AoI 与漂移时，可手动运行：

```bash
.venv/bin/python -m mytirex eval-aoi --with-model --delivery \
  --max-series 1 --max-queries 100 --query-stride 1 \
  --profile jitter_loss_tail --scenario timestamp_noise
```

可用 `--compute-estimate-ms 17 --warmup 5` 调整启动参数；无需另建配置。去掉 `--delivery` 继续原来的六种时间确定方法实验；不加 `--with-model` 的时间测量入口仍不加载模型。

### 结果解释

输出位于独立的 `outputs/<时间>_delivery_current/`，成功指针为 `outputs/latest_delivery.json`。报告比较同一条短轨迹上的两种选值：

| 方法 | 交付时刻使用的预测 |
|---|---|
| `aoi_only` | 保留调用时刻 `q` 的选值，计算期间没有向后移动 |
| `compute_compensated` | 按实际计算耗时移动到 `delivery_time` 重新选值 |

两者共享历史、推理结果和交付时间，隔离计算时延补偿的选时效果。报告中的 `start_speed_mps_*` 保留调用时刻误差，`speed_mps_*` 改为交付时刻误差；`compute_ms_P50/P95/P99` 来自本轮实际单请求，不是批量摊销值。AoI 误差列仍评价调用时刻的时间测量。

`aoi_queries.csv` 保存 `compute_estimate_us`、`compute_us`、`queue_us`、`delivery_time_us`、`effective_horizon_us`、`selected_source_offset_us`、轨迹长度及覆盖状态。轨迹不能覆盖实际完成时刻时，保留 `delivery_horizon_exhausted` 并用最新已收到观测回退；不外推，也不再次推理。

交付参考按实际完成时刻换算到源域，从现有 **40 Hz 因果 ZOH** 真值中选此前最近一格；不插值未来真值，超出数据块末端的目标不评分。由于多数计算耗时小于 25 ms，调用与交付参考可能相同，这是该语料的时间分辨率限制；本阶段不据此声称亚采样级真实车辆状态精度或已实现在线 40 Hz 发布。

## 已有验证结果

以下是使用者完成的时间测量与当前版 GPU validation，复核日期为 2026-10-11。它们保留历史源码身份；本次目录整理没有重新执行全量评估。

### 完整性与身份

| 运行 | 任务数 | 查询组合数 | 六种方法的记录行数 | 总耗时 | 报告 |
|---|---:|---:|---:|---:|---|
| timing | 9,408 | 4,978,752 | 29,872,512 | 2.185 h | [report.html](../outputs/20261010_190625_942320_aoi_timing/report.html) |
| current | 9,408 | 4,978,752 | 29,872,512 | 4.028 h | [report.html](../outputs/20261010_234729_859814_aoi_current/report.html) |

147 个 validation 块来自 11 条记录、6 个车辆／日期 group；这些 group 的车辆标识相同。固定网络各运行一次，两种随机网络各重复三次，因此每个块有 `8 × (1 + 1 + 3 + 3) = 64` 个任务。

两组均为 `status=success`，没有 `max-series`、`max-queries` 或指定 series 的限制。每组全部 56,448 个分块／重复／场景／方法组合及其查询数量与冻结数据 inventory 计算的计划一致。两组的全部分块时间指标完全相同，汇总计数与分块之和一致，所有时间不可用原因均能对上数量。原始 CSV 另抽查了开头 1,000 个查询的时间、观测、真值和查询键；没有逐行比对全部原始 CSV。

数据清单、当前版 vendor 和 checkpoint 的内容身份再次核对通过，27 项复核全部通过。完整逐查询 CSV 已无损压缩为 `aoi_queries.csv.gz`，汇总、配置和身份未改，存储校验见各目录 `storage.json`。两个 validation 运行记录的共同集成源码 SHA256 为 `91b90cfc9e13726be53348daf44d82cf51ae0ee30089c01140a3132936770d39`，AoI JSON SHA256 为 `00807b1b1cd68a097756167b39dc4d377b640e799d120281a8c3987961d940a8`。原复核时的身份保存在 [audit.json](../outputs/validation_aoi_review/audit.json) 和历史 [配置记录](../configs/aoi_validation_freeze.json) 中；本次目录整理后的身份见 [短验收记录](../outputs/verification/20261011_cleanup/verification.json) 同目录各结果的 `metadata.json`。历史正式结果不代表修改后的代码已完成全量重跑。[review.py](../outputs/validation_aoi_review/review.py) 仅复核时间测量与当前版已有结果，不运行推理；分发包需另恢复两份完整压缩表后才能执行该复核。

当前版 GPU 运行实际使用 `cuda:0`，每个查询使用 256 点接收历史并生成 320 步轨迹。全部预测保留评分，推理失败、缺失预测和跨度超限均为零。有 28,672,089 行成功选值、1,200,423 行保持值回退；回退来自 `offset_only` 和 `synchronized` 两种方法的时间不可用，并非 GPU 失败。

耗时是全量批处理总耗时，不能用它除以查询数量评价在线单请求时延。

### 时间判断

下面均为 `jitter_loss_tail` 网络、`synchronized` 方法；可用误差只统计 `timing_valid=true` 的查询，区间覆盖率以有限区间为分母。

| 时钟场景 | 时间可用率 | 可用年龄 MAE ms | 可用绝对误差 P99 ms | 有限区间覆盖率 |
|---|---:|---:|---:|---:|
| shared_clock_control | 99.964% | ≈0 | ≈0 | 100% |
| clock_offset | 99.285% | 0.2084 | 0.6987 | 100% |
| clock_drift | 99.247% | 0.2080 | 0.7002 | 100% |
| timestamp_noise | 99.270% | 2.2176 | 7.6251 | 100% |
| asymmetric_path | 98.526% | 22.0011 | 22.3284 | 100% |
| source_stamp_bias | 98.526% | 10.0031 | 10.6151 | 100% |
| sync_outage | 85.519% | 0.2014 | 0.7129 | 100% |
| underestimated_timestamp_error | 21.436% | 1.2578 | 3.8414 | 87.808% |

在含时间噪声的代表性场景中，固定 175/300 ms 的年龄 MAE 为 **61.5547 / 124.5763 ms**；同步方法的全部有限点估计 MAE 为 **2.2169 ms**，可用点估计 MAE 为 **2.2176 ms**。两者统计范围不同。

声明界限与注入误差匹配的七个场景，在全部四种网络上的有限 grid/native 区间都覆盖真实年龄；所有场景的保守步数均未低估真实步数。这是本轮受控实验的结果，区间仍依赖声明的频率和时间戳误差条件，不能据此保证任意现场测量。

压力场景揭示了具体边界：

- 请求／响应路径设为 1/45 ms 时，点估计有约 22 ms 偏差。四时间戳本身无法区分固定路径不对称与时钟偏差；区间能覆盖这种偏差，现场精度仍需要独立参考。
- 20–30 s 同步中断时，系统会在最近同步超过 3 s 后标记过期。当前版模型评估有 89,122 个同步方法回退查询，占该场景全部网络查询的约 14.32%；这些查询保持最后收到的车速并继续计入速度评分。
- 时间误差实际为 ±5 ms、估计器却只声明 ±1 ms 时，同步方法在全部网络上的 622,344 个查询中有 490,666 个回退，约 78.84%。仍有有限区间未覆盖真实年龄；可用样本的较小平均误差不能掩盖这个问题。
- 观测时间戳偏快 10 ms 时，年龄点估计仍有约 10 ms 偏差；声明相应误差界使区间覆盖，但没有自动消除源传感器的标记偏差。
- 很新的观测可能因估计偏差得到负年龄，现有策略明确回退。理想共享时钟的随机网络也有少量浮点边界回退；本轮保留现有策略与结果，不在冻结后改变其处理规则。

完整表见 [synchronized_timing.csv](../outputs/validation_aoi_review/synchronized_timing.csv)。

### 当前版预测器接入结果

本节是已经完成的可选模型评估，检查选时误差对后续预测的影响。代表性 `jitter_loss_tail × timestamp_noise` 场景包含 233,379 个查询组合，汇总全部 147 个块和 3 次随机网络重复，包含保持值回退。

| 选时方法 | current MAE m/s |
|---|---:|
| fixed_175 | 0.02982425 |
| fixed_300 | 0.04248578 |
| raw_timestamps | 0.12690717 |
| offset_only | 0.02180585 |
| synchronized | 0.02182657 |
| oracle | 0.02177432 |

同步方法相对固定 175/300 ms 的 MAE 降低 **26.82% / 48.63%**。其平均绝对误差约为 **0.0786 km/h**，对应固定 175 ms 的 **0.1074 km/h**，绝对改善约 **0.029 km/h**；不能把相对百分比解释为巨大的实车速度变化。

这里的 oracle 只消除了选时误差：它与其他方法使用同一条当前版模型轨迹，不是完美车速预测器，也不是所有模型可能达到的最优误差。同步方法与这一对照的 MAE 差距为 **0.24%**，说明这个场景中选时误差已较小。

该场景在全部 6 个记录日期、3 次网络重复中均有改善，147 个块中有 145 个块的平均误差下降。这里只描述 validation 的稳定性，不把近邻查询或同一记录日期的块当成独立样本，也不宣称实车效果或统计显著性。

仅校正偏移已经取得大部分收益。在无时间噪声的漂移场景，拟合漂移把可用年龄 MAE 从 **0.9136 ms** 降到 **0.2080 ms**；加入 ±5 ms 时间噪声后，偏移对照的年龄 MAE **1.7777 ms** 小于漂移拟合的 **2.2176 ms**，速度 MAE 也略小。因此不能宣称拟合漂移在任何噪声水平下都优于仅偏移；时间配置继续保留两种方法。

可复核的汇总及日期／重复／分块表在 [validation_aoi_review](../outputs/validation_aoi_review/audit.json) 同目录。

### 时间配置与可选 held-out test

[aoi_frozen.json](../configs/aoi_frozen.json) 是验证过的 [aoi.json](../configs/aoi.json) 的逐字节副本，SHA256 相同。[aoi_validation_freeze.json](../configs/aoi_validation_freeze.json) 记录时间配置、数据、当前版模型参数、两份已有 validation 输出身份，以及原复核时的集成代码身份。它是历史复核记录，CLI 的 `--aoi-config` 应指向 `aoi_frozen.json`。

冻结配置自身保留原来的 `split=validation`、`replicates=1`；需要追加 test 时使用明确的命令行覆盖，与本轮 validation 的三次重复设置一致。单独验证时间测量使用以下命令，不加载模型：

```bash
cd /home/patton/MyTirex

.venv/bin/python -m mytirex eval-aoi \
  --aoi-config configs/aoi_frozen.json \
  --split test --replicates 3
```

按冻结 inventory，该 test 命令为 **91 个块、5,824 个任务、3,086,912 个查询组合、18,521,472 行方法记录**。只有另有预测器评估需求时才添加 `--with-model`，此入口仅使用当前版 TiRex。

test 评估由使用者运行；本次复核只核对数据内容身份并用 inventory 计算 test 任务数，没有分析 test 速度数组，也没有执行 test 推理。时间评估重点检查年龄误差、可用率和区间覆盖率；若后续调整时间方案，应把修改作为新的实验并重新验证。

本轮 validation 证据来自因果 ZOH CAN 参考速度及受控通信／时钟，只评价查询时刻。交付与发布时刻由独立入口评价；双设备现场时间精度仍需独立参考。
