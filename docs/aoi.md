# 真实时间戳、动态观测年龄与时间测量误差实验

本阶段对应 [总方案](总方案.md) 的第一步：把预设的 175/300 ms 延迟扩展为接收端实时测得的观测年龄。实现包括应用层四时间戳同步、两个单调时钟之间的偏移／漂移估计、实际收到的源历史、查询时刻动态 AoI、时间误差区间，以及可独立运行的受控实验和 UDP 实机采集。

代码已做单元测试、localhost UDP 采集／重放和少量真实 GPU 功能检查；记录见 [verification.md](verification.md)。正式数据评估及双设备网络实验由使用者运行。本阶段的速度目标是查询时刻 `q`；计算耗时与交付时刻补偿按方案下一步推进。

2026-10-11 已复核使用者完成的时间测量与当前版 GPU 全量 validation；结果、限制和配置记录见 [aoi_validation.md](aoi_validation.md)。需要追加 held-out 时间评估时使用 [aoi_frozen.json](../configs/aoi_frozen.json)，该文件与验证过的时间配置逐字节相同。模型使用限制见 [AGENTS.md](../AGENTS.md)。

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

### 两台设备的真实网络测量

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

## 代码入口与下一步接入

| 模块 | 责任 |
|---|---|
| `timing.py` | 四时间戳、因果时钟映射、误差界、观测年龄与源跨度 |
| `receiver.py` | 实际收到的包、序号窗口、native 时间和不可变接收视图 |
| `transport.py` | 真实 UDP 同步／回放源／采集和事件重放 |
| `aoi_simulation.py` | 私有仿真时钟、同步扰动及只推进已完成事件的驱动 |
| `aoi_experiment.py` / `aoi_metrics.py` | 六种选时对照、模型隔离、流式查询输出与指标 |
| `model.py` | 动态观测跨度接口、完整分位数轨迹和源网格插值 |

接入实际传感器时，应在源端形成规则网格并携带真实 native 时间、网格时间、序号和本次启动的 clock ID；接收端把收到的包交给 `TimestampedReceiver.ingest()`，把完成的同步交换交给 `ClockSynchronizer.update()`。每次查询取当时的 `clock.snapshot(q)` 与 `receiver.snapshot(q, mapping)`，仅把不可变视图传给模型。

这些边界可用于后续异步 worker。下一步需要在同一个接收单调时钟域记录 snapshot、enqueue、start、delivery，并把取值／评分目标扩展到实际交付时刻；本阶段已提供的源网格跨度和插值接口可继续使用。
