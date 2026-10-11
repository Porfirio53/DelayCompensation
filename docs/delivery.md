# 计算时延补偿与交付时刻评分

第二步沿用已有接收历史、时钟映射和轨迹插值，只在当前版 TiRex 外增加单请求耗时估计与完成后选值。入口是 `eval-aoi --with-model --delivery`，固定 batch=1。GPU 推理、预热和数据评估均由使用者手动运行。

使用者提供的两份 1000 请求报告中，175 ms 场景稳定 P50/P95/P99 为 **13.919/15.507/16.599 ms**，300 ms 场景为 **13.602/14.544/15.746 ms**。因此取向上取整的 **17 ms** 作为计算耗时初值；每次成功推理后更新 `estimate = 0.8 × estimate + 0.2 × actual`。这些测量来自当前版，原始报告保留在 `outputs/20261011_142805_811118_benchmark_current/` 和 `outputs/20261011_142913_491793_benchmark_current/`。

## 处理流程

1. 获取调用时刻 `q` 的接收视图及测得的 grid AoI。只使用已经收到的历史和已完成同步。
2. 以 AoI 上界加预计计算耗时确定预测长度，再留 **一个 25 ms 网格步**的余量。接收域的时间通过同步速率换算到源域；长度不使用真实时钟或未来标签。
3. 同步等待 GPU 完成、CPU 轨迹返回，记录单调时钟测得的 `compute_us`。启动时默认预热 5 次；这些请求不评分，首请求也不进入 EWMA。
4. 用实际完成时间选择轨迹中位数，支持非网格插值：

```text
delivery_time = q + queue_us + compute_us
h_eff         = measured_grid_AoI(q) + queue_us + compute_us
source_offset = measured_source_offset(q) + (queue_us + compute_us) / estimated_rate
```

例如同域 grid AoI 为 175 ms、实测计算耗时为 14 ms 时，交付跨度为 **189 ms**。选值位于 175 与 200 ms 预测点之间，不将计算耗时取整成 25 ms。当前回放没有推理任务队列，`queue_us=0`；`forecast_delivery()` 已接受调用者提供的实际排队时间，方便后续流式系统复用。

`TiRexModel.forecast_observation()` 新增 `extra_delivery_us` 与 `margin_steps` 参数；默认值保留查询时刻预测。`delivery.py` 的 `forecast_delivery(model, view, estimator, queue_us=0)` 封装计算耗时测量和实际完成时刻选值，不接触评分真值。计时包含跨度规划、上下文处理、模型调用、GPU 完成与 CPU 轨迹取回；接收事件回放及返回后的取值不计入该计算耗时。

## 手动运行

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

## 结果解释

输出位于独立的 `outputs/<时间>_delivery_current/`，成功指针为 `outputs/latest_delivery.json`。报告比较同一条短轨迹上的两种选值：

| 方法 | 交付时刻使用的预测 |
|---|---|
| `aoi_only` | 保留调用时刻 `q` 的选值，计算期间没有向后移动 |
| `compute_compensated` | 按实际计算耗时移动到 `delivery_time` 重新选值 |

两者共享历史、推理结果和交付时间，隔离计算时延补偿的选时效果。报告中的 `start_speed_mps_*` 保留调用时刻误差，`speed_mps_*` 改为交付时刻误差；`compute_ms_P50/P95/P99` 来自本轮实际单请求，不是批量摊销值。AoI 误差列仍评价调用时刻的时间测量。

`aoi_queries.csv` 保存 `compute_estimate_us`、`compute_us`、`queue_us`、`delivery_time_us`、`effective_horizon_us`、`selected_source_offset_us`、轨迹长度及覆盖状态。轨迹不能覆盖实际完成时刻时，保留 `delivery_horizon_exhausted` 并用最新已收到观测回退；不外推，也不再次推理。

交付参考按实际完成时刻换算到源域，从现有 **40 Hz 因果 ZOH** 真值中选此前最近一格；不插值未来真值，超出数据块末端的目标不评分。由于多数计算耗时小于 25 ms，调用与交付参考可能相同，这是该语料的时间分辨率限制；本阶段不据此声称亚采样级真实车辆状态精度或已实现在线 40 Hz 发布。
