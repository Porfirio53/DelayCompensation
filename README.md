# MyTirex

面向动态通信与计算延迟的 TiRex-2 在线车辆状态预测补偿研究工程。提供 Comma2k19 speed 因果回放、离线本地模型加载、真实 GPU 单请求测量，以及四时间戳同步、动态观测年龄、交付时刻补偿和异步连续状态发布。第一步见 [AoI 指南](docs/aoi.md)，第二步见 [计算时延补偿](docs/delivery.md)，第三步见 [流式系统与实验](docs/streaming.md)，研究方向见 [总方案](docs/总方案.md)。

开发、测试和日常评估只使用 `vendor/tirex/` 当前版。模型使用权限与后续工作约束见 [AGENTS.md](AGENTS.md)。时间测量验收默认不加载模型。

本机环境、源码、权重和数据已经迁移并完成 GPU 小规模运行。旧 `/home/patton/experiment2` 保留为只读来源；运行本项目不需要访问它。实际结果及验证范围见 [验证记录](docs/verification.md)，迁移来源见 [迁移说明](docs/migration.md)。

## 直接运行

在项目根目录使用已建好的 Conda prefix 环境，无需激活环境：

```bash
cd ~/MyTirex
.venv/bin/python -m mytirex doctor --smoke --verify-assets
.venv/bin/python -m mytirex eval-speed --sequence speed_seg01 --delay-ms 175 --max-queries 32
.venv/bin/python -m mytirex benchmark --batch-size 1 --samples 32 --warmup 5
.venv/bin/python -m pytest
```

`doctor --smoke` 运行真实 GPU 预测，检查所有模型参数的设备、实际包位置、完整 9 分位数轨迹和原生扩展链接；`--verify-assets` 重新校验全部迁移文件。第一次请求可能编译 FlashRNN/Triton 内核，日志和 metadata 会将它与后续请求区分。

入口按自身安装位置解析根目录。也可以从任意工作目录运行：

```bash
/home/patton/MyTirex/.venv/bin/python -m mytirex eval-speed \
  --root /home/patton/MyTirex --sequence speed_seg01 --delay-ms 300 --max-queries 32
```

## 环境准备

本工程使用 `~/MyTirex/.venv` **Conda prefix**，不是 `python -m venv`，也未复制旧环境。以下命令用于在 Linux x86_64 上重新建立环境；当前机器已经完成安装：

```bash
cd ~/MyTirex
~/anaconda3/bin/conda create -y --copy --prefix "$PWD/.venv" \
  --file configs/conda-linux-64.lock
.venv/bin/python -m pip install --no-deps --only-binary=:all: \
  -r configs/requirements-runtime.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m mytirex doctor --smoke --verify-assets
```

如果 Conda 安装在其他位置，替换第一条命令的程序路径。Conda lock 固定 Python、NVIDIA `cuda-12.8.1` 开发工具链及其宿主编译器的包构建；pip lock 固定实际推理、绘图及轻量测试所需依赖。XLSTM 的包元数据另声明了训练和 Notebook 依赖，本项目按已验证的推理导入闭包用 `--no-deps` 安装，因此常规 `pip check` 会报告这些未安装的可选工作流依赖。不要用无约束安装补进整套训练环境。

已验证版本：Python 3.12.15、PyTorch 2.8.0+cu128、CUDA runtime 12.8、nvcc 12.8.93、flashrnn 1.0.8、xlstm 2.0.3、mlstm-kernels 2.0.6。GPU 为 NVIDIA GeForce RTX 4080 Laptop GPU。运行需要 GPU 驱动以及 nvcc、开发头文件和开发链接库；只有 PyTorch CUDA runtime 不足以编译 FlashRNN。

`gpu.py` 会优先定位当前 prefix 的 nvcc，设置 `CUDA_HOME`，组合 Conda `targets/*/include`、NVIDIA wheel include、`FLASHRNN_EXTRA_INCLUDE_PATHS`、`XLSTM_EXTRA_INCLUDE_PATHS` 和开发库搜索路径。默认 `MAX_JOBS=4`，可用环境变量调整。内核缓存写入本项目 `.cache/{torch_extensions,triton,inductor}`，不复用旧工作空间编译产物。显式设置的 `CUDA_HOME` 会被尊重，应指向匹配的 12.8 工具链。

当前版使用 `models/tirex2/`，加载固定为 `compile=False`、`use_flex_attention=False`、`local_files_only=True`，同时关闭 Hub 联网。源码和本地权重已经复制；运行期间不下载模型。

## 速度评估

默认运行 `vendor/tirex/` 当前副本。子集直接回放指定查询，不要求先生成全量预测，也不依赖其他模型或旧结果指针：

```bash
.venv/bin/python -m mytirex eval-speed \
  --sequence speed_seg01 --delay-ms 175 --delay-ms 300 --max-queries 32
```

`--sequence` 和 `--delay-ms` 可重复指定。`--max-queries` 是每个“序列 × 延迟”的前 N 个评分查询。默认吞吐批量为 256；可用 `--batch-size` 改变回放批量。它不代表单请求延迟。

全量评估命令（本轮未自动执行）：

```bash
.venv/bin/python -m mytirex eval-speed
```

协议固定为五条各 180 秒、7201 点的 speed 序列，40 Hz / 25000 µs，速度单位 m/s；固定通信延迟为 175 和 300 ms，从第 5 秒评分。每组 7001 查询，一个模型全量 70010 查询。到达边界使用整数微秒，`source_time_us + delay_us <= query_time_us`；恰好到达的观测可用。模型只收到脱离真值数组的接收历史；GPU 预测完成、轨迹返回 CPU 后才附加目标真值。

预测分别取完整未来轨迹第 7 / 12 步的 0.5 分位数。上下文截取、padding、趋势检测、差分、归一化及还原沿用复制的已验证实现。当前配置 [tirex_tuning.json](configs/tirex_tuning.json) 为：

| 参数 | 固定值 |
|---|---:|
| context_length | 256 |
| pad_context | true |
| tta_diff | true |
| trend_window_points | 32 |
| trend_threshold | 1.5 |
| trend_min_r2 | 0.6 |

`pad_context=true` 保留 vendor 的行为：截取后的短上下文仍左补到 checkpoint 的 2048 点模型长度。256 点配置不意味着只计算 256 点 patch。

日常配置位于 [experiment.json](configs/experiment.json)：批量、线程数、种子和所选 tuning 文件。试验其他参数时复制 tuning JSON，通过 `--tirex-config configs/my_setting.json` 指定，保留默认回归配置；源码路径通过 [source_index.json](configs/source_index.json) 映射。当前源码或配置变化后，不再将不匹配的历史 current 预测作为复现目标。

## 当前版历史回归

默认评估结束后才读取当前版的可选历史预测。它只适用于原五条插值轨迹和当前固定延迟协议，校验内容、全部查询键、到达/真值字段、权重、源码及配置来源；`--reference none` 可以关闭历史回归读取。当前版参考来自配置匹配的 `20261009_172124_750261_C`。新语料或动态通信协议单独生成结果，不复用旧协议预测。

## 单请求性能

短检查和正式测量分别使用：

```bash
.venv/bin/python -m mytirex benchmark --batch-size 1 --delay-ms 175 --samples 32 --warmup 5
.venv/bin/python -m mytirex benchmark --batch-size 1 --delay-ms 300 --samples 32 --warmup 5
# 正式测量，需手动运行；本轮未执行
.venv/bin/python -m mytirex benchmark --batch-size 1 --delay-ms 175 --samples 1000 --warmup 20
.venv/bin/python -m mytirex benchmark --batch-size 1 --delay-ms 300 --samples 1000 --warmup 20
```

默认从 `speed_seg01` 的第 1200 个查询（30 秒）开始逐个调用，后续使用连续查询；可调整 `--sequence`、`--query-index`。计时包括接收历史复制、输入校验/截取/转换、模型调用、GPU 完成、CPU 轨迹取回和目标点选值，每次请求前后等待 GPU。入口只接受 batch=1。

`benchmark.csv` 保留每个样本；`metadata.json` 分开记录加载、第一请求（可能编译）、预热和稳定运行阶段，给出稳定样本数、P50/P95/P99 和峰值显存。已有磁盘缓存时，第一请求仍可能涉及 tracing、缓存加载或新形状的编译，不能将全部首请求耗时称为 nvcc 冷编译。32 样本尾部分位数只用于部署检查；正式尾延迟使用至少 1000 个稳定请求。此入口测模型请求端到端耗时；在线调度、时钟同步与连续发布由流式入口评价。

## 计算时延补偿

`eval-aoi --with-model --delivery` 以 batch=1 运行短轨迹，按实际 GPU 完成与 CPU 轨迹返回时刻重新选值，同时比较调用时刻与交付时刻误差。计算耗时初值为 17 ms，后续用 EWMA 更新，预测长度另留一个 25 ms 网格步余量。以下命令由使用者手动运行：

```bash
.venv/bin/python -m mytirex eval-aoi --with-model --delivery \
  --max-series 1 --max-queries 32 \
  --profile fixed_175 --profile fixed_300 --scenario shared_clock_control
```

结果保存在独立 `outputs/<时间>_delivery_current/`，成功指针为 `outputs/latest_delivery.json`；报告包含本轮实际计算耗时 P50/P95/P99、两种选值的交付误差和覆盖率。参数、时间域及 40 Hz 参考值的分辨率限制见 [计算时延补偿指南](docs/delivery.md)。

## 流式输入与连续输出

新增 `eval-stream` 和 `stream-receive`，共享单请求异步运行器。持续接收与 25 ms 状态发布独立于 GPU 推理；后台请求完成后截取最新历史，不累积待推理任务。轨迹长度覆盖观测年龄、计算耗时和下一次更新前的时间，发布时按当前时钟映射从缓存轨迹取值。

```bash
# 无模型的短流式检查
.venv/bin/python -m mytirex eval-stream --duration-s 1 \
  --profile jitter_loss_tail --scenario timestamp_noise
# 当前版 GPU 验收，由使用者手动运行；比较 40/10 Hz 请求节奏下的 40 Hz 发布
.venv/bin/python -m mytirex eval-stream --with-model --duration-s 10 \
  --inference-period-ms 25 --inference-period-ms 100
```

`states.jsonl` 在运行时逐行发布；`publications.csv`、`inferences.csv` 和 HTML 报告分别记录发布、推理和效果。`stream-receive --host <源端IP> --with-model --stdout` 接入现有 `sync-source`，标准输出可作为下游 JSON 流。真实 GPU 运行及正式评估由使用者执行；实现、验证记录及最多 15 min 的精简实验方案见 [流式指南](docs/streaming.md)。

## 数据与接口

| 新路径 | 用途 |
|---|---|
| `data/regression/comma2k19/` | 原四文件、五条**插值**速度轨迹，固定协议开发回归 |
| `data/prepared/comma2k19/v1/` | 已准备的**因果 ZOH** CAN 语料，原 train/validation/test 与旧数据隔离策略 |
| `data/raw/comma2k19/processed/` | 已解压原始数组；来源清单在同级 `download_manifest.json` |
| `references/current_speed/` | 当前版历史预测和原始 provenance，独立于新输出 |

没有复制约 45 GB 的 archives，也不重新下载、解压或准备已有数据。数据目录迁移由 [data_index.json](configs/data_index.json) 记录；原清单中的身份、SHA 和历史路径保持原样。历史路径仅用于追溯。

`data.py` / `arrival.py` / `prepared.py` 管理数据和接收可见历史，`model.py` 返回完整 `ForecastTrajectory`（9 分位数、未来速度、最后可见源时间、25000 µs 时间轴），`evaluation.py` 管理回放与评分，`report.py` 管理展示。`value_at()` 接受预测范围内的未来网格点；`value_at_offset()` 支持从已收到观测锚点到未来中位数轨迹的非网格插值。

准备语料通过 `PreparedCorpus.iter_examples()` 按需读取；仅将样本的 `.view` 传给模型，`.target_mps` 由评分端使用。接收视图保留源时间、接收时间、原传感器时间和未收到标记，不包含未来到达计划、最终丢包结局或目标速度。不生成重复窗口缓存。该语料尚未接入 `eval-speed` 的旧插值评分协议。

## 真实时间戳与动态 AoI

时间测量 validation 与当前版 GPU validation 已由使用者完成并复核，范围见 [AoI validation 记录](docs/aoi_validation.md)。当前需求首先检查算出的数据年龄与真实时间相差多少；`eval-aoi` 默认只测时间。双设备现场精度仍需独立参考。

新增 `eval-aoi` 使用已有因果 CAN 语料，分别比较固定 175/300 ms、直接相减时间戳、仅偏移同步、偏移／漂移同步和 oracle 选时。覆盖抖动、丢包、乱序、时钟偏移／漂移、同步中断、链路不对称和时间戳误差界低估；记录 grid/native AoI、区间、预测跨度及失败／回退。小规模检查：

```bash
# 仅时间测量，不加载 GPU
.venv/bin/python -m mytirex eval-aoi \
  --max-series 1 --max-queries 16 --profile jitter_loss_tail --scenario timestamp_noise
# 在相同轨迹上比较六种选时方法
.venv/bin/python -m mytirex eval-aoi --with-model --source current \
  --max-series 1 --max-queries 4 --profile jitter_loss_tail --scenario timestamp_noise --batch-size 4
```

默认使用 validation，256 点接收历史、10 Hz 评分查询。`--with-model` 才加载当前版 TiRex 并评价车速预测；纯时间测量不需要这个参数。新协议单独生成结果，不使用旧固定预测 CSV。

`sync-source` / `sync-capture` 在两台设备上进行真实 UDP 四时间戳交换和 CAN 回放采集；`replay-aoi` 按原事件顺序及采集配置重放。同步器将源单调时钟换算到接收域，并在每次查询重新计算最新观测的 AoI；模型的 `forecast_observation()` 可按测得的年龄上界确定预测长度。完整公式、误差界条件、双设备命令、正式 validation/test 评估和结果解释见 [docs/aoi.md](docs/aoi.md)。

## 结果查看

每次运行使用独立 `outputs/<时间>_<类型>_<源码>/`。速度回放输出 `predictions.csv`、汇总/分序列/失败指标、`config.json`、`metadata.json`、少量 PNG 和 `report.html`。误差包括 MAE、RMSE、Bias、绝对误差 P95/P99；有限回退保留评分并统计，GPU 不可用或初始整批失败会报错。

`outputs/latest_speed.json`、`latest_doctor.json`、`latest_benchmark.json` 仅指向对应类型最新**成功**的当前版运行，源码身份见该目录 metadata。失败目录保留错误，不覆盖成功指针。直接用浏览器打开结果目录中的 `report.html`，或用本机静态服务查看：

```bash
.venv/bin/python -m http.server 8000 --bind 127.0.0.1 --directory outputs
```

`predictions.csv` 中 `elapsed_s` 是批量耗时的摊销值，`batch_elapsed_s` 是实际同步批耗时；速度评分表中的历史 `inference_*` 列也沿用这个约定。单请求性能以 benchmark 的稳定阶段为准。

AoI 实验输出 `aoi_queries.csv`、时间／速度指标、配置身份、PNG 和 HTML，成功指针为 `outputs/latest_aoi.json`。实机采集另有 `events.jsonl`，采集和重放分别更新 `latest_aoi_capture.json`、`latest_aoi_trace.json`。真实网络 AoI 的估计误差需要独立同步参考；localhost 接口验证以及使用者运行的时间测量、当前版 GPU validation 已完成。后续正式评估继续由使用者手动运行。

权重、数据、预测 CSV、输出、环境和编译缓存由 `.gitignore` 排除；源码、配置身份索引和第三方许可证可纳入新仓库。已有模型外部异步流式系统，输入、发布和模型更新频率分别评价。模型内部增量状态复用仍属下一阶段；Adapter 网络与训练入口继续暂缓。
