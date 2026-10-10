# 本轮实际验证记录

验证日期：2026-10-10（Asia/Shanghai）。运行均使用 `/home/patton/MyTirex/.venv/bin/python`，GPU 为 NVIDIA GeForce RTX 4080 Laptop GPU。没有执行模型训练、70010 查询的全量评估或 1000 样本的正式长基准。

## 内容与协议

- 7462 个迁移文件、789091865 字节逐文件 SHA256 校验通过，包括两份完整 vendor、共享 checkpoint、原四文件、准备语料和全部已解压数组；来源和 SHA 见 [迁移清单](../configs/migration_manifest.json)。
- 最终轻量检查 **31 passed in 3.54s**，日志为 [validation_tests_final.log](../outputs/validation_tests_final.log)。覆盖整数微秒包含等号的到达边界、7/12 步及中位数、预测输入与未来数组隔离、原上下文和 vendor 趋势/差分/还原数值回归、原数据划分与隔离、乱序/缺失、trace 前缀、参考身份与全部查询键、有限回退评分、初始整批失败和源码串用保护。
- 当前 tuning 与 `20261009_172124_750261_C` 的源码/配置快照匹配；官方固定预测分别按其原 2048 点配置辨认。两个参考各自全部 70010 行经过键和到达/真值核验，该核验没有重新运行 70010 次推理。
- 新环境为独立 Conda prefix，nvcc 12.8.93、Conda target 开发头文件、NVIDIA wheel include 和开发链接库均核对。依赖实际导入位置在新 `.venv`，原生 FlashRNN 扩展在新缓存成功编译，`ldd` 无未解析库。

## GPU 诊断与子集评估

| 运行 | 实际范围 | 报告 |
|---|---|---|
| doctor current | 全资产校验；两种延迟各一次完整 GPU 轨迹 | [report.html](../outputs/20261010_145642_525322_doctor_current/report.html) |
| speed current | speed_seg01，175/300 ms 各 32 查询，共 64 | [report.html](../outputs/20261010_150659_092645_speed_current/report.html) |
| speed official | 独立进程，speed_seg01，两种延迟各 16 查询，共 32 | [report.html](../outputs/20261010_150756_787806_speed_official/report.html) |
| 独立性检查 | 从 `/tmp` 运行，关闭参考、batch=1，speed_seg02 / 175 ms 的 2 查询 | [审计记录](../outputs/validation_independence.json) |

所有上述真实 GPU 预测都成功，失败和回退均为零。模型所有参数设备为 `cuda:0`；current 的 `tirex2` 来自新 `vendor/tirex/src/tirex2`，official 来自新 `vendor/tirex_official/src/tirex2`。doctor 确认 9×7 和 9×12 的完整分位数轨迹，目标中位数分别为 32.5087890625 和 32.534080505371094 m/s。

doctor 加载 2.0284 s；空内核缓存的第一次预测 35.3123 s，缓存由 0 个文件增至 109 个、7605002 字节，包含初次 FlashRNN/Triton 部署工作。其第二次请求约 18.95 ms，只是烟测样本，不作为性能分位数。

两次 speed 运行生成了 predictions、汇总/分序列/失败指标、配置、metadata、参考比较、三个 PNG 和 HTML；相对资源链接已核对。子集指标如下，仅描述对应短片段：

| 新运行 | 延迟 ms | 查询数 | MAE m/s | RMSE m/s |
|---|---:|---:|---:|---:|
| current | 175 | 32 | 0.01461209 | 0.01939831 |
| current | 300 | 32 | 0.01683218 | 0.02185633 |
| official | 175 | 16 | 0.00853434 | 0.01137423 |
| official | 300 | 16 | 0.01101445 | 0.01305767 |

这些片段查询数量不同，不用于推断全量模型优劣。current 与配置匹配历史预测的最大绝对差为 **0.0001258850 m/s**、平均绝对差 0.0000274777 m/s；official 与固定官方预测最大差 **0.0000839233 m/s**、平均差 0.0000201464 m/s。配置、源码和权重身份吻合，实际批量布局与历史 256 不同，metadata 已将这个条件与数值差单独记录；未宣称位级复现。

独立性检查在预测过程中用 Python audit hook 禁止打开/列举旧项目路径，成功生成新输出且无旧目录访问尝试；参考读取由 `--reference none` 关闭。结合实际包位置和编译配置路径核查，确认新链路不需要旧导入、数据或结果指针。审计的范围为 Python 可审计文件操作，并非操作系统级完整 syscall 追踪。其命令与事件计数见审计 JSON，日志为 [validation_independence.log](../outputs/validation_independence.log)。

## batch=1 短性能测试

每种延迟运行一个独立进程，各记录 1 个第一请求、5 个预热请求和 32 个稳定请求。从 speed_seg01 的 query index 1200 开始连续查询。逐请求同步 GPU，计时包括接收历史复制、输入处理、模型调用、CPU 完整轨迹返回和目标点选择。

| 延迟 ms | 加载 s | 第一请求 s | 预热样本 | 稳定样本 | P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 175 | 3.5122 | 28.2117 | 5 | 32 | 14.2918 | 15.0667 | 15.2574 |
| 300 | 3.7483 | 26.4172 | 5 | 32 | 12.9616 | 13.8609 | 14.1622 |

两次稳定阶段的 PyTorch 峰值分配显存均为 **342128128 字节（326.28 MiB）**，峰值保留显存 **367001600 字节（350 MiB）**；这是进程内 PyTorch allocator 指标，不是整张 GPU 的总占用。

这两次 benchmark 开始时已有内核磁盘缓存，第一请求仍有进程内初始化、可能的 tracing/编译或缓存加载开销。记录的是第一次请求总耗时，不能把 26–28 秒全部归因于 nvcc 冷编译。稳定阶段未混入第一请求和预热样本。

报告与逐样本记录：

- 175 ms：[report.html](../outputs/20261010_150855_648928_benchmark_current/report.html)、[benchmark.csv](../outputs/20261010_150855_648928_benchmark_current/benchmark.csv)。
- 300 ms：[report.html](../outputs/20261010_151139_480822_benchmark_current/report.html)、[benchmark.csv](../outputs/20261010_151139_480822_benchmark_current/benchmark.csv)。

32 样本 P95/P99 仅作部署检查。正式长测试的直接命令已在 [README](../README.md) 给出；上述结果不能代表全部轨迹/趋势状态、持续在线排队或 40 Hz 模型更新能力。

## 历史全量指标与未执行范围

复制的官方固定预测在 175/300 ms 上历史全量 MAE 分别为 **0.02390364 / 0.03013425 m/s**；配置匹配 current 历史预测为 **0.01743197 / 0.02479049 m/s**。这些是导入的历史结果，未在本项目重新执行全量预测。

本轮没有执行新全量回放、正式 1000 样本性能测量、准备语料上的新模型评估、训练、真实时钟同步、异步在线系统或模型内部增量推理。准备语料只做内容、划分、因果处理和少量接收视图检查。交付时刻评分、动态跨度和连续发布机制列在 [实施路线](roadmap.md)。

新项目已初始化自己的 Git 仓库，未提交；旧仓库没有迁入。忽略规则已核对，权重、数据、预测 CSV、输出、环境和缓存不会进入源码版本管理。
