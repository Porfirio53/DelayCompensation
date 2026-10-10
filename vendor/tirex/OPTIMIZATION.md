# 本项目中的 TiRex 优化副本

本目录是 `baselines/tirex-2/` 的独立复制，供延迟补偿精度优化。官方目录保留对照，C、D 通过外层 `experiment/tirex_tuning.py` 导入本目录的 `src/tirex2`，继续读取 `models/tirex2` 原权重。

当前只实现无需训练的上下文与趋势差分实验。新增 `PostProcessorConfig.trend_window_points`：默认 `None` 保持官方检测行为，设置整数后只对最近若干已到达目标值拟合趋势。其余差分、积分还原及模型计算保持原有逻辑。

日常参数调整位于根目录 `configs/tirex_tuning.json`，完整实验说明见 [README_EXPERIMENT.md](../README_EXPERIMENT.md) 第 11 节。C、D 每次重新运行本副本，与 checksum 固定的官方预测及已有其他方法结果比较。

```bash
.venv/bin/python run_tirex.py
.venv/bin/python run_multivariate.py
```

指令在项目根目录执行。所有完整模型实验由用户运行。
