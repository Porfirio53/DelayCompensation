# 迁移说明

本次从 `/home/patton/experiment2` 只读核对并真实复制到 `/home/patton/MyTirex`。没有修改或清理旧项目，没有将软链接作为新项目的数据、源码或权重入口。新环境在目标目录重新安装，FlashRNN/Triton 在新缓存编译。

## 复制范围

共复制并逐文件比对 SHA256 **7462 个文件，789091865 字节**，明细及审阅来源见 [migration_manifest.json](../configs/migration_manifest.json)。`doctor --verify-assets` 只读取目标目录来复核这些内容。

| 旧相对路径 | 新相对路径 | 文件数 |
|---|---|---:|
| `TiRex/` | `vendor/tirex/` | 122 |
| `baselines/tirex-2/` | `vendor/tirex_official/` | 121 |
| `models/tirex2/` | `models/tirex2/` | 2 |
| `data/comma2k19/` | `data/regression/comma2k19/` | 4 |
| `data/adapter_training/comma2k19/v1/` | `data/prepared/comma2k19/v1/` | 1031 |
| `data/raw/comma2k19/processed/` | 同名新目录 | 6168 |
| `data/raw/comma2k19/download_manifest.json` | 同名新文件 | 1 |
| 官方固定 C 的预测、配置、metadata、两份指标 | `references/official_speed/` | 5 |
| `results/20261009_172124_750261_C/` 选定预测与配置快照 | `references/current_speed/` | 4 |
| 原 experiment 与 reference 配置 | `references/provenance/` | 2 |
| 当前 tuning JSON | `configs/tirex_tuning.json` | 1 |
| `docs/2607.01204v1.pdf` | 同名新文件 | 1 |

官方预测来源 `results/tirex_official/C` 的历史 run 身份为 `20261008_192857_796410_C`，目标文件保持原样；不迁移旧指针机制。当前参考按实际 tuning 快照选择 R²=0.6 的运行，不使用 R²=0.5/0.65 或旧 README 参数对应的其他运行。

准备语料保留 1028 个 NPZ block 与三个清单。身份仍为 `662e2eef6ef7d84c04f99d95de6bd7fb6627ef66fe704ddd771f6981f4f99828`；按车辆与录制日分组，旧五条开发轨迹所在日继续隔离，原 train/validation/test 分配不变：

| 集合 | day groups | routes | blocks | duration_s |
|---|---:|---:|---:|---:|
| train | 27 | 60 | 790 | 47274.575 |
| validation | 6 | 11 | 147 | 8801.25 |
| test | 6 | 10 | 91 | 5455.9 |

清单中的 `adapter` 历史 namespace、来源路径、内容身份和种子规则没有改名。中性目录通过 `configs/data_index.json` 映射，读取依据 NPZ 相对文件名，不解析旧环境路径。原插值回归数据与 ZOH 因果语料分别管理，不共享评分缓存。

## 提取及必要改造

| 原实现 | 本项目承接 | 必要改造 |
|---|---|---|
| `run_tirex.py`、`tirex_tuning_runner.py`、`replay.py` | `cli.py`、`evaluation.py` | 提取独立 speed 链路，删除其他方法结果/cache/多变量前置依赖 |
| `baseline_adapters.py` 中 TiRex 部分、`tirex_tuning.py` | `model.py` + 原样 vendor | 保留加载和上下文/趋势参数；从标量边界扩为带源时间的完整预测轨迹 |
| `cuda_runtime.py` | `gpu.py` | 复用 nvcc/include 逻辑，补开发链接库搜索和本项目编译缓存 |
| `dataset_api.py`、`method_api.py` | `data.py`、`ReceiverView`、`ForecastTrajectory` | 保留整数微秒到达规则，复制且隔离预测输入，不带目标真值 |
| `metrics.py`、`plots.py` | `metrics.py`、`report.py` | 保留误差算法、有限回退评分及绘图抽稀；移除 Ours 分支和大批图表 |
| `tirex_reference.py`、`tirex_diagnostics.py` | `references.py`、`doctor.py` | 固定参考身份核验、独立诊断，不迁移复杂的多方法缓存 |
| 通用 `adapter_training/corpus.py`、`dataset.py`、`arrival.py`、`common.py` | `prepared.py`、`arrival.py` | 提取 raw reader、ZOH、分组、按需窗口、到达模拟及接收视图；移除训练配置/特征缓存/Adapter 依赖 |

归一化、patch、TTA、趋势差分和还原仍由 vendor 完成，没有重新实现已验证的模型算法。对当前 vendor 的原有修改保持逐文件内容一致。新工程仅增加协议边界校验、身份记录、失败保护、统一 CLI、报告和真实 batch=1 计时。

模型加载保持 `compile=False`、`use_flex_attention=False`、`local_files_only=True`，原始 checkpoint 不训练。初始整批失败和无 GPU 明确报错；后续有限保持值回退有失败标记并继续评分。

## 固定身份与环境

两份源码的历史 base commit 都为 `32b6645e5f0aded2e93fd22de261bfa5de1e148c`。内容指纹使用原 cache 的相对文件 SHA 聚合规则，目录搬迁不会改变身份：

| 内容 | 身份 |
|---|---|
| 当前源码 | `54cd32d1ae8baf53c883cd5f11e9e610c308643d1e2fc8a92cbddcc14ffb8cc3` |
| 官方源码 | `a8c746e38e58e9300724d8160b1ca99a08e7d1efe8a037b00d424bf9f0cfcaab` |
| TiRex-2 revision | `30788a311699acba7e6393ad9700ea7b7cb9bb2d` |
| `model-config.yaml` SHA256 | `cddbe6d0be4919cb7b0cad747fb19a295264e129d1c0790cf62133b4cb5da727` |
| `model.ckpt` SHA256 | `184b160ffbe4c01a26beeba14015ff3507c7497e1f3577114187bbc1d19fcac1` |

权重索引从原 manifest 仅提取 TiRex-2；源码、数据、参考映射分别在 `configs/source_index.json`、`data_index.json`、`references.json`。原始 provenance JSON 不改内容。审阅过的背景文档和旧集成文件的 SHA 也在迁移 manifest 中记录；旧路径仅表示来源。

新 `.venv` 是 Conda prefix。按旧项目验证过的 NVIDIA `cuda-12.8.1` channel 包构建安装 nvcc、CUDA 开发头文件/库和宿主编译器，另安装固定的精简推理依赖。未复制旧 prefix、site-packages、脚本、编译扩展或环境缓存。锁与实际版本分别记录在 `configs/conda-linux-64.lock`、`requirements-runtime.lock` 和各运行 metadata 中；首次 GPU 验证确认扩展已在新目录编译并成功链接。

## 省略范围与许可证

省略原 archives、完整 results 历史、旧结果指针、其他模型源码/权重/预测、Adapter 网络与训练入口、特征训练缓存和网格搜索工具。没有批量准备数据或生成窗口缓存，没有执行训练和全量预测。

未复制旧 `.git`、环境、`__pycache__` 或编译产物。两份 vendor 的 [LICENSE](../vendor/tirex/LICENSE)、[NOTICE](../vendor/tirex/NOTICE) 及 inference 中的第三方许可证说明原样保留，官方副本亦保留对应文件。vendor 自带示例/文档作为原源码的一部分保存，但不会安装或运行它们涉及的额外工作流。

本次实际验证与新输出目录见 [验证记录](verification.md)。
