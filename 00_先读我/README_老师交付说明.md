# 老师交付用：训练集、验证集、测试集与主代码索引

生成日期：2026-08-29  
核查范围：`project/data` 全部数据文件、当前正式实验入口及其核心数据路径。

## 1. 先看结论

- 数据文件共 **273** 个，完整性检查错误 **0** 个；每个文件的路径、行数、列数和 SHA-256 见 `all_data_manifest.csv`。
- 当前唯一正式母数据源是 `project/data/raw/足尺环道车辙数据.xlsx`；`ruttingData_Final_Corrected.xlsx` 是旧版长表，不是当前 Stage 01 的输入。
- 母表共有 **2793** 个结构—时刻记录，其中 **2753** 个实测车辙值、**40** 个未测时刻，覆盖 **19** 个结构。
- 母表逐年总记录：{2016: 57, 2017: 437, 2018: 418, 2019: 380, 2020: 665, 2021: 836}；逐年实测记录：{2016: 57, 2017: 418, 2018: 407, 2019: 378, 2020: 660, 2021: 833}。
- Qwen 与下游预测器使用两套不同的时间划分，不能混写为一套 train/validation/test。
- 当前 `project/reports/v43/stage13_summary.json` 已确认 LOSGO **126/126** 个正式运行完成、0 个 smoke run、7/7 隔离审计通过。

## 2. Qwen 状态转移生成器数据

|用途|文件|记录数|目标年份|是否用于参数选择|
|---|---|---:|---|---|
|训练|`project/data/v31_qwen/qwen_train_2016_2017.jsonl`|304|2017|是，训练|
|验证|`project/data/v31_qwen/qwen_dev_2018.jsonl`|278|2018|是，选择 epoch|
|最终重拟合|`project/data/v31_qwen/qwen_refit_2016_2018.jsonl`|676|2017–2018|否，按已选 epoch 重拟合|
|2019 保留/生成目标|`project/data/v31_qwen/qwen_holdout_2019.jsonl`|277|2019|否，不参与 Qwen 参数选择|

对应的可审计宽表为 `train_windows.csv`、`dev_windows.csv`、`refit_windows.csv` 和 `holdout_windows.csv`。Qwen 不读取 2020 或 2021 目标。

## 3. TDR-PRSN / GRU 下游数据

下游数据原来由代码从 `rutting_master_v31.csv` 动态切分。本次已按原函数原样导出显式快照：

|用途|年份|样本数|精确数组|元数据|
|---|---|---:|---|---|
|训练基准 R100|2016–2019|1241|`derived_splits/train_2016_2019_seq6.npz`|`derived_splits/train_2016_2019_metadata.csv`|
|验证/早停|2020|660|`derived_splits/validation_2020_seq6.npz`|`derived_splits/validation_2020_metadata.csv`|
|冻结测试|2021|833|`derived_splits/test_2021_seq6.npz`|`derived_splits/test_2021_metadata.csv`|

每个 NPZ 均含 `X`、`y`、`previous`、`peak`；`X` 为 `(样本数, 6, 23)`。三个集合按 `Year + STR_name + Observation_Order` 检查，交集为零。

Q25/Q50/Q75/Q100、R100+Q50 和生成器对照的数据不应复制成新的“测试集”。它们只改变 2019 训练更新池，文件位于：

- `project/data/v31_updates/`：3 类生成器 × 4 个比例 × 5 个分配 × 2 种文件，共 120 个 CSV；
- 不带 `_states` 的 CSV 是下游训练更新行；带 `_states` 的 CSV 是生成轨迹审计文件；
- 2020 验证集和 2021 测试集在所有主实验条件中保持不变。

## 4. 结构 OOD / LOSGO 数据

`project/data/v43_losgo/` 包含 7 个 held-out 结构组。每折均有独立 Qwen train/dev/refit/2019 holdout 文件和 5 个 Q50 分配更新；held-out 结构不进入 Qwen 微调、QC 参考、缩放器、TDR 训练或模型选择。完整文件逐项见 `all_data_manifest.csv`。

## 5. 主代码入口

完整路径和准确行号见 `main_code_index.csv`。最关键的入口是：

1. `01_V31_构建母数据库_PyCharm.py` → 构建母表；
2. `02_V31_构建千问时序数据_PyCharm.py` → 构建 Qwen 四套时间数据；
3. `03_V31_训练并重拟合千问_PyCharm.py` → QLoRA 训练、2018 选择、2016–2018 重拟合；
4. `06_V31_等预算真实模拟替代主实验_PyCharm.py` → R100 与 Q25–Q100 主替代实验；
5. `11_V41_Additive_Augmentation_PyCharm.py` → R100+Q50、R50、R50+Q50；
6. `10_V40_MultiHorizon_OpenLoop_PyCharm.py` → 多预测跨度与 open-loop；
7. `12_V42_TDR_Qwen_Interaction_Validation_PyCharm.py` → 架构、容量和消融；
8. `13_V431_LOSGO_ONECLICK_PyCharm.py` → 7 折结构 OOD 正式流程。

注意：磁盘上的部分中文入口文件名被保存成 `#Uxxxx` 形式。`main_code_index.csv` 同时给出实际路径和解码后的中文显示名。

## 6. 交付时必须说明的外部依赖

- Qwen 基座模型未包含在数据代码包中。代码默认指向 `K:\ollama\Qwen3.5-9B-hf`，也可通过环境变量 `QWEN_BASE_MODEL_DIR` 指定。
- `project/models/v31_qwen/` 内只包含 LoRA adapter，不等于完整 Qwen 基座。
- 训练环境以根目录 `environment_qwen_tdr_v31.yml` 为准：Python 3.10、PyTorch 2.5.1、CUDA 12.4、Transformers 5.5.4、PEFT 0.19.1、bitsandbytes 0.49.2。
- `project/reports/v31/qwen_training/training_manifest.json` 中的 `K:` 路径是原训练机历史路径；本机 adapter 的实际位置是 `project/models/v31_qwen/`。

## 7. 不应作为当前正式入口的内容

- `qwen_tdr_v41_stage11_20260817/` 是早期 V4.1 快照，不包含后续完整 V4.2/V4.3 状态；
- 根目录 `PACKAGE_MANIFEST.json` 的旧验证字段不能代表当前正式完成状态；
- `project/data/raw/legacy_generate_dataset.py` 与 `ruttingData_Final_Corrected.xlsx` 属于旧数据路线；
- `project/manuscript/`、`project/figures/` 和大量 `reports/` 是论文与结果产物，不是训练/验证/测试输入。

## 8. 文件说明

- `all_data_manifest.csv`：全部数据文件、角色、大小、行列数、SHA-256、解析状态；
- `generator_split_summary.csv`：Qwen 四套数据的年份和记录数；
- `downstream_split_summary.csv`：下游显式快照及校验值；
- `main_code_index.csv`：入口与核心代码的相对路径、准确行号和用途；
- `teacher_data_code_handoff_20260829.zip`：数据、核心代码、测试、协议和本说明，不含模型权重、检查点、报告和图片。
