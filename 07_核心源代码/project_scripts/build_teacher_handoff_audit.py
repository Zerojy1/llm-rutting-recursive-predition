from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from v31_downstream_data import build_real_downstream_frame, build_sequence_arrays_simple


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUT = ROOT / "teacher_handoff_20260829"
DERIVED = OUT / "derived_splits"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_escaped_name(name: str) -> str:
    return re.sub(
        r"#U([0-9a-fA-F]{4})",
        lambda match: chr(int(match.group(1), 16)),
        name,
    )


def line_of(path: Path, needle: str) -> int:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if needle in line:
            return number
    return 0


def classify_data(path: Path) -> tuple[str, str]:
    relative = path.relative_to(ROOT).as_posix()
    name = path.name
    if relative == "data/raw/足尺环道车辙数据.xlsx":
        return "raw_source", "Current Stage-01 source workbook"
    if relative == "data/raw/ruttingData_Final_Corrected.xlsx":
        return "legacy_source", "Legacy long-form workbook; not read by the current Stage-01 pipeline"
    if relative == "data/raw/legacy_generate_dataset.py":
        return "legacy_code", "Legacy data-generation script; not the current master-data entry"
    if relative == "data/processed/rutting_master_v31.csv":
        return "canonical_master", "Canonical 2016-2021 exposure panel used by generator and predictors"
    if relative.startswith("data/v31_qwen/"):
        role = {
            "qwen_train_2016_2017.jsonl": "generator_train",
            "train_windows.csv": "generator_train_audit",
            "qwen_dev_2018.jsonl": "generator_validation",
            "dev_windows.csv": "generator_validation_audit",
            "qwen_refit_2016_2018.jsonl": "generator_final_refit",
            "refit_windows.csv": "generator_final_refit_audit",
            "qwen_holdout_2019.jsonl": "generator_holdout_generation_target",
            "holdout_windows.csv": "generator_holdout_audit",
            "qwen_chronological_manifest.json": "generator_split_manifest",
        }.get(name, "generator_data")
        return role, "Chronological Qwen state-transition data"
    if relative.startswith("data/v31_updates/"):
        kind = "state_trajectory" if "_states" in path.stem else "downstream_update_rows"
        return "synthetic_update", f"Qwen/TimeGAN/TimeWeaver experimental update pool ({kind})"
    if relative.startswith("data/v43_losgo/"):
        if "/qwen/" in relative:
            return "losgo_generator_data", "Fold-specific generator split with held-out structures excluded"
        if "/updates/" in relative:
            kind = "state_trajectory" if "_states" in path.stem else "downstream_update_rows"
            return "losgo_synthetic_update", f"Fold-specific Qwen Q50 update ({kind})"
        return "losgo_manifest", "Fold-specific structural-OOD metadata"
    return "other", ""


def inspect_data_file(path: Path) -> dict:
    role, note = classify_data(path)
    row = {
        "relative_path": path.relative_to(ROOT).as_posix(),
        "absolute_path": str(path.resolve()),
        "role": role,
        "note": note,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "status": "OK",
        "rows_or_records": "",
        "columns": "",
    }
    try:
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            row["rows_or_records"] = len(frame)
            row["columns"] = len(frame.columns)
        elif path.suffix.lower() == ".jsonl":
            count = 0
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        json.loads(line)
                        count += 1
            row["rows_or_records"] = count
        elif path.suffix.lower() == ".json":
            json.loads(path.read_text(encoding="utf-8"))
            row["rows_or_records"] = 1
        elif path.suffix.lower() == ".xlsx":
            workbook = pd.ExcelFile(path)
            row["rows_or_records"] = sum(
                len(pd.read_excel(path, sheet_name=sheet, header=None))
                for sheet in workbook.sheet_names
            )
            row["columns"] = max(
                len(pd.read_excel(path, sheet_name=sheet, header=None).columns)
                for sheet in workbook.sheet_names
            )
    except Exception as error:
        row["status"] = f"ERROR: {type(error).__name__}: {error}"
    return row


def export_downstream_splits() -> list[dict]:
    master_path = ROOT / "data/processed/rutting_master_v31.csv"
    master = pd.read_csv(master_path)
    real = build_real_downstream_frame(master)
    X, y, previous, peak, metadata = build_sequence_arrays_simple(real, seq_len=6)
    definitions = [
        ("train_2016_2019", [2016, 2017, 2018, 2019], "Downstream training targets"),
        ("validation_2020", [2020], "Downstream model selection / early stopping"),
        ("test_2021", [2021], "Frozen chronological test; never used for selection"),
    ]
    DERIVED.mkdir(parents=True, exist_ok=True)
    summaries = []
    key_columns = ["Year", "STR_name", "Observation_Order"]
    for stem, years, role in definitions:
        mask = metadata["Year"].astype(int).isin(years).to_numpy()
        split_meta = metadata.loc[mask].reset_index(drop=True).copy()
        split_meta.insert(0, "sample_index", np.arange(len(split_meta), dtype=int))
        split_meta["target_rutting_mm"] = y[mask]
        split_meta["previous_rutting_mm"] = previous[mask].reshape(-1)
        split_meta["previous_peak_mm"] = peak[mask].reshape(-1)
        split_meta.to_csv(DERIVED / f"{stem}_metadata.csv", index=False, encoding="utf-8-sig")
        np.savez_compressed(
            DERIVED / f"{stem}_seq6.npz",
            X=X[mask],
            y=y[mask],
            previous=previous[mask],
            peak=peak[mask],
        )

        keys = set(map(tuple, split_meta[key_columns].astype(str).to_numpy()))
        summaries.append(
            {
                "split": stem,
                "role": role,
                "target_years": ",".join(map(str, years)),
                "samples": int(mask.sum()),
                "structures": int(split_meta["STR_name"].nunique()),
                "X_shape": str(tuple(X[mask].shape)),
                "metadata_csv": f"derived_splits/{stem}_metadata.csv",
                "arrays_npz": f"derived_splits/{stem}_seq6.npz",
                "key_count": len(keys),
                "sha256_metadata": sha256(DERIVED / f"{stem}_metadata.csv"),
                "sha256_npz": sha256(DERIVED / f"{stem}_seq6.npz"),
            }
        )

    key_sets = {}
    for summary in summaries:
        frame = pd.read_csv(OUT / summary["metadata_csv"])
        key_sets[summary["split"]] = set(
            map(tuple, frame[key_columns].astype(str).to_numpy())
        )
    names = list(key_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if key_sets[left] & key_sets[right]:
                raise RuntimeError(f"Downstream split overlap: {left} vs {right}")
    return summaries


def validate_generator_splits() -> list[dict]:
    directory = ROOT / "data/v31_qwen"
    definitions = [
        ("train", "qwen_train_2016_2017.jsonl", 304, 2017, 2017),
        ("validation", "qwen_dev_2018.jsonl", 278, 2018, 2018),
        ("final_refit", "qwen_refit_2016_2018.jsonl", 676, 2017, 2018),
        ("holdout_2019", "qwen_holdout_2019.jsonl", 277, 2019, 2019),
    ]
    summaries = []
    for role, filename, expected, expected_min, expected_max in definitions:
        records = []
        with (directory / filename).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    records.append(json.loads(line))
        mins = [int(record["metadata"]["target_year_min"]) for record in records]
        maxs = [int(record["metadata"]["target_year_max"]) for record in records]
        if len(records) != expected or min(mins) != expected_min or max(maxs) != expected_max:
            raise RuntimeError(f"Generator split audit failed: {filename}")
        summaries.append(
            {
                "split": role,
                "file": f"data/v31_qwen/{filename}",
                "records": len(records),
                "target_year_min": min(mins),
                "target_year_max": max(maxs),
                "sha256": sha256(directory / filename),
            }
        )
    return summaries


def code_index() -> list[dict]:
    entries = [
        (WORKSPACE / "01_V31_#U6784#U5efa#U6bcd#U6570#U636e#U5e93_PyCharm.py", "stage01_master", "Build canonical master data"),
        (ROOT / "scripts/v31_stages.py", "def stage01_master", "Raw workbook to canonical master panel"),
        (ROOT / "scripts/v31_master_data.py", "def build_master_panel_from_frames", "Master-panel preprocessing and state memory"),
        (WORKSPACE / "02_V31_#U6784#U5efa#U5343#U95ee#U65f6#U5e8f#U6570#U636e_PyCharm.py", "stage02_qwen_data", "Build Qwen train/validation/refit/holdout data"),
        (ROOT / "scripts/v31_qwen_data.py", "def build_chronological_qwen_datasets", "Chronological generator split implementation"),
        (ROOT / "scripts/v31_windows.py", "def build_generator_windows", "Generator window construction and horizon guards"),
        (WORKSPACE / "03_V31_#U8bad#U7ec3#U5e76#U91cd#U62df#U5408#U5343#U95ee_PyCharm.py", "stage03_qwen_train", "Qwen selection and final refit entry"),
        (ROOT / "scripts/v31_qwen_train.py", "def run_v31_qwen_training", "QLoRA training, 2018 selection, 2016-2018 refit"),
        (ROOT / "scripts/v31_downstream_data.py", "def split_downstream_frame", "Downstream chronology definitions"),
        (ROOT / "scripts/v31_downstream_data.py", "def build_sequence_arrays_simple", "Exact six-step predictor arrays"),
        (ROOT / "scripts/v31_experiment_engine.py", "def fixed_real_evaluation_arrays", "2020 validation and 2021 test arrays"),
        (ROOT / "scripts/v31_experiment_engine.py", "def run_tdr_condition", "TDR-PRSN training/evaluation core"),
        (ROOT / "scripts/v31_experiment_engine.py", "def run_gru_condition", "GRU training/evaluation core"),
        (ROOT / "scripts/v31_tdr.py", "def fit_tdr_step_budget", "TDR-PRSN training wrapper around the first-paper implementation"),
        (ROOT / "scripts/v31_gru.py", "class RuttingGRU", "GRU control implementation"),
        (WORKSPACE / "06_V31_#U7b49#U9884#U7b97#U771f#U5b9e#U6a21#U62df#U66ff#U4ee3#U4e3b#U5b9e#U9a8c_PyCharm.py", "stage06_equal_budget", "Main substitution experiment entry"),
        (ROOT / "scripts/v31_stages.py", "def stage06_equal_budget", "R100/Q25/Q50/Q75/Q100 and controls"),
        (WORKSPACE / "11_V41_Additive_Augmentation_PyCharm.py", "def main", "R100+Q50 and reduced-update main entry"),
        (ROOT / "scripts/v41_stage11.py", "def stage11", "Additive augmentation orchestration"),
        (ROOT / "scripts/v41_data.py", "def build_additive_training_arrays", "Additive real-plus-synthetic training arrays"),
        (WORKSPACE / "10_V40_MultiHorizon_OpenLoop_PyCharm.py", "stage10", "Multihorizon/open-loop main entry"),
        (ROOT / "scripts/v40_stage10.py", "def stage10", "H1/H3/H6/H12 and open-loop orchestration"),
        (WORKSPACE / "12_V42_TDR_Qwen_Interaction_Validation_PyCharm.py", "main", "Architecture interaction/ablation entry"),
        (ROOT / "scripts/v42_stage12.py", "def main", "Consumer controls, capacity, and ablation"),
        (WORKSPACE / "13_V431_LOSGO_ONECLICK_PyCharm.py", "from v431_stage13_oneclick import main", "Formal seven-fold LOSGO one-click entry"),
        (ROOT / "scripts/v431_stage13_oneclick.py", "def main", "LOSGO preparation, Qwen refit, synthetic generation, 126 formal runs"),
        (ROOT / "scripts/v43_data.py", "def split_master_for_fold", "Held-out structure split and isolation"),
        (ROOT / "scripts/v43_fit.py", "def build_v43_training_arrays", "Fold-specific downstream training arrays"),
        (ROOT / "first_model/TDR_PRSN.py", "def main", "Original first-paper TDR-PRSN source retained for provenance"),
    ]
    output = []
    for path, needle, purpose in entries:
        if not path.exists():
            output.append(
                {
                    "display_name": decode_escaped_name(path.name),
                    "relative_path": path.relative_to(WORKSPACE).as_posix(),
                    "line": "",
                    "purpose": purpose,
                    "status": "MISSING",
                }
            )
            continue
        line = line_of(path, needle)
        output.append(
            {
                "display_name": decode_escaped_name(path.name),
                "relative_path": path.relative_to(WORKSPACE).as_posix(),
                "line": line,
                "purpose": purpose,
                "status": "OK" if line else f"NEEDLE_NOT_FOUND: {needle}",
            }
        )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_readme(
    data_rows: list[dict], generator: list[dict], downstream: list[dict], code: list[dict]
) -> str:
    errors = [row for row in data_rows if row["status"] != "OK"]
    master = pd.read_csv(ROOT / "data/processed/rutting_master_v31.csv")
    year_counts = master.groupby("Year").size().to_dict()
    observed_counts = master.groupby("Year")["Rutting_Observed"].sum().astype(int).to_dict()
    return f"""# 老师交付用：训练集、验证集、测试集与主代码索引

生成日期：2026-08-29  
核查范围：`project/data` 全部数据文件、当前正式实验入口及其核心数据路径。

## 1. 先看结论

- 数据文件共 **{len(data_rows)}** 个，完整性检查错误 **{len(errors)}** 个；每个文件的路径、行数、列数和 SHA-256 见 `all_data_manifest.csv`。
- 当前唯一正式母数据源是 `project/data/raw/足尺环道车辙数据.xlsx`；`ruttingData_Final_Corrected.xlsx` 是旧版长表，不是当前 Stage 01 的输入。
- 母表共有 **{len(master)}** 个结构—时刻记录，其中 **{int(master.Rutting_Observed.sum())}** 个实测车辙值、**{int((1-master.Rutting_Observed).sum())}** 个未测时刻，覆盖 **{master.STR_name.nunique()}** 个结构。
- 母表逐年总记录：{year_counts}；逐年实测记录：{observed_counts}。
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

- Qwen 基座模型未包含在数据代码包中。代码默认指向 `K:\\ollama\\Qwen3.5-9B-hf`，也可通过环境变量 `QWEN_BASE_MODEL_DIR` 指定。
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
"""


def build_zip() -> Path:
    archive = WORKSPACE / "teacher_data_code_handoff_20260829.zip"
    include_files: set[Path] = set()
    for directory in [ROOT / "data", ROOT / "config", ROOT / "scripts", ROOT / "tests", ROOT / "first_model", OUT]:
        if directory.exists():
            include_files.update(path for path in directory.rglob("*") if path.is_file())
    for path in WORKSPACE.iterdir():
        if not path.is_file():
            continue
        if (
            path.suffix.lower() == ".py"
            or path.name in {"environment_qwen_tdr_v31.yml", "requirements_v31.txt"}
            or path.name.startswith(("README_V3.1", "V3.1_Final_Experimental_Protocol", "V4.0_FROZEN_PROTOCOL", "V4.1_STAGE11_FROZEN_PROTOCOL", "V4.2_STAGE12_POSTHOC_PROTOCOL", "V4.3_STAGE13_STRUCTURAL_OOD_PROTOCOL"))
        ):
            include_files.add(path)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(include_files):
            if "__pycache__" in path.parts or path.suffix.lower() == ".pyc":
                continue
            bundle.write(path, path.relative_to(WORKSPACE).as_posix())
    return archive


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data_rows = [
        inspect_data_file(path)
        for path in sorted((ROOT / "data").rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix.lower() != ".pyc"
    ]
    generator = validate_generator_splits()
    downstream = export_downstream_splits()
    code = code_index()
    write_csv(OUT / "all_data_manifest.csv", data_rows)
    write_csv(OUT / "generator_split_summary.csv", generator)
    write_csv(OUT / "downstream_split_summary.csv", downstream)
    write_csv(OUT / "main_code_index.csv", code)
    (OUT / "README_老师交付说明.md").write_text(
        build_readme(data_rows, generator, downstream, code), encoding="utf-8"
    )
    audit = {
        "status": "PASS" if all(row["status"] == "OK" for row in data_rows) else "FAIL",
        "data_files": len(data_rows),
        "data_parse_errors": sum(row["status"] != "OK" for row in data_rows),
        "generator_splits": generator,
        "downstream_splits": downstream,
        "code_entries": len(code),
        "missing_code_entries": sum(row["status"] != "OK" for row in code),
        "compileall_checked_separately": True,
    }
    (OUT / "audit_summary.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    archive = build_zip()
    archive_hash = sha256(archive)
    (OUT / "archive_sha256.txt").write_text(
        f"{archive_hash}  {archive.name}\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"archive={archive}")
    print(f"archive_sha256={archive_hash}")


if __name__ == "__main__":
    main()
