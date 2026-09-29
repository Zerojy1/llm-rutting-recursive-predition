# -*- coding: utf-8 -*-
"""
TDR-PRSN Research-Plus Pipeline
==============================

面向北京足尺路面环道车辙数据的完整程序：
1. 原始宽表读取与长表重构
2. STR1-STR19 结构厚度物理语义映射
3. 工程时序特征构造
4. 跨结构 GroupKFold 验证
5. TDR-PRSN 温载双状态概率残差状态空间网络训练
6. 预测、不确定性、损伤分量、风险概率与工程指标输出
7. 主文核心图筛选、状态增强 XGBoost 公平基线、验证频率加速
8. 材料覆盖约束的跨结构验证，避免未知材料外推与结构泛化混淆
9. 基线模型论文对比图：parity、演化曲线、残差箱线图、指标柱状图
10. 新增 LSTM 与 Transformer 深度学习序列基线，用于论文神经网络对比
11. 首观测点冷启动过滤：每个结构首个观测点仅用于状态初始化，不参与正式评价

推荐运行：
python tdr_prsn_full_pipeline.py --input "足尺环道车辙数据.xlsx" --output-dir outputs_tdr --epochs 220

如果你希望保持原始 0.1 mm 数值不换算成 mm：
python tdr_prsn_full_pipeline.py --input "ruttingData_Final_Corrected.xlsx" --rutting-scale-to-mm 1.0

作者建议：
- 原始论文数据表头注明车辙单位为 0.1 mm，因此默认 rutting_scale_to_mm=0.1。
- 若你的论文和前期代码已经把数值直接当 mm 使用，请显式设置 --rutting-scale-to-mm 1.0。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 该数据集属于小样本工程时序数据。CPU 多线程会带来明显调度开销，
# 固定为 1 个线程通常更快、更稳定；使用 GPU 时不受影响。
torch.set_num_threads(1)

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


# =============================================================================
# 0. 配置
# =============================================================================

SEMANTIC_COLS: List[str] = [
    "Surf_AC13_I_SBS", "Surf_AC13_II_SBS", "Surf_SMA13", "Surf_PAC13",
    "Mid_AC20_AH30", "Mid_AC20_SBS", "Mid_AC20_AH50",
    "Bot_AC25_AH30", "Bot_AC25_AH50", "Bot_AC25_AH70", "Bot_AC25_Re_AH70", "Bot_AC10_SBS",
    "Base_CBG_A", "Base_CBG_B", "Base_CS", "Base_CC", "Base_LCC", "Base_GA",
]

ENV_COLS: List[str] = [
    "Cum_Load_10k", "Delta_Load_10k", "Avg_Temp_C", "Delta_Temp_C", "Temp_MA3_C",
]

FEATURE_COLS: List[str] = ENV_COLS + SEMANTIC_COLS


@dataclass
class Config:
    input: str
    output_dir: str = "outputs_tdr_prsn"
    rutting_scale_to_mm: float = 0.1
    seq_len: int = 6
    folds: int = 5
    epochs: int = 220
    batch_size: int = 64
    lr: float = 1.5e-3
    weight_decay: float = 1e-4
    d_model: int = 64
    dropout: float = 0.10
    patience: int = 40
    seed: int = 42
    risk_threshold_mm: float = 10.0
    threshold_list_mm: str = "8,10,12"
    run_xgb: bool = True
    run_dl_baselines: bool = True
    dl_baseline_epochs: int = 160
    device: str = "auto"
    save_excel: bool = True
    run_ablation: bool = True
    ablation_epochs: int = 120
    ablation_variants: str = "A2_flat_thickness,A4_no_thermal_gate,A5_no_reversible,A7_no_obs_adapter"
    val_every: int = 5
    figure_mode: str = "main"  # main: 只输出主文核心图；all: 输出全部诊断图
    delta_loss_weight: float = 0.25
    delta_loss_scale_mm: float = 8.0
    # 双状态分解强化项：冷启动过滤后，早期剧烈波动样本减少，
    # 可逆响应分支对RMSE的直接贡献容易被观测残差适配器吸收。
    # 下面三个参数让永久-可逆状态在训练中获得更明确的监督，
    # 同时降低对可逆项幅值的过强惩罚，使A5消融更能真实检验双状态机制。
    dual_state_loss_weight: float = 0.05
    reversible_aux_weight: float = 0.03
    reversible_reg_weight: float = 5e-4
    min_physics_fusion_gate: float = 0.15
    enable_residual_calibration: bool = True
    residual_calibrator_strength: float = 0.75
    residual_calibrator_estimators: int = 260
    use_rc_for_main_outputs: bool = True
    split_mode: str = "material_aware"  # material_aware/groupkfold
    material_anchor_structures: str = "auto"  # auto 或逗号分隔 STR 名称，例如 STR4,STR5
    save_split_report: bool = True
    # TDR-PRSN 是以上一期实测车辙作为状态锚点的递推预测模型。
    # 每个结构的首个观测点没有真实上一期状态，容易形成冷启动低估；
    # 默认将其作为状态初始化点，不参与训练、验证、指标和论文图。
    exclude_initial_observation_targets: bool = True

    @property
    def thresholds(self) -> List[float]:
        values = []
        for item in str(self.threshold_list_mm).split(","):
            item = item.strip()
            if item:
                values.append(float(item))
        return values

    @property
    def ablation_variant_list(self) -> List[str]:
        return [v.strip() for v in str(self.ablation_variants).split(",") if v.strip()]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def log(msg: str) -> None:
    print(msg, flush=True)


# =============================================================================
# 0.5 材料覆盖约束的跨结构划分
# =============================================================================

def _sort_str_names(names: List[str]) -> List[str]:
    """按照 STR 数值编号排序，非标准名称放在最后。"""
    def key(x: str) -> Tuple[int, str]:
        try:
            return int(str(x).replace("STR", "")), str(x)
        except Exception:
            return 10_000, str(x)
    return sorted([str(n) for n in names], key=key)


def build_structure_material_dict() -> Dict[str, set]:
    """基于内置结构映射表，返回每个结构中非零厚度材料集合。"""
    struct_map = build_structure_map()
    material_dict: Dict[str, set] = {}
    for _, row in struct_map.iterrows():
        sid = str(row["STR_name"])
        material_dict[sid] = {c for c in SEMANTIC_COLS if float(row[c]) > 1e-12}
    return material_dict


def parse_anchor_structures(anchor_arg: str, available_structs: List[str], material_dict: Dict[str, set]) -> List[str]:
    """解析材料锚点结构。auto 时自动固定含唯一出现材料的结构在训练集。"""
    available = set(available_structs)
    if str(anchor_arg).strip().lower() in {"", "none", "no", "false"}:
        return []
    if str(anchor_arg).strip().lower() != "auto":
        anchors = [x.strip() for x in str(anchor_arg).split(",") if x.strip()]
        return _sort_str_names([x for x in anchors if x in available])

    mat_to_structs: Dict[str, List[str]] = {m: [] for m in SEMANTIC_COLS}
    for sid in available_structs:
        for m in material_dict.get(sid, set()):
            mat_to_structs[m].append(sid)

    anchors = set()
    for m, holders in mat_to_structs.items():
        holders = [h for h in holders if h in available]
        if len(holders) == 1:
            anchors.add(holders[0])
    return _sort_str_names(list(anchors))


def _coverage_missing(train_structs: List[str], val_structs: List[str], material_dict: Dict[str, set]) -> List[str]:
    train_mats = set().union(*(material_dict.get(s, set()) for s in train_structs)) if train_structs else set()
    val_mats = set().union(*(material_dict.get(s, set()) for s in val_structs)) if val_structs else set()
    return sorted(list(val_mats - train_mats))


def _repair_validation_structures(
    train_structs: List[str],
    val_structs: List[str],
    material_dict: Dict[str, set],
) -> Tuple[List[str], List[str], List[str]]:
    """若验证集包含训练集中未覆盖材料，则把相关验证结构移回训练集，直到材料覆盖成立。"""
    train_set = set(train_structs)
    val_set = set(val_structs)
    moved: List[str] = []

    while True:
        missing = _coverage_missing(_sort_str_names(list(train_set)), _sort_str_names(list(val_set)), material_dict)
        if not missing:
            break
        # 从验证结构中选择一个覆盖缺失材料最多的结构移回训练集。
        candidates = []
        for sid in val_set:
            mats = material_dict.get(sid, set())
            score = len(set(missing) & mats)
            if score > 0:
                candidates.append((score, len(mats), sid))
        if not candidates:
            break
        candidates.sort(reverse=True)
        sid_to_move = candidates[0][2]
        val_set.remove(sid_to_move)
        train_set.add(sid_to_move)
        moved.append(sid_to_move)
        if not val_set:
            break

    return _sort_str_names(list(train_set)), _sort_str_names(list(val_set)), _sort_str_names(moved)


def make_structure_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: Config,
    out_dir: Optional[Path] = None,
    report_prefix: str = "cv",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    构造跨结构验证划分。

    split_mode=groupkfold：原始 GroupKFold，仅保证同一 STR 不泄漏；
    split_mode=material_aware：材料覆盖约束 GroupKFold，保证 validation materials ⊆ training materials。
    该策略避免把“未见结构组合泛化”与“未知材料体系外推”混在一起。
    """
    mode = str(getattr(cfg, "split_mode", "material_aware")).strip().lower()
    unique_structs = _sort_str_names(np.unique(groups).tolist())
    n_splits = min(int(cfg.folds), len(unique_structs))
    if n_splits < 2:
        raise ValueError("跨结构验证至少需要 2 个不同结构。")

    report_rows: List[Dict[str, object]] = []
    occurrence_rows: List[Dict[str, object]] = []
    material_dict = build_structure_material_dict()

    mat_to_structs: Dict[str, List[str]] = {m: [] for m in SEMANTIC_COLS}
    for sid in unique_structs:
        for m in material_dict.get(sid, set()):
            mat_to_structs[m].append(sid)
    for m in SEMANTIC_COLS:
        occurrence_rows.append({
            "Material": m,
            "Occurrence_n_structures": len(mat_to_structs[m]),
            "Structures": ",".join(_sort_str_names(mat_to_structs[m])),
        })

    if mode in {"groupkfold", "original", "unconstrained"}:
        gkf = GroupKFold(n_splits=n_splits)
        splits = []
        for fold, (tr, va) in enumerate(gkf.split(X, y, groups), start=1):
            train_structs = _sort_str_names(np.unique(groups[tr]).tolist())
            val_structs = _sort_str_names(np.unique(groups[va]).tolist())
            missing = _coverage_missing(train_structs, val_structs, material_dict)
            report_rows.append({
                "Fold": fold,
                "Split_Mode": "groupkfold",
                "Train_Structures": ",".join(train_structs),
                "Validation_Structures": ",".join(val_structs),
                "Anchor_Structures": "",
                "Moved_Back_To_Train": "",
                "Missing_Materials_In_Train": ",".join(missing),
                "Material_Coverage_OK": len(missing) == 0,
                "N_Train": int(len(tr)),
                "N_Validation": int(len(va)),
            })
            splits.append((tr, va))
    elif mode in {"material_aware", "material-aware", "coverage"}:
        anchors = parse_anchor_structures(getattr(cfg, "material_anchor_structures", "auto"), unique_structs, material_dict)
        candidate_structs = [s for s in unique_structs if s not in set(anchors)]
        if len(candidate_structs) < 2:
            raise ValueError("材料锚点结构过多，剩余可验证结构不足。请减少 --material-anchor-structures 或使用 --split-mode groupkfold。")

        n_splits = min(int(cfg.folds), len(candidate_structs))
        candidate_mask = np.isin(groups, candidate_structs)
        cand_idx = np.where(candidate_mask)[0]
        gkf = GroupKFold(n_splits=n_splits)
        splits = []
        for fold, (_, va_cand_rel) in enumerate(gkf.split(X[cand_idx], y[cand_idx], groups[cand_idx]), start=1):
            va_idx = cand_idx[va_cand_rel]
            val_structs_initial = _sort_str_names(np.unique(groups[va_idx]).tolist())
            train_structs_initial = _sort_str_names([s for s in unique_structs if s not in set(val_structs_initial)])
            train_structs, val_structs, moved = _repair_validation_structures(train_structs_initial, val_structs_initial, material_dict)
            if not val_structs:
                raise ValueError(f"材料覆盖修正后 Fold {fold} 验证结构为空，请减少折数或改用 groupkfold。")

            train_idx = np.where(np.isin(groups, train_structs))[0]
            val_idx = np.where(np.isin(groups, val_structs))[0]
            missing = _coverage_missing(train_structs, val_structs, material_dict)
            if missing:
                raise ValueError(f"Fold {fold} 材料覆盖约束仍未满足，缺失材料: {missing}")
            report_rows.append({
                "Fold": fold,
                "Split_Mode": "material_aware",
                "Train_Structures": ",".join(train_structs),
                "Validation_Structures": ",".join(val_structs),
                "Anchor_Structures": ",".join(anchors),
                "Moved_Back_To_Train": ",".join(moved),
                "Missing_Materials_In_Train": "",
                "Material_Coverage_OK": True,
                "N_Train": int(len(train_idx)),
                "N_Validation": int(len(val_idx)),
            })
            splits.append((train_idx, val_idx))
    else:
        raise ValueError("未知 split_mode。可选: material_aware 或 groupkfold。")

    if out_dir is not None and bool(getattr(cfg, "save_split_report", True)):
        ensure_dir(out_dir)
        pd.DataFrame(report_rows).to_csv(out_dir / f"{report_prefix}_split_material_coverage_report.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(occurrence_rows).to_csv(out_dir / f"{report_prefix}_material_occurrence_report.csv", index=False, encoding="utf-8-sig")

    log(f"[划分] split_mode={mode}, folds={len(splits)}")
    if mode in {"material_aware", "material-aware", "coverage"}:
        log("[划分] 已启用材料覆盖约束：验证集出现的材料均已在训练集中出现。")
    return splits


# =============================================================================
# 1. 结构物理语义映射
# =============================================================================

def build_structure_map() -> pd.DataFrame:
    """构建 STR1-STR19 的 18 维材料厚度矩阵。厚度单位按原始结构图理解为 cm。"""
    str_names = [f"STR{i}" for i in range(1, 20)]
    df_struct = pd.DataFrame(0.0, index=str_names, columns=SEMANTIC_COLS)

    # 第 I 类：半刚性基层结构
    df_struct.loc["STR1", ["Surf_AC13_I_SBS", "Mid_AC20_AH30", "Base_CBG_A", "Base_CS"]] = [4, 8, 40, 40]
    df_struct.loc["STR2", ["Surf_AC13_I_SBS", "Mid_AC20_AH30", "Base_CBG_A", "Base_CS"]] = [4, 8, 40, 20]
    df_struct.loc["STR3", ["Surf_AC13_I_SBS", "Mid_AC20_AH30", "Base_CBG_A", "Base_GA"]] = [4, 8, 40, 20]

    # 第 II 类：刚性复合式基层结构
    df_struct.loc["STR4", ["Surf_AC13_II_SBS", "Mid_AC20_AH30", "Bot_AC10_SBS", "Base_LCC", "Base_CBG_A", "Base_CS"]] = [4, 6, 2, 24, 20, 20]
    df_struct.loc["STR5", ["Surf_AC13_II_SBS", "Mid_AC20_SBS", "Bot_AC10_SBS", "Base_CC", "Base_CBG_A", "Base_CS"]] = [4, 6, 2, 24, 20, 20]

    # 第 III 类：半刚性基层结构
    df_struct.loc["STR6", ["Surf_AC13_II_SBS", "Bot_AC25_AH30", "Bot_AC10_SBS", "Base_CBG_A", "Base_CS"]] = [4, 10, 2, 38, 20]
    df_struct.loc["STR7", ["Surf_AC13_II_SBS", "Mid_AC20_SBS", "Bot_AC25_AH70", "Base_CBG_A", "Base_CS"]] = [4, 6, 8, 38, 20]
    df_struct.loc["STR8", ["Surf_AC13_II_SBS", "Mid_AC20_SBS", "Bot_AC25_AH70", "Base_CBG_B", "Base_CS"]] = [4, 6, 8, 38, 20]
    df_struct.loc["STR9", ["Surf_PAC13", "Mid_AC20_SBS", "Bot_AC25_AH70", "Base_CBG_B", "Base_CS"]] = [4, 6, 8, 38, 20]

    # 第 IV 类：倒装式基层结构
    df_struct.loc["STR10", ["Surf_AC13_I_SBS", "Mid_AC20_SBS", "Bot_AC25_AH70", "Bot_AC10_SBS", "Base_GA", "Base_CBG_B", "Base_CS"]] = [4, 6, 16, 2, 20, 20, 20]
    df_struct.loc["STR12", ["Surf_AC13_I_SBS", "Mid_AC20_SBS", "Bot_AC25_AH70", "Base_GA", "Base_CBG_B", "Base_CS"]] = [4, 8, 12, 20, 20, 20]

    # 第 V 类：厚沥青混凝土结构 I
    df_struct.loc["STR11", ["Surf_AC13_I_SBS", "Mid_AC20_SBS", "Bot_AC25_AH70", "Base_CBG_A", "Base_CS"]] = [4, 6, 18, 40, 20]
    df_struct.loc["STR13", ["Surf_AC13_I_SBS", "Mid_AC20_SBS", "Bot_AC25_AH70", "Base_CBG_A", "Base_CS"]] = [4, 8, 12, 40, 20]
    df_struct.loc["STR14", ["Surf_AC13_I_SBS", "Mid_AC20_SBS", "Bot_AC25_Re_AH70", "Base_CBG_B", "Base_CS"]] = [4, 8, 12, 40, 20]

    # 第 VI 类：厚沥青混凝土结构 II
    df_struct.loc["STR15", ["Surf_AC13_I_SBS", "Mid_AC20_AH50", "Bot_AC25_AH50", "Base_CBG_A", "Base_GA"]] = [4, 8, 24, 20, 44]
    df_struct.loc["STR16", ["Surf_SMA13", "Mid_AC20_SBS", "Bot_AC25_AH70", "Base_CBG_A", "Base_CS"]] = [4, 8, 24, 20, 20]
    df_struct.loc["STR17", ["Surf_SMA13", "Mid_AC20_AH30", "Bot_AC25_AH30", "Base_CBG_A", "Base_CS"]] = [4, 8, 24, 20, 20]

    # 第 VII 类：全厚式结构
    df_struct.loc["STR18", ["Surf_SMA13", "Mid_AC20_AH50", "Bot_AC25_AH50", "Bot_AC10_SBS", "Base_GA"]] = [4, 8, 36, 4, 48]
    df_struct.loc["STR19", ["Surf_SMA13", "Mid_AC20_AH50", "Bot_AC25_AH30", "Base_CBG_B"]] = [4, 8, 36, 20]

    class_map = {
        "STR1": "I_SemiRigid_12cmAC", "STR2": "I_SemiRigid_12cmAC", "STR3": "I_SemiRigid_12cmAC",
        "STR4": "II_RigidComposite_12cmAC", "STR5": "II_RigidComposite_12cmAC",
        "STR6": "III_SemiRigid_16_18cmAC", "STR7": "III_SemiRigid_16_18cmAC", "STR8": "III_SemiRigid_16_18cmAC", "STR9": "III_SemiRigid_16_18cmAC",
        "STR10": "IV_Inverted_24_28cmAC", "STR12": "IV_Inverted_24_28cmAC",
        "STR11": "V_ThickAC_I_24_28cm", "STR13": "V_ThickAC_I_24_28cm", "STR14": "V_ThickAC_I_24_28cm",
        "STR15": "VI_ThickAC_II_36cm", "STR16": "VI_ThickAC_II_36cm", "STR17": "VI_ThickAC_II_36cm",
        "STR18": "VII_FullDepth_48_52cm", "STR19": "VII_FullDepth_48_52cm",
    }

    df_struct["STR_name"] = df_struct.index
    df_struct["Struct_Class"] = df_struct["STR_name"].map(class_map)
    df_struct["Struct_ID"] = np.arange(len(df_struct), dtype=int)
    return df_struct.reset_index(drop=True)


def recover_str_name_from_thickness(df: pd.DataFrame, struct_map: pd.DataFrame) -> pd.Series:
    """当处理后表缺少 STR_name 时，根据 18 维厚度向量反推 STR 名称。"""
    lookup = {}
    for _, row in struct_map.iterrows():
        key = tuple(float(row[c]) for c in SEMANTIC_COLS)
        lookup[key] = row["STR_name"]

    names = []
    for _, row in df.iterrows():
        key = tuple(float(row[c]) for c in SEMANTIC_COLS)
        names.append(lookup.get(key, "UNKNOWN"))
    return pd.Series(names, index=df.index)


# =============================================================================
# 2. 数据读取、长表重构、工程特征
# =============================================================================

def read_excel_auto(path: str | Path) -> pd.DataFrame:
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到输入文件: {path}")

    # 优先按原始车辙宽表格式读取。该格式通常第 2 行才是 STR1-STR19 表头。
    try:
        df_h1 = pd.read_excel(path, header=1)
        if {f"STR{i}" for i in range(1, 20)}.issubset(set(map(str, df_h1.columns))):
            return df_h1
    except Exception:
        pass

    # 若已经是处理后长表，则 header=0。
    df_h0 = pd.read_excel(path, header=0)
    return df_h0


def is_raw_wide_table(df: pd.DataFrame) -> bool:
    cols = set(map(str, df.columns))
    return {f"STR{i}" for i in range(1, 20)}.issubset(cols)


def is_processed_long_table(df: pd.DataFrame) -> bool:
    cols = set(map(str, df.columns))
    has_target = ("Rutting_Depth_0_1mm" in cols) or ("Rutting_Depth_mm" in cols)
    has_semantics = set(SEMANTIC_COLS).issubset(cols)
    return has_target and has_semantics


def build_long_panel_from_raw(raw_df: pd.DataFrame, struct_map: pd.DataFrame, rutting_scale_to_mm: float) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {
        "Unnamed: 0": "Year",
        "Unnamed: 1": "Loading_Date",
        "Unnamed: 2": "Cycle_ID",
        "Unnamed: 3": "Cum_Load_10k",
        "Unnamed: 4": "Avg_Temp_C",
        "年": "Year",
        "加载日期": "Loading_Date",
        "加载周期编号": "Cycle_ID",
        "当量设计轴载累计作用次数（万次）": "Cum_Load_10k",
        "平均气温(℃)": "Avg_Temp_C",
        "平均气温（℃）": "Avg_Temp_C",
    }
    df = df.rename(columns=rename_map)

    required_meta = ["Cum_Load_10k", "Avg_Temp_C"]
    missing_meta = [c for c in required_meta if c not in df.columns]
    if missing_meta:
        raise ValueError(f"原始宽表缺少必要字段: {missing_meta}。请检查 Excel 表头。")

    for col in ["Year", "Loading_Date", "Cycle_ID"]:
        if col not in df.columns:
            df[col] = np.nan

    # 年份在原始表中经常只在每年第一行出现，需要向下填充。
    df["Year"] = df["Year"].ffill()

    str_cols = [f"STR{i}" for i in range(1, 20)]
    long_df = df[["Year", "Loading_Date", "Cycle_ID", "Cum_Load_10k", "Avg_Temp_C"] + str_cols].melt(
        id_vars=["Year", "Loading_Date", "Cycle_ID", "Cum_Load_10k", "Avg_Temp_C"],
        value_vars=str_cols,
        var_name="STR_name",
        value_name="Rutting_Depth_raw",
    )

    long_df["STR_num"] = long_df["STR_name"].str.replace("STR", "", regex=False).astype(int)
    long_df["Rutting_Depth_raw"] = pd.to_numeric(long_df["Rutting_Depth_raw"], errors="coerce")
    long_df["Cum_Load_10k"] = pd.to_numeric(long_df["Cum_Load_10k"], errors="coerce")
    long_df["Avg_Temp_C"] = pd.to_numeric(long_df["Avg_Temp_C"], errors="coerce")

    long_df = long_df.dropna(subset=["Rutting_Depth_raw", "Cum_Load_10k", "Avg_Temp_C"]).copy()
    long_df = long_df.merge(struct_map, on="STR_name", how="left")

    if long_df["Struct_Class"].isna().any():
        bad = long_df.loc[long_df["Struct_Class"].isna(), "STR_name"].unique().tolist()
        raise ValueError(f"结构映射失败，未知结构: {bad}")

    long_df["Rutting_Depth_mm"] = long_df["Rutting_Depth_raw"] * float(rutting_scale_to_mm)
    return long_df


def build_long_panel_from_processed(proc_df: pd.DataFrame, struct_map: pd.DataFrame, rutting_scale_to_mm: float) -> pd.DataFrame:
    df = proc_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "Rutting_Depth_mm" not in df.columns:
        if "Rutting_Depth_0_1mm" not in df.columns:
            raise ValueError("处理后表缺少 Rutting_Depth_mm 或 Rutting_Depth_0_1mm。")
        df["Rutting_Depth_raw"] = pd.to_numeric(df["Rutting_Depth_0_1mm"], errors="coerce")
        df["Rutting_Depth_mm"] = df["Rutting_Depth_raw"] * float(rutting_scale_to_mm)
    else:
        df["Rutting_Depth_mm"] = pd.to_numeric(df["Rutting_Depth_mm"], errors="coerce")
        df["Rutting_Depth_raw"] = df["Rutting_Depth_mm"] / max(float(rutting_scale_to_mm), 1e-12)

    for c in ["Cum_Load_10k", "Avg_Temp_C"] + SEMANTIC_COLS:
        if c not in df.columns:
            raise ValueError(f"处理后表缺少必要列: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "STR_name" not in df.columns:
        df["STR_name"] = recover_str_name_from_thickness(df, struct_map)

    df = df.merge(struct_map[["STR_name", "Struct_ID", "Struct_Class"]], on="STR_name", how="left", suffixes=("", "_map"))

    if "Struct_ID_map" in df.columns:
        df["Struct_ID"] = df["Struct_ID"].fillna(df["Struct_ID_map"]) if "Struct_ID" in df.columns else df["Struct_ID_map"]
        df.drop(columns=["Struct_ID_map"], inplace=True)
    if "Struct_Class_map" in df.columns:
        df["Struct_Class"] = df["Struct_Class"].fillna(df["Struct_Class_map"]) if "Struct_Class" in df.columns else df["Struct_Class_map"]
        df.drop(columns=["Struct_Class_map"], inplace=True)

    # 缺失的元信息补齐。
    for col in ["Year", "Loading_Date", "Cycle_ID"]:
        if col not in df.columns:
            df[col] = np.nan

    if "STR_num" not in df.columns:
        df["STR_num"] = df["STR_name"].str.replace("STR", "", regex=False).astype(int)

    df = df.dropna(subset=["Rutting_Depth_mm", "Cum_Load_10k", "Avg_Temp_C", "STR_name"]).copy()
    return df


def add_engineering_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    df = panel_df.copy()

    df["STR_num"] = df["STR_name"].str.replace("STR", "", regex=False).astype(int)
    df = df.sort_values(["STR_num", "Cum_Load_10k", "Cycle_ID"], kind="mergesort").reset_index(drop=True)

    # 每个结构内的时序编号。
    df["Observation_Order"] = df.groupby("STR_name").cumcount() + 1

    # 荷载增量：第一个观测点相当于从 0 加载到当前累计荷载。
    df["Delta_Load_10k"] = df.groupby("STR_name")["Cum_Load_10k"].diff()
    first_delta = df.groupby("STR_name")["Cum_Load_10k"].transform("first")
    df["Delta_Load_10k"] = df["Delta_Load_10k"].fillna(first_delta)
    df["Delta_Load_10k"] = df["Delta_Load_10k"].clip(lower=0)

    df["Delta_Temp_C"] = df.groupby("STR_name")["Avg_Temp_C"].diff().fillna(0.0)
    df["Temp_MA3_C"] = df.groupby("STR_name")["Avg_Temp_C"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["Temp_MA5_C"] = df.groupby("STR_name")["Avg_Temp_C"].transform(lambda s: s.rolling(5, min_periods=1).mean())

    df["Prev_Rutting_mm"] = df.groupby("STR_name")["Rutting_Depth_mm"].shift(1).fillna(0.0)

    prev_peak = []
    for _, g in df.groupby("STR_name", sort=False):
        vals = g["Rutting_Depth_mm"].values
        peaks = np.maximum.accumulate(vals)
        peaks_shift = np.concatenate([[0.0], peaks[:-1]])
        prev_peak.extend(peaks_shift.tolist())
    df["Prev_Peak_Rutting_mm"] = prev_peak

    df["Rutting_Increment_mm"] = df["Rutting_Depth_mm"] - df["Prev_Rutting_mm"]

    # 服役阶段：按累计荷载分位粗分，用于后续误差诊断。
    try:
        df["Load_Stage"] = pd.qcut(df["Cum_Load_10k"], q=4, labels=["Early", "Middle", "Late", "UltraLate"])
    except Exception:
        df["Load_Stage"] = "Unknown"

    final_cols = [
        "Year", "Loading_Date", "Cycle_ID", "Observation_Order",
        "STR_name", "STR_num", "Struct_ID", "Struct_Class",
        "Cum_Load_10k", "Delta_Load_10k", "Avg_Temp_C", "Delta_Temp_C", "Temp_MA3_C", "Temp_MA5_C",
    ] + SEMANTIC_COLS + [
        "Rutting_Depth_raw", "Rutting_Depth_mm", "Prev_Rutting_mm", "Prev_Peak_Rutting_mm",
        "Rutting_Increment_mm", "Load_Stage",
    ]
    existing_cols = [c for c in final_cols if c in df.columns]
    return df[existing_cols].copy()


def load_and_prepare_dataset(input_path: str | Path, rutting_scale_to_mm: float) -> pd.DataFrame:
    struct_map = build_structure_map()
    raw = read_excel_auto(input_path)

    if is_raw_wide_table(raw):
        log("[数据] 识别为原始宽表，开始宽表转长表。")
        panel = build_long_panel_from_raw(raw, struct_map, rutting_scale_to_mm)
    elif is_processed_long_table(raw):
        log("[数据] 识别为已处理长表，开始补齐结构与时序特征。")
        panel = build_long_panel_from_processed(raw, struct_map, rutting_scale_to_mm)
    else:
        raise ValueError(
            "无法识别输入表格式。请提供原始 STR1-STR19 宽表，或包含 18 维结构厚度和车辙目标列的处理后长表。"
        )

    panel = add_engineering_features(panel)
    return panel


# =============================================================================
# 3. 数据审查
# =============================================================================

def audit_panel_dataset(df: pd.DataFrame, cfg: Config, out_dir: Path) -> Dict[str, object]:
    report: Dict[str, object] = {}
    report["n_rows"] = int(len(df))
    report["n_cols"] = int(df.shape[1])
    report["n_structures"] = int(df["STR_name"].nunique())
    report["structures"] = sorted(df["STR_name"].unique().tolist(), key=lambda x: int(x.replace("STR", "")))
    report["missing_total"] = int(df.isna().sum().sum())
    report["rutting_scale_to_mm"] = cfg.rutting_scale_to_mm
    report["target_min_mm"] = float(df["Rutting_Depth_mm"].min())
    report["target_max_mm"] = float(df["Rutting_Depth_mm"].max())
    report["load_min_10k"] = float(df["Cum_Load_10k"].min())
    report["load_max_10k"] = float(df["Cum_Load_10k"].max())

    non_monotonic = []
    for sid, g in df.groupby("STR_name"):
        if (g["Cum_Load_10k"].diff().dropna() < -1e-9).any():
            non_monotonic.append(sid)
    report["non_monotonic_load_structures"] = non_monotonic

    unknown_struct = sorted(df.loc[df["STR_name"].eq("UNKNOWN"), "STR_name"].unique().tolist())
    report["unknown_structures"] = unknown_struct

    duplicate_count = int(df.duplicated(subset=["STR_name", "Cum_Load_10k", "Observation_Order"]).sum())
    report["duplicate_key_rows"] = duplicate_count

    log("\n========== 数据审查报告 ==========")
    log(f"样本行数: {report['n_rows']}")
    log(f"结构数量: {report['n_structures']} -> {', '.join(report['structures'])}")
    log(f"缺失值总数: {report['missing_total']}")
    log(f"车辙尺度: 原始值 × {cfg.rutting_scale_to_mm} = mm")
    log(f"车辙范围: {report['target_min_mm']:.4f} ~ {report['target_max_mm']:.4f} mm")
    log(f"累计荷载范围: {report['load_min_10k']:.2f} ~ {report['load_max_10k']:.2f} 万次")
    log(f"荷载非单调结构: {non_monotonic if non_monotonic else '无'}")
    log(f"重复键行数: {duplicate_count}")
    log("==================================\n")

    with open(out_dir / "audit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


# =============================================================================
# 4. 时序样本构建
# =============================================================================

def build_sequence_arrays(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    返回：
    X_seq_raw: [N, L, F]
    y: [N]
    prev_y: [N, 1]
    prev_peak_y: [N, 1]
    groups: [N]
    meta: 每个样本的元信息
    """
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    prev_y_list: List[float] = []
    prev_peak_list: List[float] = []
    groups: List[str] = []
    meta_rows: List[pd.Series] = []

    sort_cols = ["STR_num", "Cum_Load_10k", "Observation_Order"]
    df_sorted = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    for sid, g in df_sorted.groupby("STR_name", sort=False):
        g = g.sort_values(["Cum_Load_10k", "Observation_Order"], kind="mergesort").reset_index(drop=True)
        X = g[feature_cols].astype(float).values
        y = g["Rutting_Depth_mm"].astype(float).values
        prev_y = g["Prev_Rutting_mm"].astype(float).values
        prev_peak = g["Prev_Peak_Rutting_mm"].astype(float).values

        pad_X = np.vstack([np.tile(X[0], (seq_len - 1, 1)), X])
        for i in range(len(g)):
            window = pad_X[i: i + seq_len]
            X_list.append(window)
            y_list.append(float(y[i]))
            prev_y_list.append(float(prev_y[i]))
            prev_peak_list.append(float(prev_peak[i]))
            groups.append(sid)
            meta_rows.append(g.iloc[i][[
                "Year", "Loading_Date", "Cycle_ID", "Observation_Order", "STR_name", "STR_num", "Struct_Class",
                "Cum_Load_10k", "Delta_Load_10k", "Avg_Temp_C", "Rutting_Depth_mm", "Prev_Rutting_mm", "Prev_Peak_Rutting_mm",
            ]])

    X_seq = np.asarray(X_list, dtype=np.float32)
    y_arr = np.asarray(y_list, dtype=np.float32)
    prev_y_arr = np.asarray(prev_y_list, dtype=np.float32).reshape(-1, 1)
    prev_peak_arr = np.asarray(prev_peak_list, dtype=np.float32).reshape(-1, 1)
    groups_arr = np.asarray(groups)
    meta_df = pd.DataFrame(meta_rows).reset_index(drop=True)
    return X_seq, y_arr, prev_y_arr, prev_peak_arr, groups_arr, meta_df


def exclude_initial_observation_targets(
    X_seq_raw: np.ndarray,
    y: np.ndarray,
    prev_y: np.ndarray,
    prev_peak: np.ndarray,
    groups: np.ndarray,
    meta_df: pd.DataFrame,
    enabled: bool = True,
    out_dir: Optional[Path] = None,
    report_prefix: str = "main",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """将每个结构的首个观测点作为状态初始化样本，不作为预测目标。

    注意：过滤发生在滑动窗口构建之后，因此第二个观测点的历史窗口中
    仍可包含第一个观测点的信息。这符合工程监测递推任务：已有一次
    实测车辙后，预测下一阶段车辙演化。
    """
    if not enabled:
        return X_seq_raw, y, prev_y, prev_peak, groups, meta_df
    if "Observation_Order" not in meta_df.columns:
        log("[冷启动过滤] 未发现 Observation_Order 列，跳过首观测点过滤。")
        return X_seq_raw, y, prev_y, prev_peak, groups, meta_df

    obs_order = pd.to_numeric(meta_df["Observation_Order"], errors="coerce").fillna(999999).astype(int).values
    keep_mask = obs_order > 1
    removed = int((~keep_mask).sum())
    kept = int(keep_mask.sum())
    if removed == 0:
        log(f"[冷启动过滤] 未发现首观测点；保留样本 {kept}。")
        return X_seq_raw, y, prev_y, prev_peak, groups, meta_df

    removed_rows = meta_df.loc[~keep_mask, [c for c in ["STR_name", "Observation_Order", "Cum_Load_10k", "Rutting_Depth_mm", "Prev_Rutting_mm", "Prev_Peak_Rutting_mm"] if c in meta_df.columns]].copy()
    log(f"[冷启动过滤] 已将每个结构首个观测点作为状态初始化，不参与训练/评价：移除 {removed}，保留 {kept}。")
    if out_dir is not None:
        ensure_dir(out_dir)
        removed_rows.to_csv(out_dir / f"{report_prefix}_excluded_initial_observations.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{
            "Report": report_prefix,
            "Exclude_Initial_Observation_Targets": True,
            "Removed_Samples": removed,
            "Kept_Samples": kept,
            "Reason": "First observation of each structure has no previous measured rutting state; used only for state initialization.",
        }]).to_csv(out_dir / f"{report_prefix}_cold_start_filter_report.csv", index=False, encoding="utf-8-sig")

    return (
        X_seq_raw[keep_mask],
        y[keep_mask],
        prev_y[keep_mask],
        prev_peak[keep_mask],
        groups[keep_mask],
        meta_df.loc[keep_mask].reset_index(drop=True),
    )


# =============================================================================
# 5. 模型：TDR-PRSN
# =============================================================================

class ResidualTCNBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        residual = x
        y = x.transpose(1, 2)
        y = self.conv(y)
        y = y[:, :, :x.size(1)]
        y = y.transpose(1, 2)
        x = self.norm1(residual + self.dropout(y))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class MaterialTokenEncoder(nn.Module):
    def __init__(self, num_materials: int = 18, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        self.num_materials = num_materials
        self.material_embedding = nn.Embedding(num_materials, d_model)
        self.layer_position_embedding = nn.Embedding(num_materials, d_model)
        self.thickness_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # 数据规模较小，结构 token 编码采用轻量交互层，避免 CPU 上 Transformer 过慢。
        self.encoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

    def forward(self, thickness_vec_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # thickness_vec_raw: [B, 18]，单位 cm。使用 log1p 降低厚度数值跨度。
        B, M = thickness_vec_raw.shape
        device = thickness_vec_raw.device
        material_ids = torch.arange(M, device=device).unsqueeze(0).expand(B, M)
        thickness_token = torch.log1p(torch.relu(thickness_vec_raw)).unsqueeze(-1)
        token = (
            self.material_embedding(material_ids)
            + self.layer_position_embedding(material_ids)
            + self.thickness_proj(thickness_token)
        )
        token = self.encoder(token)
        attn = torch.softmax(self.attn_pool(token), dim=1)
        struct_z = torch.sum(attn * token, dim=1)
        return struct_z, token, attn.squeeze(-1)



# =============================================================================
# 5.5 深度学习基线模型：LSTM / Transformer
# =============================================================================

class VanillaLSTM(nn.Module):
    """纯数据驱动 LSTM 序列基线。

    该模型使用与 TDR-PRSN 相同的时序输入，并将上一期车辙与历史峰值
    作为附加状态通道拼接到每个时间步。它不包含结构材料 token 编码、
    温载门控、双状态恢复、观测适配器或物理引导损失。
    """

    def __init__(self, num_features: int, d_model: int = 64, dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        self.num_features = int(num_features)
        self.lstm = nn.LSTM(
            input_size=self.num_features,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    @staticmethod
    def _augment_with_state(x_seq_scaled: torch.Tensor, prev_y_raw: torch.Tensor, prev_peak_y_raw: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
        # 使用与主模型训练目标一致的 log1p 标准化状态，避免量纲不公平。
        prev_y_scaled = (torch.log1p(torch.relu(prev_y_raw)) - y_mean) / y_std
        prev_peak_scaled = (torch.log1p(torch.relu(prev_peak_y_raw)) - y_mean) / y_std
        state = torch.cat([prev_y_scaled, prev_peak_scaled], dim=-1).unsqueeze(1).expand(-1, x_seq_scaled.size(1), -1)
        return torch.cat([x_seq_scaled, state], dim=-1)

    def forward(
        self,
        x_seq_scaled: torch.Tensor,
        x_seq_raw: Optional[torch.Tensor] = None,
        prev_y_raw: Optional[torch.Tensor] = None,
        prev_peak_y_raw: Optional[torch.Tensor] = None,
        y_mean: Optional[torch.Tensor] = None,
        y_std: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if prev_y_raw is not None and prev_peak_y_raw is not None and y_mean is not None and y_std is not None:
            x = self._augment_with_state(x_seq_scaled, prev_y_raw, prev_peak_y_raw, y_mean, y_std)
        else:
            x = x_seq_scaled
        _, (h_n, _) = self.lstm(x)
        mu_scaled = self.head(h_n[-1])
        # 仅用于保持接口一致；论文中不对该确定性基线报告概率指标。
        return {"mu_scaled": mu_scaled, "var_scaled": torch.full_like(mu_scaled, 1e-4)}


class VanillaTransformer(nn.Module):
    """纯数据驱动 Transformer Encoder 序列基线。

    该模型加入可学习位置编码，使 Transformer 能区分不同加载历史位置。
    概率方差仅为接口占位，不参与概率指标比较。
    """

    def __init__(self, num_features: int, seq_len: int, d_model: int = 64, dropout: float = 0.1, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.num_features = int(num_features)
        self.seq_len = int(seq_len)
        self.input_proj = nn.Linear(self.num_features, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _augment_with_state(x_seq_scaled: torch.Tensor, prev_y_raw: torch.Tensor, prev_peak_y_raw: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
        prev_y_scaled = (torch.log1p(torch.relu(prev_y_raw)) - y_mean) / y_std
        prev_peak_scaled = (torch.log1p(torch.relu(prev_peak_y_raw)) - y_mean) / y_std
        state = torch.cat([prev_y_scaled, prev_peak_scaled], dim=-1).unsqueeze(1).expand(-1, x_seq_scaled.size(1), -1)
        return torch.cat([x_seq_scaled, state], dim=-1)

    def forward(
        self,
        x_seq_scaled: torch.Tensor,
        x_seq_raw: Optional[torch.Tensor] = None,
        prev_y_raw: Optional[torch.Tensor] = None,
        prev_peak_y_raw: Optional[torch.Tensor] = None,
        y_mean: Optional[torch.Tensor] = None,
        y_std: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if prev_y_raw is not None and prev_peak_y_raw is not None and y_mean is not None and y_std is not None:
            x = self._augment_with_state(x_seq_scaled, prev_y_raw, prev_peak_y_raw, y_mean, y_std)
        else:
            x = x_seq_scaled
        if x.size(1) != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, but got {x.size(1)}.")
        h = self.input_proj(x) + self.pos_embed
        h = self.transformer(h)
        mu_scaled = self.head(h[:, -1, :])
        return {"mu_scaled": mu_scaled, "var_scaled": torch.full_like(mu_scaled, 1e-4)}


def build_neural_model(
    model_type: str,
    num_features: int,
    cfg: Config,
    variant: str,
    device: torch.device,
) -> Tuple[nn.Module, str, bool, bool]:
    """统一构建 TDR-PRSN 与深度学习基线。

    返回：model, model_label, use_physics_losses, use_probabilistic_metrics。
    LSTM/Transformer 使用相同输入信息，但关闭所有物理损失和概率指标。
    """
    model_type_norm = str(model_type).strip().upper()
    if model_type_norm == "LSTM":
        model = VanillaLSTM(
            num_features=num_features + 2,
            d_model=cfg.d_model,
            dropout=cfg.dropout,
            num_layers=2,
        ).to(device)
        return model, "Vanilla_LSTM", False, False
    if model_type_norm in {"TRANSFORMER", "VANILLA_TRANSFORMER"}:
        model = VanillaTransformer(
            num_features=num_features + 2,
            seq_len=cfg.seq_len,
            d_model=cfg.d_model,
            dropout=cfg.dropout,
            nhead=4,
            num_layers=2,
        ).to(device)
        return model, "Vanilla_Transformer", False, False
    if model_type_norm in {"TDRPRSN", "TDR-PRSN"}:
        flags = get_ablation_flags(variant)
        model = TDRPRSN(
            num_features=num_features,
            seq_len=cfg.seq_len,
            d_model=cfg.d_model,
            dropout=cfg.dropout,
            variant=variant,
            min_physics_fusion_gate=float(getattr(cfg, "min_physics_fusion_gate", 0.15)),
        ).to(device)
        return model, str(flags.get("label", f"TDR-PRSN-{variant}")), True, bool(flags.get("use_probabilistic", True))
    raise ValueError(f"未知 model_type={model_type!r}。可选：TDRPRSN, LSTM, Transformer。")


ABLATION_REGISTRY: Dict[str, Dict[str, object]] = {
    "full": {
        "label": "A9 Full TDR-PRSN",
        "claim": "完整的结构语义、时序记忆、温载门控、双状态、观测锚定、概率风险输出。",
        "use_structure": True,
        "use_material_token": True,
        "use_tcn": True,
        "use_temporal_attention": True,
        "use_thermal_gate": True,
        "use_reversible_state": True,
        "use_prev_y_state": True,
        "use_prev_peak_state": True,
        "use_observation_adapter": True,
        "use_probabilistic": True,
        "use_risk": True,
        "use_state_calibrator": True,
    },
    "A1_env_only": {
        "label": "A1 w/o structure semantics",
        "claim": "只保留荷载-温度时序，不输入结构层位厚度，检验结构语义映射的价值。",
        "use_structure": False,
        "use_material_token": False,
    },
    "A2_flat_thickness": {
        "label": "A2 flat thickness MLP",
        "claim": "仍输入 18 维厚度，但取消材料 token 与层位注意力，检验层次化结构编码的价值。",
        "use_material_token": False,
    },
    "A3_no_tcn": {
        "label": "A3 w/o dilated TCN",
        "claim": "取消多尺度扩张卷积，只依赖 GRU 时序，检验局部-多尺度服役记忆的价值。",
        "use_tcn": False,
    },
    "A4_no_thermal_gate": {
        "label": "A4 w/o thermo-load gate",
        "claim": "温度门控恒为 1，检验温度对损伤速率激活的调制作用。",
        "use_thermal_gate": False,
    },
    "A5_no_reversible": {
        "label": "A5 w/o reversible state",
        "claim": "取消可逆温度响应，只保留永久损伤状态，检验双状态分解的必要性。",
        "use_reversible_state": False,
    },
    "A6_no_peak_state": {
        "label": "A6 w/o irreversible peak state",
        "claim": "不用历史峰值作为永久损伤锚点，检验不可逆累积状态约束的价值。",
        "use_prev_peak_state": False,
    },
    "A7_no_obs_adapter": {
        "label": "A7 w/o observation adapter",
        "claim": "取消上一期实测值残差锚定，检验工程监测短期惯性的贡献。",
        "use_observation_adapter": False,
    },
    "A8_deterministic": {
        "label": "A8 deterministic mean only",
        "claim": "取消 NLL 概率训练与风险分支，只优化均值，检验不确定性和风险任务的辅助收益。",
        "use_probabilistic": False,
        "use_risk": False,
    },
    "A9_no_state_calibrator": {
        "label": "A9 w/o state residual calibrator",
        "claim": "取消神经状态残差校准层，检验物理状态到观测均值的细粒度修正能力。",
        "use_state_calibrator": False,
    },
}


def get_ablation_flags(variant: str = "full") -> Dict[str, object]:
    """返回消融变体的开关。未显式声明的开关继承 full。"""
    base = dict(ABLATION_REGISTRY["full"])
    if variant not in ABLATION_REGISTRY:
        raise ValueError(f"未知消融变体: {variant}。可选: {list(ABLATION_REGISTRY.keys())}")
    base.update(ABLATION_REGISTRY[variant])
    return base


class TDRPRSN(nn.Module):
    """
    TDR-PRSN: Thermo-load Dual-state Recovery Probabilistic Residual State-space Network

    模型内涵：
    1) 结构语义层：把 18 维材料/层位厚度编码为结构状态 token，而不是把厚度当普通静态数字；
    2) 温载损伤层：荷载增量提供损伤驱动力，温度门控调制损伤速率；
    3) 双状态层：历史峰值近似不可逆永久损伤，温度响应分支允许短期可逆波动；
    4) 观测方程层：上一期实测车辙作为监测状态锚点，增强跨结构小样本预测稳定性；
    5) 概率风险层：同时输出均值、异方差不确定性和养护阈值超限概率。
    """

    def __init__(
        self,
        num_features: int,
        seq_len: int = 6,
        num_materials: int = 18,
        d_model: int = 64,
        dropout: float = 0.1,
        variant: str = "full",
        min_physics_fusion_gate: float = 0.15,
    ):
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.num_materials = num_materials
        self.variant = variant
        self.flags = get_ablation_flags(variant)
        # 观测残差锚定很强，但如果融合门控完全偏向观测残差通道，
        # 双状态分解的永久/可逆物理状态可能只成为旁路解释变量。
        # 设置一个较小的物理状态最低融合比例，使双状态路径始终参与最终均值，
        # 但不破坏上一期实测状态的动态锚定作用。
        self.min_physics_fusion_gate = float(min(max(min_physics_fusion_gate, 0.0), 0.5))

        self.struct_encoder = MaterialTokenEncoder(num_materials=num_materials, d_model=d_model, dropout=dropout)
        self.thickness_mlp = nn.Sequential(
            nn.Linear(num_materials, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.env_proj = nn.Sequential(
            nn.Linear(5, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        self.input_fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        if bool(self.flags["use_tcn"]):
            self.tcn = nn.Sequential(
                ResidualTCNBlock(d_model, dilation=1, dropout=dropout),
                ResidualTCNBlock(d_model, dilation=2, dropout=dropout),
                ResidualTCNBlock(d_model, dilation=4, dropout=dropout),
            )
        else:
            self.tcn = nn.Identity()

        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)

        self.temporal_attn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

        self.state_fusion = nn.Sequential(
            nn.Linear(2 * d_model + 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        self.damage_rate_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        self.thermal_gate_head = nn.Sequential(
            nn.Linear(d_model + 3, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self.recovery_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        self.obs_delta_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.obs_gate_head = nn.Sequential(
            nn.Linear(d_model + 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self.var_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self.risk_head = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # 神经状态残差校准层：
        # 在不破坏永久损伤/可逆响应等物理状态输出的前提下，
        # 对物理均值进行小幅 log-domain 修正，用来吸收测量噪声、结构细节与未建模局部效应。
        self.state_calibrator_head = nn.Sequential(
            nn.Linear(d_model + 5, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.state_calibrator_gate = nn.Sequential(
            nn.Linear(d_model + 5, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        x_seq_scaled: torch.Tensor,
        x_seq_raw: torch.Tensor,
        prev_y_raw: torch.Tensor,
        prev_peak_y_raw: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        flags = self.flags
        env_scaled = x_seq_scaled[:, :, 0:5]
        thickness_raw = x_seq_raw[:, :, 5:5 + self.num_materials]

        env_z = self.env_proj(env_scaled)
        B, L, M = thickness_raw.shape

        if bool(flags["use_structure"]):
            if bool(flags["use_material_token"]):
                thickness_flat = thickness_raw.reshape(B * L, M)
                struct_z_flat, _, attn_flat = self.struct_encoder(thickness_flat)
                struct_z_seq = struct_z_flat.reshape(B, L, -1)
                material_attention = attn_flat.reshape(B, L, M)
            else:
                thickness_feat = torch.log1p(torch.relu(thickness_raw))
                struct_z_seq = self.thickness_mlp(thickness_feat)
                material_attention = thickness_feat / (thickness_feat.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            struct_z_seq = torch.zeros_like(env_z)
            material_attention = torch.zeros(B, L, M, device=x_seq_raw.device, dtype=x_seq_raw.dtype)

        h = self.input_fusion(torch.cat([env_z, struct_z_seq], dim=-1))
        h = self.tcn(h)
        h, _ = self.gru(h)

        h_last = h[:, -1, :]
        if bool(flags["use_temporal_attention"]):
            temporal_weight = torch.softmax(self.temporal_attn(h), dim=1)
            h_context = torch.sum(temporal_weight * h, dim=1)
        else:
            temporal_weight = torch.zeros(B, L, 1, device=x_seq_raw.device, dtype=x_seq_raw.dtype)
            temporal_weight[:, -1, :] = 1.0
            h_context = h_last

        curr_env_raw = x_seq_raw[:, -1, 0:5]
        delta_load = torch.relu(curr_env_raw[:, 1:2])
        temp = curr_env_raw[:, 2:3]
        delta_temp = curr_env_raw[:, 3:4]
        temp_ma3 = curr_env_raw[:, 4:5]

        prev_y_state = torch.log1p(torch.relu(prev_y_raw)) if bool(flags["use_prev_y_state"]) else torch.zeros_like(prev_y_raw)
        prev_peak_state = torch.log1p(torch.relu(prev_peak_y_raw)) if bool(flags["use_prev_peak_state"]) else torch.zeros_like(prev_peak_y_raw)
        state_input = torch.cat([h_context, h_last, prev_y_state, prev_peak_state], dim=-1)
        state_z = self.state_fusion(state_input)

        temp_norm = temp / 40.0
        delta_temp_norm = delta_temp / 20.0
        temp_ma3_norm = temp_ma3 / 40.0
        if bool(flags["use_thermal_gate"]):
            thermal_gate = torch.sigmoid(
                self.thermal_gate_head(torch.cat([state_z, temp_norm, delta_temp_norm, temp_ma3_norm], dim=-1))
            )
        else:
            thermal_gate = torch.ones_like(delta_load)

        damage_rate = F.softplus(self.damage_rate_head(state_z))
        delta_load_scale = torch.log1p(delta_load) / math.log(1000.0 + 1.0)
        damage_increment = delta_load_scale * damage_rate * thermal_gate
        damage_increment = torch.clamp(damage_increment, min=0.0, max=5.0)

        base_damage = prev_peak_y_raw if bool(flags["use_prev_peak_state"]) else prev_y_raw
        permanent_damage = torch.relu(base_damage) + damage_increment

        recovery_bound = 0.20 * torch.sqrt(torch.relu(permanent_damage) + 1.0) + 1.5
        if bool(flags["use_reversible_state"]):
            recovery_raw = torch.tanh(self.recovery_head(state_z))
            reversible_response = recovery_bound * recovery_raw
            reversible_enabled = torch.ones_like(permanent_damage)
        else:
            reversible_response = torch.zeros_like(permanent_damage)
            reversible_enabled = torch.zeros_like(permanent_damage)

        dual_state_mu = torch.relu(permanent_damage + reversible_response)

        if bool(flags["use_observation_adapter"]):
            obs_delta = 1.2 * torch.tanh(self.obs_delta_head(state_z))
            residual_state_mu = torch.relu(prev_y_raw + damage_increment + obs_delta)
            raw_obs_gate = torch.sigmoid(self.obs_gate_head(torch.cat([state_z, temp_norm, delta_temp_norm], dim=-1)))
            # 保留一个小的物理双状态最低融合比例，避免最终均值完全绕开双状态路径。
            obs_gate = self.min_physics_fusion_gate + (1.0 - self.min_physics_fusion_gate) * raw_obs_gate
            mu_raw = torch.relu(obs_gate * dual_state_mu + (1.0 - obs_gate) * residual_state_mu)
        else:
            obs_gate = torch.ones_like(dual_state_mu)
            residual_state_mu = dual_state_mu
            mu_raw = dual_state_mu

        physics_mu_raw = mu_raw
        physics_mu_log = torch.log1p(torch.relu(physics_mu_raw))

        if bool(flags.get("use_state_calibrator", True)):
            calib_input = torch.cat([
                state_z,
                torch.log1p(torch.relu(prev_y_raw)),
                torch.log1p(torch.relu(prev_peak_y_raw)),
                delta_load_scale,
                temp_norm,
                delta_temp_norm,
            ], dim=-1)
            # 采用有界 log-domain 修正，避免校准层把物理状态完全覆盖。
            state_calibration_gate = torch.sigmoid(self.state_calibrator_gate(calib_input))
            state_calibration_delta_log = 0.18 * torch.tanh(self.state_calibrator_head(calib_input))
            mu_log = physics_mu_log + state_calibration_gate * state_calibration_delta_log
            mu_raw = torch.relu(torch.expm1(mu_log))
        else:
            state_calibration_gate = torch.zeros_like(physics_mu_raw)
            state_calibration_delta_log = torch.zeros_like(physics_mu_raw)
            mu_log = physics_mu_log

        mu_scaled = (mu_log - y_mean) / y_std

        var_scaled = F.softplus(self.var_head(state_z)) + 1e-4
        risk_logit = self.risk_head(torch.cat([state_z, mu_raw], dim=-1))
        risk_prob = torch.sigmoid(risk_logit)

        return {
            "mu_scaled": mu_scaled,
            "var_scaled": var_scaled,
            "mu_raw": mu_raw,
            "physics_mu_raw": physics_mu_raw,
            "state_calibration_delta_mm": mu_raw - physics_mu_raw,
            "state_calibration_gate": state_calibration_gate,
            "state_calibration_delta_log": state_calibration_delta_log,
            "permanent_damage": permanent_damage,
            "damage_increment": damage_increment,
            "reversible_response": reversible_response,
            "recovery_bound": recovery_bound,
            "reversible_enabled": reversible_enabled,
            "dual_state_mu": dual_state_mu,
            "residual_state_mu": residual_state_mu,
            "thermal_gate": thermal_gate,
            "risk_prob": risk_prob,
            "obs_gate": obs_gate,
            "material_attention": material_attention,
            "temporal_attention": temporal_weight.squeeze(-1),
        }


# =============================================================================
# 6. 损失函数与指标
# =============================================================================

def tdr_loss(
    out: Dict[str, torch.Tensor],
    y_scaled: torch.Tensor,
    y_raw_mm: torch.Tensor,
    risk_threshold_mm: float,
    epoch: int,
    total_epochs: int,
    use_nll: bool = True,
    use_risk: bool = True,
    prev_y_raw: Optional[torch.Tensor] = None,
    delta_loss_weight: float = 0.0,
    delta_loss_scale_mm: float = 8.0,
    dual_state_loss_weight: float = 0.0,
    reversible_aux_weight: float = 0.0,
    reversible_reg_weight: float = 0.002,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    mu = out["mu_scaled"]
    var = out["var_scaled"]

    huber = F.smooth_l1_loss(mu, y_scaled)
    residual = y_scaled - mu
    nll = (0.5 * torch.log(var) + 0.5 * residual.pow(2) / var).mean()

    # 风险分支：预测是否超过养护阈值。
    risk_target = (y_raw_mm >= risk_threshold_mm).float()
    if use_risk:
        risk_loss = F.binary_cross_entropy(out["risk_prob"], risk_target)
    else:
        risk_loss = torch.zeros((), dtype=y_scaled.dtype, device=y_scaled.device)

    # 回弹分支不应无限制补偿主预测，轻微正则即可。
    # 注意：冷启动过滤后，可逆响应幅值通常更小，若正则过强会导致A5难以体现双状态价值。
    recovery_reg = (out["reversible_response"].pow(2).mean())
    increment_reg = out["damage_increment"].abs().mean()

    # 双状态均值辅助监督：让永久+可逆状态本身也具备解释观测车辙的能力，
    # 防止最终预测完全被 observation adapter 吸收。
    if float(dual_state_loss_weight) > 0 and "dual_state_mu" in out:
        scale = max(float(delta_loss_scale_mm), 1e-6)
        dual_state_huber = F.smooth_l1_loss((out["dual_state_mu"] - y_raw_mm) / scale, torch.zeros_like(y_raw_mm))
    else:
        dual_state_huber = torch.zeros((), dtype=y_scaled.dtype, device=y_scaled.device)

    # 可逆响应弱监督：用观测值相对于永久损伤状态的短期偏差作为可逆分支的软目标。
    # 仅在完整模型启用可逆分支时生效，避免A5变体因不存在该状态而被额外惩罚。
    if (
        float(reversible_aux_weight) > 0
        and "recovery_bound" in out
        and "reversible_enabled" in out
        and bool((out["reversible_enabled"].detach().mean() > 0.5).item())
    ):
        scale = max(float(delta_loss_scale_mm), 1e-6)
        rev_target = y_raw_mm - out["permanent_damage"].detach()
        bound = out["recovery_bound"].detach()
        rev_target = torch.maximum(torch.minimum(rev_target, bound), -bound)
        reversible_aux = F.smooth_l1_loss(out["reversible_response"] / scale, rev_target / scale)
    else:
        reversible_aux = torch.zeros((), dtype=y_scaled.dtype, device=y_scaled.device)

    # 增量监督：车辙演化数据具有强短期惯性，直接约束 y_t-y_{t-1}
    # 能迫使模型学习“在上一期状态基础上增长多少”，而不是只拟合绝对值。
    if prev_y_raw is not None and float(delta_loss_weight) > 0:
        scale = max(float(delta_loss_scale_mm), 1e-6)
        pred_delta = (out["mu_raw"] - prev_y_raw) / scale
        true_delta = (y_raw_mm - prev_y_raw) / scale
        delta_huber = F.smooth_l1_loss(pred_delta, true_delta)
    else:
        delta_huber = torch.zeros((), dtype=y_scaled.dtype, device=y_scaled.device)

    # 概率损失逐步加权，前期先让均值拟合稳定。
    nll_weight = 0.02 + 0.08 * min(epoch / max(total_epochs, 1), 1.0) if use_nll else 0.0
    risk_weight = 0.03 if use_risk else 0.0
    loss = (
        huber
        + nll_weight * nll
        + risk_weight * risk_loss
        + float(delta_loss_weight) * delta_huber
        + float(dual_state_loss_weight) * dual_state_huber
        + float(reversible_aux_weight) * reversible_aux
        + float(reversible_reg_weight) * recovery_reg
        + 0.001 * increment_reg
    )

    parts = {
        "loss": float(loss.detach().cpu()),
        "huber": float(huber.detach().cpu()),
        "nll": float(nll.detach().cpu()),
        "risk": float(risk_loss.detach().cpu()),
        "delta_huber": float(delta_huber.detach().cpu()),
        "dual_state_huber": float(dual_state_huber.detach().cpu()),
        "reversible_aux": float(reversible_aux.detach().cpu()),
        "recovery_reg": float(recovery_reg.detach().cpu()),
        "increment_reg": float(increment_reg.detach().cpu()),
    }
    return loss, parts


def inverse_transform_prediction(
    mu_scaled: np.ndarray,
    var_scaled: np.ndarray,
    y_scaler: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从标准化 log1p 响应域恢复到 mm，同时使用 Delta Method 估计原始尺度标准差。"""
    mu_scaled = mu_scaled.reshape(-1, 1)
    mu_log = y_scaler.inverse_transform(mu_scaled).reshape(-1)
    mu_raw = np.expm1(mu_log)
    mu_raw = np.maximum(mu_raw, 0.0)

    var_scaled = np.maximum(var_scaled.reshape(-1), 1e-9)
    var_log = var_scaled * float(y_scaler.scale_[0] ** 2)
    std_raw = (mu_raw + 1.0) * np.sqrt(var_log)
    std_raw = np.maximum(std_raw, 1e-9)
    return mu_raw, std_raw, mu_log


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_std: Optional[np.ndarray] = None) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    result = {
        "R2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
        "RMSE_mm": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE_mm": float(mean_absolute_error(y_true, y_pred)),
    }
    if y_std is not None:
        y_std = np.asarray(y_std).reshape(-1)
        lower = y_pred - 1.96 * y_std
        upper = y_pred + 1.96 * y_std
        result["PICP_95"] = float(np.mean((y_true >= lower) & (y_true <= upper)))
        result["MPIW_95_mm"] = float(np.mean(upper - lower))
        result["NLL_raw_proxy"] = float(np.mean(0.5 * np.log(2 * np.pi * y_std ** 2) + 0.5 * ((y_true - y_pred) ** 2) / (y_std ** 2)))
    return result


def threshold_time_error(df_pred: pd.DataFrame, thresholds: List[float]) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        for sid, g in df_pred.groupby("STR_name"):
            g = g.sort_values("Cum_Load_10k")
            true_hit = g.loc[g["y_true_mm"] >= thr, "Cum_Load_10k"]
            pred_hit = g.loc[g["y_pred_mm"] >= thr, "Cum_Load_10k"]
            true_t = float(true_hit.iloc[0]) if len(true_hit) else np.nan
            pred_t = float(pred_hit.iloc[0]) if len(pred_hit) else np.nan
            if np.isnan(true_t) and np.isnan(pred_t):
                abs_err = np.nan
                status = "both_not_reached"
            elif np.isnan(true_t):
                abs_err = np.nan
                status = "false_alarm"
            elif np.isnan(pred_t):
                abs_err = np.nan
                status = "missed"
            else:
                abs_err = abs(pred_t - true_t)
                status = "ok"
            rows.append({
                "Threshold_mm": thr,
                "STR_name": sid,
                "True_Time_10k": true_t,
                "Pred_Time_10k": pred_t,
                "Abs_Time_Error_10k": abs_err,
                "Status": status,
            })
    return pd.DataFrame(rows)


# =============================================================================
# 7. 训练与验证
# =============================================================================

def make_loader(
    X_scaled: np.ndarray,
    X_raw: np.ndarray,
    y_scaled: np.ndarray,
    y_raw: np.ndarray,
    prev_y: np.ndarray,
    prev_peak: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(X_raw, dtype=torch.float32),
        torch.tensor(y_scaled.reshape(-1, 1), dtype=torch.float32),
        torch.tensor(y_raw.reshape(-1, 1), dtype=torch.float32),
        torch.tensor(prev_y.reshape(-1, 1), dtype=torch.float32),
        torch.tensor(prev_peak.reshape(-1, 1), dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate_tdr_model(
    model: nn.Module,
    X_scaled: np.ndarray,
    X_raw: np.ndarray,
    y_scaler: StandardScaler,
    prev_y: np.ndarray,
    prev_peak: np.ndarray,
    y_mean_t: torch.Tensor,
    y_std_t: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> Dict[str, np.ndarray]:
    model.eval()
    outputs: Dict[str, List[np.ndarray]] = {
        "mu_scaled": [], "var_scaled": [], "mu_raw_model": [],
        "physics_mu_raw": [], "state_calibration_delta_mm": [], "state_calibration_gate": [],
        "permanent_damage": [], "damage_increment": [], "reversible_response": [],
        "recovery_bound": [], "dual_state_mu": [], "residual_state_mu": [],
        "thermal_gate": [], "risk_prob": [], "obs_gate": [],
        "material_attention": [], "temporal_attention": [],
    }

    n = X_scaled.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xs = torch.tensor(X_scaled[i:i + batch_size], dtype=torch.float32, device=device)
            xr = torch.tensor(X_raw[i:i + batch_size], dtype=torch.float32, device=device)
            py = torch.tensor(prev_y[i:i + batch_size].reshape(-1, 1), dtype=torch.float32, device=device)
            pp = torch.tensor(prev_peak[i:i + batch_size].reshape(-1, 1), dtype=torch.float32, device=device)
            out = model(xs, xr, py, pp, y_mean_t, y_std_t)
            mu_t = out["mu_scaled"]
            var_t = out.get("var_scaled", torch.full_like(mu_t, 1e-4))
            nan_t = torch.full_like(mu_t, float("nan"))
            zero_t = torch.zeros_like(mu_t)
            outputs["mu_scaled"].append(mu_t.detach().cpu().numpy().reshape(-1))
            outputs["var_scaled"].append(var_t.detach().cpu().numpy().reshape(-1))
            outputs["mu_raw_model"].append(out.get("mu_raw", nan_t).detach().cpu().numpy().reshape(-1))
            outputs["physics_mu_raw"].append(out.get("physics_mu_raw", nan_t).detach().cpu().numpy().reshape(-1))
            outputs["state_calibration_delta_mm"].append(out.get("state_calibration_delta_mm", nan_t).detach().cpu().numpy().reshape(-1))
            outputs["state_calibration_gate"].append(out.get("state_calibration_gate", nan_t).detach().cpu().numpy().reshape(-1))
            outputs["permanent_damage"].append(out.get("permanent_damage", nan_t).detach().cpu().numpy().reshape(-1))
            outputs["damage_increment"].append(out.get("damage_increment", nan_t).detach().cpu().numpy().reshape(-1))
            outputs["reversible_response"].append(out.get("reversible_response", nan_t).detach().cpu().numpy().reshape(-1))
            outputs["recovery_bound"].append(out.get("recovery_bound", nan_t).detach().cpu().numpy().reshape(-1))
            outputs["dual_state_mu"].append(out.get("dual_state_mu", nan_t).detach().cpu().numpy().reshape(-1))
            outputs["residual_state_mu"].append(out.get("residual_state_mu", nan_t).detach().cpu().numpy().reshape(-1))
            outputs["thermal_gate"].append(out.get("thermal_gate", nan_t).detach().cpu().numpy().reshape(-1))
            outputs["risk_prob"].append(out.get("risk_prob", nan_t).detach().cpu().numpy().reshape(-1))
            outputs["obs_gate"].append(out.get("obs_gate", nan_t).detach().cpu().numpy().reshape(-1))
            material_attention = out.get("material_attention", torch.zeros(xs.size(0), len(SEMANTIC_COLS), device=device, dtype=xs.dtype))
            if material_attention.ndim == 3:
                material_attention = material_attention[:, -1, :]
            outputs["material_attention"].append(material_attention.detach().cpu().numpy())
            temporal_attention = out.get("temporal_attention", torch.zeros(xs.size(0), X_scaled.shape[1], device=device, dtype=xs.dtype))
            outputs["temporal_attention"].append(temporal_attention.detach().cpu().numpy())

    out_np = {k: np.concatenate(v) for k, v in outputs.items()}
    mu_pred, std_pred, _ = inverse_transform_prediction(out_np["mu_scaled"], out_np["var_scaled"], y_scaler)
    out_np["y_pred_mm"] = mu_pred
    out_np["y_std_mm"] = std_pred
    # 对确定性基线，mu_raw_model 等物理状态不存在，用最终反变换预测补齐均值字段，
    # 其余物理解释字段保留 NaN，避免误用于双状态解释。
    if np.all(np.isnan(out_np.get("mu_raw_model", np.array([np.nan])))):
        out_np["mu_raw_model"] = mu_pred
    return out_np


def build_residual_calibrator_features(
    X_raw: np.ndarray,
    prev_y: np.ndarray,
    prev_peak: np.ndarray,
    pred: Dict[str, np.ndarray],
) -> np.ndarray:
    """构建残差校正器输入特征。

    特征只来自当前折训练数据可见的信息和 TDR-PRSN 的物理状态输出，
    不使用验证结构真实标签，避免跨结构信息泄漏。
    """
    X_last = X_raw[:, -1, :].astype(np.float32)
    cols = [
        X_last,
        prev_y.reshape(-1, 1).astype(np.float32),
        prev_peak.reshape(-1, 1).astype(np.float32),
        pred["y_pred_mm"].reshape(-1, 1).astype(np.float32),
        pred["y_std_mm"].reshape(-1, 1).astype(np.float32),
        pred.get("physics_mu_raw", pred["y_pred_mm"]).reshape(-1, 1).astype(np.float32),
        pred.get("state_calibration_delta_mm", np.zeros_like(pred["y_pred_mm"])).reshape(-1, 1).astype(np.float32),
        pred.get("state_calibration_gate", np.zeros_like(pred["y_pred_mm"])).reshape(-1, 1).astype(np.float32),
        pred["permanent_damage"].reshape(-1, 1).astype(np.float32),
        pred["damage_increment"].reshape(-1, 1).astype(np.float32),
        pred["reversible_response"].reshape(-1, 1).astype(np.float32),
        pred.get("dual_state_mu", pred["y_pred_mm"]).reshape(-1, 1).astype(np.float32),
        pred.get("residual_state_mu", pred["y_pred_mm"]).reshape(-1, 1).astype(np.float32),
        pred["thermal_gate"].reshape(-1, 1).astype(np.float32),
        pred["obs_gate"].reshape(-1, 1).astype(np.float32),
    ]
    X = np.concatenate(cols, axis=1)
    return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def train_one_fold(
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    X_seq_raw: np.ndarray,
    y: np.ndarray,
    prev_y: np.ndarray,
    prev_peak: np.ndarray,
    meta_df: pd.DataFrame,
    cfg: Config,
    device: torch.device,
    variant: str = "full",
    model_label: Optional[str] = None,
    model_type: str = "TDRPRSN",
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float], Optional[pd.DataFrame], Optional[Dict[str, float]]]:
    X_tr_raw, X_va_raw = X_seq_raw[train_idx], X_seq_raw[val_idx]
    y_tr, y_va = y[train_idx], y[val_idx]
    py_tr, py_va = prev_y[train_idx], prev_y[val_idx]
    pp_tr, pp_va = prev_peak[train_idx], prev_peak[val_idx]

    num_features = X_seq_raw.shape[-1]

    x_scaler = StandardScaler()
    X_tr_scaled_flat = x_scaler.fit_transform(X_tr_raw.reshape(-1, num_features))
    X_va_scaled_flat = x_scaler.transform(X_va_raw.reshape(-1, num_features))
    X_tr_scaled = X_tr_scaled_flat.reshape(X_tr_raw.shape).astype(np.float32)
    X_va_scaled = X_va_scaled_flat.reshape(X_va_raw.shape).astype(np.float32)

    y_scaler = StandardScaler()
    y_tr_log = np.log1p(np.maximum(y_tr, 0.0)).reshape(-1, 1)
    y_va_log = np.log1p(np.maximum(y_va, 0.0)).reshape(-1, 1)
    y_tr_scaled = y_scaler.fit_transform(y_tr_log).reshape(-1).astype(np.float32)
    y_va_scaled = y_scaler.transform(y_va_log).reshape(-1).astype(np.float32)

    y_mean_t = torch.tensor(float(y_scaler.mean_[0]), dtype=torch.float32, device=device)
    y_std_t = torch.tensor(float(y_scaler.scale_[0]), dtype=torch.float32, device=device)

    train_loader = make_loader(
        X_tr_scaled, X_tr_raw, y_tr_scaled, y_tr, py_tr, pp_tr,
        batch_size=cfg.batch_size, shuffle=True,
    )

    model, default_model_label, use_physics_losses, use_probabilistic_metrics = build_neural_model(
        model_type=model_type,
        num_features=num_features,
        cfg=cfg,
        variant=variant,
        device=device,
    )
    if model_label is None:
        if str(model_type).strip().upper() in {"TDRPRSN", "TDR-PRSN"} and variant == "full":
            model_label = "TDR-PRSN-Full"
        else:
            model_label = default_model_label
    variant_flags = get_ablation_flags(variant) if use_physics_losses else {"use_probabilistic": False, "use_risk": False}

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.05)

    best_state = copy.deepcopy(model.state_dict())
    best_val_rmse = float("inf")
    best_epoch = 0
    wait = 0

    iterator = range(1, cfg.epochs + 1)
    if tqdm is not None:
        iterator = tqdm(iterator, desc=f"Fold {fold} {model_label}", leave=False)

    history = []
    for epoch in iterator:
        model.train()
        train_losses = []
        for xs, xr, ys, yr, py, pp in train_loader:
            xs = xs.to(device)
            xr = xr.to(device)
            ys = ys.to(device)
            yr = yr.to(device)
            py = py.to(device)
            pp = pp.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(xs, xr, py, pp, y_mean_t, y_std_t)
            if use_physics_losses:
                loss, parts = tdr_loss(
                    out, ys, yr, cfg.risk_threshold_mm, epoch, cfg.epochs,
                    use_nll=bool(variant_flags["use_probabilistic"]),
                    use_risk=bool(variant_flags["use_risk"]),
                    prev_y_raw=py,
                    delta_loss_weight=float(getattr(cfg, "delta_loss_weight", 0.0)),
                    delta_loss_scale_mm=float(getattr(cfg, "delta_loss_scale_mm", 8.0)),
                    dual_state_loss_weight=float(getattr(cfg, "dual_state_loss_weight", 0.0)),
                    reversible_aux_weight=float(getattr(cfg, "reversible_aux_weight", 0.0)),
                    reversible_reg_weight=float(getattr(cfg, "reversible_reg_weight", 0.002)),
                )
            else:
                # 纯数据驱动深度学习基线：只优化均值回归损失，不使用物理损失、风险损失或双状态正则。
                loss = F.smooth_l1_loss(out["mu_scaled"], ys)
                parts = {"loss": float(loss.detach().cpu()), "huber": float(loss.detach().cpu())}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(parts["loss"])

        scheduler.step()

        # 为了避免小样本多折训练时每个 epoch 都验证造成大量重复开销，
        # 默认每 val_every 个 epoch 验证一次；第 1 轮和最后一轮仍强制验证。
        val_every = max(1, int(getattr(cfg, "val_every", 1)))
        do_validate = (epoch == 1) or (epoch % val_every == 0) or (epoch == cfg.epochs)
        if do_validate:
            pred_val = evaluate_tdr_model(model, X_va_scaled, X_va_raw, y_scaler, py_va, pp_va, y_mean_t, y_std_t, device)
            val_rmse = float(np.sqrt(mean_squared_error(y_va, pred_val["y_pred_mm"])))
            val_mae = float(mean_absolute_error(y_va, pred_val["y_pred_mm"]))
            history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_rmse": val_rmse, "val_mae": val_mae})

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1

            if tqdm is not None and hasattr(iterator, "set_postfix"):
                iterator.set_postfix({"rmse": f"{val_rmse:.4f}", "best": f"{best_val_rmse:.4f}", "val_every": val_every})

            if wait >= cfg.patience:
                break

    model.load_state_dict(best_state)
    pred = evaluate_tdr_model(model, X_va_scaled, X_va_raw, y_scaler, py_va, pp_va, y_mean_t, y_std_t, device)

    pred_df = meta_df.iloc[val_idx].copy().reset_index(drop=True)
    pred_df["Fold"] = fold
    pred_df["Model"] = model_label
    pred_df["y_true_mm"] = y_va
    pred_df["y_pred_mm"] = pred["y_pred_mm"]
    pred_df["y_std_mm"] = pred["y_std_mm"] if use_probabilistic_metrics else np.nan
    pred_df["lower95_mm"] = pred_df["y_pred_mm"] - 1.96 * pred_df["y_std_mm"]
    pred_df["upper95_mm"] = pred_df["y_pred_mm"] + 1.96 * pred_df["y_std_mm"]
    # 作图和工程解释时车辙深度不能为负，保留原始 lower95，同时新增物理裁剪后的下界。
    pred_df["lower95_mm_clipped"] = np.maximum(pred_df["lower95_mm"], 0.0)
    pred_df["physics_mu_raw_mm"] = pred.get("physics_mu_raw", pred["y_pred_mm"])
    pred_df["state_calibration_delta_mm"] = pred.get("state_calibration_delta_mm", np.zeros_like(pred["y_pred_mm"]))
    pred_df["state_calibration_gate"] = pred.get("state_calibration_gate", np.zeros_like(pred["y_pred_mm"]))
    pred_df["permanent_damage_mm"] = pred["permanent_damage"]
    pred_df["damage_increment_mm"] = pred["damage_increment"]
    pred_df["reversible_response_mm"] = pred["reversible_response"]
    pred_df["thermal_gate"] = pred["thermal_gate"]
    pred_df["risk_prob"] = pred["risk_prob"]
    pred_df["obs_gate"] = pred["obs_gate"]
    if "material_attention" in pred:
        for mi, col in enumerate(SEMANTIC_COLS):
            pred_df[f"mat_attn_{col}"] = pred["material_attention"][:, mi]
    if "temporal_attention" in pred:
        for ti in range(pred["temporal_attention"].shape[1]):
            lag = pred["temporal_attention"].shape[1] - 1 - ti
            pred_df[f"time_attn_tminus_{lag}"] = pred["temporal_attention"][:, ti]
    pred_df["abs_error_mm"] = np.abs(pred_df["y_true_mm"] - pred_df["y_pred_mm"])

    if use_probabilistic_metrics:
        metrics = calc_metrics(y_va, pred["y_pred_mm"], pred["y_std_mm"])
    else:
        metrics = calc_metrics(y_va, pred["y_pred_mm"], None)
    metrics["Fold"] = fold
    metrics["Best_Epoch"] = best_epoch
    metrics["Best_Val_RMSE_mm"] = best_val_rmse

    # Persistence baseline: 直接使用上一期车辙作为预测。
    persistence_pred = py_va.reshape(-1)
    persistence_metrics = calc_metrics(y_va, persistence_pred, None)
    persistence_metrics["Fold"] = fold

    rc_pred_df: Optional[pd.DataFrame] = None
    rc_metrics: Optional[Dict[str, float]] = None
    if (
        variant == "full"
        and str(model_type).strip().upper() in {"TDRPRSN", "TDR-PRSN"}
        and bool(getattr(cfg, "enable_residual_calibration", True))
        and XGBOOST_AVAILABLE
        and len(y_tr) >= 50
    ):
        # 跨结构残差校正：校正器只在训练结构上学习 TDR-PRSN 的残差，
        # 再应用于当前未见验证结构，因此不引入验证标签泄漏。
        pred_tr = evaluate_tdr_model(model, X_tr_scaled, X_tr_raw, y_scaler, py_tr, pp_tr, y_mean_t, y_std_t, device)
        X_cal_tr = build_residual_calibrator_features(X_tr_raw, py_tr, pp_tr, pred_tr)
        X_cal_va = build_residual_calibrator_features(X_va_raw, py_va, pp_va, pred)
        cal_scaler = StandardScaler()
        X_cal_tr_s = cal_scaler.fit_transform(X_cal_tr)
        X_cal_va_s = cal_scaler.transform(X_cal_va)
        residual_tr = y_tr.reshape(-1) - pred_tr["y_pred_mm"].reshape(-1)
        try:
            rc_model = xgb.XGBRegressor(
                n_estimators=int(getattr(cfg, "residual_calibrator_estimators", 260)),
                max_depth=2,
                learning_rate=0.035,
                subsample=0.90,
                colsample_bytree=0.85,
                reg_lambda=4.0,
                reg_alpha=0.05,
                min_child_weight=5.0,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=cfg.seed + 1000 + fold,
            )
            rc_model.fit(X_cal_tr_s, residual_tr)
            strength = float(getattr(cfg, "residual_calibrator_strength", 0.75))
            residual_va_hat = rc_model.predict(X_cal_va_s).reshape(-1)
            y_pred_rc = np.maximum(pred["y_pred_mm"].reshape(-1) + strength * residual_va_hat, 0.0)

            # 使用训练残差的剩余尺度对区间做保守扩展，避免残差校正后过度自信。
            residual_tr_after = residual_tr - strength * rc_model.predict(X_cal_tr_s).reshape(-1)
            sigma_extra = float(np.nanstd(residual_tr_after))
            sigma_extra = max(sigma_extra, 0.05 * float(np.nanstd(y_tr)))
            y_std_rc = np.sqrt(np.maximum(pred["y_std_mm"].reshape(-1), 1e-9) ** 2 + sigma_extra ** 2)

            rc_pred_df = pred_df.copy()
            rc_pred_df["Model"] = "TDR-PRSN-RC"
            rc_pred_df["base_y_pred_mm"] = pred_df["y_pred_mm"].values
            rc_pred_df["residual_calibration_mm"] = y_pred_rc - pred_df["y_pred_mm"].values
            rc_pred_df["y_pred_mm"] = y_pred_rc
            rc_pred_df["y_std_mm"] = y_std_rc
            rc_pred_df["lower95_mm"] = rc_pred_df["y_pred_mm"] - 1.96 * rc_pred_df["y_std_mm"]
            rc_pred_df["upper95_mm"] = rc_pred_df["y_pred_mm"] + 1.96 * rc_pred_df["y_std_mm"]
            rc_pred_df["lower95_mm_clipped"] = np.maximum(rc_pred_df["lower95_mm"], 0.0)
            rc_pred_df["abs_error_mm"] = np.abs(rc_pred_df["y_true_mm"] - rc_pred_df["y_pred_mm"])

            rc_metrics = calc_metrics(y_va, y_pred_rc, y_std_rc)
            rc_metrics["Fold"] = fold
            rc_metrics["Best_Epoch"] = best_epoch
            rc_metrics["Best_Val_RMSE_mm"] = best_val_rmse
        except Exception as e:
            log(f"[提醒] Fold {fold} 残差校正失败，继续使用基础 TDR-PRSN。原因: {e}")
            rc_pred_df = None
            rc_metrics = None

    return pred_df, metrics, persistence_metrics, rc_pred_df, rc_metrics




def _make_point_prediction_frame(
    meta_df: pd.DataFrame,
    idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    fold: int,
    y_std: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """构造统一的点预测结果表，供基线模型与 TDR-PRSN 做同图对比。"""
    df = meta_df.iloc[idx].copy().reset_index(drop=True)
    df["Fold"] = fold
    df["Model"] = model_name
    df["y_true_mm"] = np.asarray(y_true).reshape(-1)
    df["y_pred_mm"] = np.asarray(y_pred).reshape(-1)
    if y_std is None:
        df["y_std_mm"] = np.nan
        df["lower95_mm"] = np.nan
        df["upper95_mm"] = np.nan
        df["lower95_mm_clipped"] = np.nan
    else:
        df["y_std_mm"] = np.asarray(y_std).reshape(-1)
        df["lower95_mm"] = df["y_pred_mm"] - 1.96 * df["y_std_mm"]
        df["upper95_mm"] = df["y_pred_mm"] + 1.96 * df["y_std_mm"]
        df["lower95_mm_clipped"] = np.maximum(df["lower95_mm"], 0.0)
    df["abs_error_mm"] = np.abs(df["y_true_mm"] - df["y_pred_mm"])
    df["residual_mm"] = df["y_pred_mm"] - df["y_true_mm"]
    return df


def _fit_xgb_cv_with_predictions(
    X_design: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    meta_df: pd.DataFrame,
    cfg: Config,
    model_name: str,
    splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> Tuple[List[Dict[str, float]], pd.DataFrame]:
    """运行 XGBoost 基线，同时返回逐样本预测，便于论文绘制基线对比图。"""
    rows: List[Dict[str, float]] = []
    pred_frames: List[pd.DataFrame] = []
    if splits is None:
        splits = make_structure_cv_splits(X_design, y, groups, cfg, out_dir=None, report_prefix="xgb")
    for fold, (tr, va) in enumerate(splits, start=1):
        x_scaler = StandardScaler()
        X_tr = x_scaler.fit_transform(X_design[tr])
        X_va = x_scaler.transform(X_design[va])
        y_tr_log = np.log1p(np.maximum(y[tr], 0.0))
        model = xgb.XGBRegressor(
            n_estimators=360,
            max_depth=4,
            learning_rate=0.035,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_lambda=2.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=cfg.seed + fold,
        )
        model.fit(X_tr, y_tr_log)
        pred = np.maximum(np.expm1(model.predict(X_va)), 0.0)
        m = calc_metrics(y[va], pred)
        m["Fold"] = fold
        m["Model"] = model_name
        rows.append(m)
        pred_frames.append(_make_point_prediction_frame(meta_df, va, y[va], pred, model_name, fold))
    pred_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    return rows, pred_df


def run_xgb_baseline_with_predictions(
    X_seq_raw: np.ndarray,
    y: np.ndarray,
    prev_y: np.ndarray,
    prev_peak: np.ndarray,
    groups: np.ndarray,
    meta_df: pd.DataFrame,
    cfg: Config,
    splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    XGBoost 基线，同时输出预测明细。
    输出用于论文对比图：parity plot、结构演化曲线、残差箱线图、指标柱状图。
    """
    if not XGBOOST_AVAILABLE or not cfg.run_xgb:
        return pd.DataFrame(), pd.DataFrame()

    log("[基线] 运行 XGBoost 跨结构基线，并保存逐样本预测用于论文对比图。")
    X_flat = X_seq_raw.reshape(X_seq_raw.shape[0], -1)
    X_last = X_seq_raw[:, -1, :]
    state = np.concatenate([prev_y.reshape(-1, 1), prev_peak.reshape(-1, 1)], axis=1)

    designs = [
        ("XGBoost_flat_sequence", X_flat),
        ("XGBoost_last_step_state", np.concatenate([X_last, state], axis=1)),
        ("XGBoost_sequence_state", np.concatenate([X_flat, state], axis=1)),
    ]

    rows: List[Dict[str, float]] = []
    pred_parts: List[pd.DataFrame] = []
    for model_name, X_design in designs:
        r, p = _fit_xgb_cv_with_predictions(X_design.astype(np.float32), y, groups, meta_df, cfg, model_name, splits=splits)
        rows.extend(r)
        if not p.empty:
            pred_parts.append(p)
    pred_df = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    return pd.DataFrame(rows), pred_df

def _fit_xgb_cv(
    X_design: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: Config,
    model_name: str,
    splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> List[Dict[str, float]]:
    """运行一个 XGBoost 设计矩阵的跨结构验证，划分与主模型保持一致。"""
    rows: List[Dict[str, float]] = []
    if splits is None:
        splits = make_structure_cv_splits(X_design, y, groups, cfg, out_dir=None, report_prefix="xgb")
    for fold, (tr, va) in enumerate(splits, start=1):
        x_scaler = StandardScaler()
        X_tr = x_scaler.fit_transform(X_design[tr])
        X_va = x_scaler.transform(X_design[va])
        y_tr_log = np.log1p(np.maximum(y[tr], 0.0))
        model = xgb.XGBRegressor(
            n_estimators=360,
            max_depth=4,
            learning_rate=0.035,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_lambda=2.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=cfg.seed,
        )
        model.fit(X_tr, y_tr_log)
        pred = np.maximum(np.expm1(model.predict(X_va)), 0.0)
        m = calc_metrics(y[va], pred)
        m["Fold"] = fold
        m["Model"] = model_name
        rows.append(m)
    return rows


def run_xgb_baseline(
    X_seq_raw: np.ndarray,
    y: np.ndarray,
    prev_y: np.ndarray,
    prev_peak: np.ndarray,
    groups: np.ndarray,
    cfg: Config,
    splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> pd.DataFrame:
    """
    XGBoost 基线。
    - XGBoost_flat_sequence：只用荷载-温度-结构序列，作为传统静态/序列展开基线；
    - XGBoost_last_step_state：使用当前步工程特征 + 上一期车辙/历史峰值，更公平地对比状态变量作用；
    - XGBoost_sequence_state：使用完整序列展开 + 上一期车辙/历史峰值，是更强的机器学习状态基线。
    """
    if not XGBOOST_AVAILABLE or not cfg.run_xgb:
        return pd.DataFrame()

    log("[基线] 运行 XGBoost 跨结构基线，包括带上一期状态的公平基线。")
    X_flat = X_seq_raw.reshape(X_seq_raw.shape[0], -1)
    X_last = X_seq_raw[:, -1, :]
    state = np.concatenate([prev_y.reshape(-1, 1), prev_peak.reshape(-1, 1)], axis=1)

    designs = [
        ("XGBoost_flat_sequence", X_flat),
        ("XGBoost_last_step_state", np.concatenate([X_last, state], axis=1)),
        ("XGBoost_sequence_state", np.concatenate([X_flat, state], axis=1)),
    ]

    rows: List[Dict[str, float]] = []
    for model_name, X_design in designs:
        rows.extend(_fit_xgb_cv(X_design.astype(np.float32), y, groups, cfg, model_name, splits=splits))
    return pd.DataFrame(rows)


def run_deep_learning_baselines_with_predictions(
    X_seq_raw: np.ndarray,
    y: np.ndarray,
    prev_y: np.ndarray,
    prev_peak: np.ndarray,
    groups: np.ndarray,
    meta_df: pd.DataFrame,
    cfg: Config,
    device: torch.device,
    splits: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """运行 LSTM 与 Transformer 深度学习序列基线，并输出逐样本预测。

    这些基线使用与 TDR-PRSN 相同的折划分、预处理协议和输入信息，
    但仅采用纯数据驱动均值回归损失，不报告概率指标。
    """
    if not bool(getattr(cfg, "run_dl_baselines", True)):
        return pd.DataFrame(), pd.DataFrame()

    log("[基线] 运行深度学习序列基线：Vanilla_LSTM 与 Vanilla_Transformer。")
    cfg_dl = copy.deepcopy(cfg)
    cfg_dl.epochs = int(getattr(cfg, "dl_baseline_epochs", cfg.epochs))
    cfg_dl.enable_residual_calibration = False
    cfg_dl.run_xgb = False
    cfg_dl.save_excel = False

    metric_rows: List[Dict[str, float]] = []
    pred_frames: List[pd.DataFrame] = []
    for model_type in ["LSTM", "Transformer"]:
        for fold, (train_idx, val_idx) in enumerate(splits, start=1):
            pred_df, metrics, _, _, _ = train_one_fold(
                fold, train_idx, val_idx,
                X_seq_raw, y, prev_y, prev_peak, meta_df,
                cfg_dl, device,
                variant="full",
                model_label=None,
                model_type=model_type,
            )
            metrics["Model"] = pred_df["Model"].iloc[0]
            metric_rows.append(metrics)
            pred_frames.append(pred_df)
            log(f"[Fold {fold}] {pred_df['Model'].iloc[0]}: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE_mm']:.4f}, MAE={metrics['MAE_mm']:.4f}")

    metrics_df = pd.DataFrame(metric_rows)
    preds_df = pd.concat(pred_frames, ignore_index=True, sort=False) if pred_frames else pd.DataFrame()
    return metrics_df, preds_df

def run_cross_structure_training(panel_df: pd.DataFrame, cfg: Config, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    log(f"[训练] 使用设备: {device}")

    X_seq_raw, y, prev_y, prev_peak, groups, meta_df = build_sequence_arrays(panel_df, FEATURE_COLS, cfg.seq_len)
    X_seq_raw, y, prev_y, prev_peak, groups, meta_df = exclude_initial_observation_targets(
        X_seq_raw, y, prev_y, prev_peak, groups, meta_df,
        enabled=bool(getattr(cfg, "exclude_initial_observation_targets", True)),
        out_dir=out_dir,
        report_prefix="main",
    )
    log(f"[训练] 序列样本: X={X_seq_raw.shape}, y={y.shape}, groups={len(np.unique(groups))}")

    splits = make_structure_cv_splits(X_seq_raw, y, groups, cfg, out_dir=out_dir, report_prefix="main")
    n_splits = len(splits)

    all_preds = []
    all_metrics = []
    persistence_metrics = []
    persistence_preds = []
    rc_preds = []
    rc_metrics_rows = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        val_structures = _sort_str_names(np.unique(groups[val_idx]).tolist())
        log(f"\n[Fold {fold}/{n_splits}] 验证结构: {', '.join(val_structures)}")
        pred_df, metrics, pm, rc_pred_df, rc_metrics = train_one_fold(
            fold, train_idx, val_idx,
            X_seq_raw, y, prev_y, prev_peak, meta_df,
            cfg, device, variant="full", model_label="TDR-PRSN-Full",
        )
        all_preds.append(pred_df)
        all_metrics.append(metrics)
        persistence_metrics.append(pm)
        persistence_preds.append(_make_point_prediction_frame(meta_df, val_idx, y[val_idx], prev_y[val_idx].reshape(-1), "Persistence_prev_y", fold))
        if rc_pred_df is not None and rc_metrics is not None:
            rc_preds.append(rc_pred_df)
            rc_metrics_rows.append(rc_metrics)
        log(f"[Fold {fold}] TDR-PRSN: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE_mm']:.4f}, MAE={metrics['MAE_mm']:.4f}, PICP={metrics.get('PICP_95', np.nan):.4f}")
        if rc_metrics is not None:
            log(f"[Fold {fold}] TDR-PRSN-RC: R2={rc_metrics['R2']:.4f}, RMSE={rc_metrics['RMSE_mm']:.4f}, MAE={rc_metrics['MAE_mm']:.4f}, PICP={rc_metrics.get('PICP_95', np.nan):.4f}")
        log(f"[Fold {fold}] Persistence: R2={pm['R2']:.4f}, RMSE={pm['RMSE_mm']:.4f}, MAE={pm['MAE_mm']:.4f}")

    pred_all = pd.concat(all_preds, ignore_index=True)
    rc_pred_all = pd.concat(rc_preds, ignore_index=True) if rc_preds else pd.DataFrame()
    pred_main = rc_pred_all if (bool(getattr(cfg, "use_rc_for_main_outputs", True)) and not rc_pred_all.empty) else pred_all

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.insert(0, "Model", "TDR-PRSN-Full")
    rc_metrics_df = pd.DataFrame(rc_metrics_rows)
    if not rc_metrics_df.empty:
        rc_metrics_df.insert(0, "Model", "TDR-PRSN-RC")
    persistence_df = pd.DataFrame(persistence_metrics)
    persistence_df.insert(0, "Model", "Persistence_prev_y")

    persistence_pred_all = pd.concat(persistence_preds, ignore_index=True) if persistence_preds else pd.DataFrame()
    xgb_metrics_df, xgb_pred_df = run_xgb_baseline_with_predictions(X_seq_raw, y, prev_y, prev_peak, groups, meta_df, cfg, splits=splits)
    dl_metrics_df, dl_pred_df = run_deep_learning_baselines_with_predictions(
        X_seq_raw, y, prev_y, prev_peak, groups, meta_df, cfg, device, splits=splits
    )
    if not dl_pred_df.empty:
        dl_pred_df.to_csv(out_dir / "predictions_deep_learning_baselines.csv", index=False, encoding="utf-8-sig")
    all_model_pred_parts = [pred_all]
    if not rc_pred_all.empty:
        all_model_pred_parts.append(rc_pred_all)
    if not persistence_pred_all.empty:
        all_model_pred_parts.append(persistence_pred_all)
    if not xgb_pred_df.empty:
        all_model_pred_parts.append(xgb_pred_df)
    if not dl_pred_df.empty:
        all_model_pred_parts.append(dl_pred_df)
    predictions_all_models = pd.concat(all_model_pred_parts, ignore_index=True, sort=False)

    parts = [metrics_df]
    if not rc_metrics_df.empty:
        parts.append(rc_metrics_df)
    parts.append(persistence_df)
    if not xgb_metrics_df.empty:
        parts.append(xgb_metrics_df)
    if not dl_metrics_df.empty:
        parts.append(dl_metrics_df)
    metrics_out = pd.concat(parts, ignore_index=True, sort=False)

    # 结构级指标
    by_struct_rows = []
    for sid, g in pred_main.groupby("STR_name"):
        m = calc_metrics(g["y_true_mm"].values, g["y_pred_mm"].values, g["y_std_mm"].values)
        m["STR_name"] = sid
        m["Struct_Class"] = g["Struct_Class"].iloc[0]
        m["N"] = len(g)
        by_struct_rows.append(m)
    by_struct = pd.DataFrame(by_struct_rows).sort_values("STR_name", key=lambda s: s.str.replace("STR", "", regex=False).astype(int))

    threshold_df = threshold_time_error(pred_main, cfg.thresholds)

    # 保存。
    pred_all.to_csv(out_dir / "predictions_tdr_prsn.csv", index=False, encoding="utf-8-sig")
    if not rc_pred_all.empty:
        rc_pred_all.to_csv(out_dir / "predictions_tdr_prsn_rc.csv", index=False, encoding="utf-8-sig")
    pred_main.to_csv(out_dir / "predictions_for_paper_figures.csv", index=False, encoding="utf-8-sig")
    predictions_all_models.to_csv(out_dir / "predictions_all_models_for_figures.csv", index=False, encoding="utf-8-sig")
    metrics_out.to_csv(out_dir / "metrics_cv.csv", index=False, encoding="utf-8-sig")
    by_struct.to_csv(out_dir / "metrics_by_structure.csv", index=False, encoding="utf-8-sig")
    threshold_df.to_csv(out_dir / "threshold_time_error.csv", index=False, encoding="utf-8-sig")

    if cfg.save_excel:
        try:
            with pd.ExcelWriter(out_dir / "tdr_prsn_results.xlsx") as writer:
                pred_all.to_excel(writer, sheet_name="predictions_full", index=False)
                if not rc_pred_all.empty:
                    rc_pred_all.to_excel(writer, sheet_name="predictions_rc", index=False)
                pred_main.to_excel(writer, sheet_name="predictions_for_figures", index=False)
                predictions_all_models.to_excel(writer, sheet_name="predictions_all_models", index=False)
                metrics_out.to_excel(writer, sheet_name="metrics_cv", index=False)
                by_struct.to_excel(writer, sheet_name="metrics_by_structure", index=False)
                threshold_df.to_excel(writer, sheet_name="threshold_time_error", index=False)
        except Exception as e:
            log(f"[提醒] Excel 汇总写入失败，但 CSV 已保存。原因: {e}")

    return pred_main, metrics_out, by_struct, predictions_all_models


# =============================================================================
# 7.5 消融试验
# =============================================================================

def _variant_safe_name(variant: str) -> str:
    return variant.replace("/", "_").replace(" ", "_")


def summarize_model_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in ["R2", "RMSE_mm", "MAE_mm", "PICP_95", "MPIW_95_mm", "NLL_raw_proxy"] if c in metrics_df.columns]
    rows = []
    for model, g in metrics_df.groupby("Model", dropna=False):
        row = {"Model": model, "Folds": int(g["Fold"].nunique()) if "Fold" in g.columns else int(len(g))}
        for c in metric_cols:
            row[f"{c}_mean"] = float(g[c].mean())
            row[f"{c}_std"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    if "RMSE_mm_mean" in out.columns:
        out = out.sort_values("RMSE_mm_mean", ascending=True)
    return out


def run_ablation_study(
    panel_df: pd.DataFrame,
    full_metrics_df: pd.DataFrame,
    cfg: Config,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """运行结构化消融试验，并输出可直接写入论文的汇总表。"""
    if not cfg.run_ablation:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    ablation_dir = ensure_dir(out_dir / "ablation")
    log("\n[消融] 启动结构化消融试验。")

    X_seq_raw, y, prev_y, prev_peak, groups, meta_df = build_sequence_arrays(panel_df, FEATURE_COLS, cfg.seq_len)
    X_seq_raw, y, prev_y, prev_peak, groups, meta_df = exclude_initial_observation_targets(
        X_seq_raw, y, prev_y, prev_peak, groups, meta_df,
        enabled=bool(getattr(cfg, "exclude_initial_observation_targets", True)),
        out_dir=ablation_dir,
        report_prefix="ablation",
    )
    splits = make_structure_cv_splits(X_seq_raw, y, groups, cfg, out_dir=ablation_dir, report_prefix="ablation")
    n_splits = len(splits)

    # 消融通常不需要和完整模型同等 epoch，否则成本过高；但保持相同折划分以保证公平。
    cfg_ab = copy.deepcopy(cfg)
    cfg_ab.epochs = int(cfg.ablation_epochs)
    cfg_ab.run_xgb = False
    cfg_ab.save_excel = False

    all_pred_rows = []
    all_metric_rows = []
    variant_rows = []
    for variant in cfg.ablation_variant_list:
        if variant == "full":
            continue
        flags = get_ablation_flags(variant)
        label = str(flags["label"])
        variant_rows.append({"Variant": variant, "Model": label, "Scientific_question": str(flags["claim"])})
        log(f"\n[消融] {label}: {flags['claim']}")
        for fold, (train_idx, val_idx) in enumerate(splits, start=1):
            pred_df, metrics, _, _, _ = train_one_fold(
                fold, train_idx, val_idx,
                X_seq_raw, y, prev_y, prev_peak, meta_df,
                cfg_ab, device,
                variant=variant,
                model_label=label,
            )
            pred_df["Model"] = label
            metrics["Model"] = label
            all_pred_rows.append(pred_df)
            all_metric_rows.append(metrics)
            log(f"[消融 Fold {fold}] {label}: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE_mm']:.4f}, MAE={metrics['MAE_mm']:.4f}")

    if not all_metric_rows:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ab_metrics = pd.DataFrame(all_metric_rows)
    full_only = full_metrics_df[full_metrics_df["Model"].eq("TDR-PRSN-Full")].copy()
    ab_metrics_all = pd.concat([full_only, ab_metrics], ignore_index=True, sort=False)
    ab_summary = summarize_model_metrics(ab_metrics_all)

    # 相对于完整模型的退化幅度，越大说明被移除模块越重要。
    full_rmse = float(ab_summary.loc[ab_summary["Model"].eq("TDR-PRSN-Full"), "RMSE_mm_mean"].iloc[0]) if (ab_summary["Model"].eq("TDR-PRSN-Full")).any() else np.nan
    full_mae = float(ab_summary.loc[ab_summary["Model"].eq("TDR-PRSN-Full"), "MAE_mm_mean"].iloc[0]) if (ab_summary["Model"].eq("TDR-PRSN-Full")).any() else np.nan
    ab_summary["Delta_RMSE_vs_Full_mm"] = ab_summary["RMSE_mm_mean"] - full_rmse
    ab_summary["Delta_MAE_vs_Full_mm"] = ab_summary["MAE_mm_mean"] - full_mae
    ab_summary["RMSE_Degradation_%"] = 100.0 * ab_summary["Delta_RMSE_vs_Full_mm"] / max(abs(full_rmse), 1e-12)

    ab_preds = pd.concat(all_pred_rows, ignore_index=True, sort=False)
    ab_preds.to_csv(ablation_dir / "ablation_predictions.csv", index=False, encoding="utf-8-sig")
    ab_metrics_all.to_csv(ablation_dir / "ablation_metrics_cv.csv", index=False, encoding="utf-8-sig")
    ab_summary.to_csv(ablation_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(variant_rows).to_csv(ablation_dir / "ablation_design_matrix.csv", index=False, encoding="utf-8-sig")

    if cfg.save_excel:
        try:
            with pd.ExcelWriter(ablation_dir / "ablation_results.xlsx") as writer:
                ab_summary.to_excel(writer, sheet_name="summary", index=False)
                ab_metrics_all.to_excel(writer, sheet_name="fold_metrics", index=False)
                pd.DataFrame(variant_rows).to_excel(writer, sheet_name="design_matrix", index=False)
                ab_preds.to_excel(writer, sheet_name="predictions", index=False)
        except Exception as e:
            log(f"[提醒] 消融 Excel 写入失败，但 CSV 已保存。原因: {e}")

    return ab_summary, ab_metrics_all, ab_preds


# =============================================================================
# 8. 绘图
# =============================================================================

def plot_parity(pred_df: pd.DataFrame, fig_dir: Path) -> None:
    y = pred_df["y_true_mm"].values
    p = pred_df["y_pred_mm"].values
    lo = min(float(np.min(y)), float(np.min(p)))
    hi = max(float(np.max(y)), float(np.max(p)))
    pad = 0.05 * (hi - lo + 1e-9)

    plt.figure(figsize=(6.5, 6), dpi=300)
    plt.scatter(y, p, alpha=0.45, s=14)
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1.5)
    m = calc_metrics(y, p)
    plt.xlabel("Measured rutting depth (mm)")
    plt.ylabel("Predicted rutting depth (mm)")
    plt.title(f"Cross-structure parity plot\nR2={m['R2']:.3f}, RMSE={m['RMSE_mm']:.3f} mm, MAE={m['MAE_mm']:.3f} mm")
    save_paper_figure(fig_dir, "fig1_parity_plot")


def _select_balanced_structure(pred_df: pd.DataFrame, prefer_reversible: bool = False) -> str:
    """选择代表性结构：误差不过分极端；若用于状态分解，则优先选择可逆响应更清楚的结构。"""
    stats = pred_df.groupby("STR_name").agg(
        mae=("abs_error_mm", "mean"),
        rev_amp=("reversible_response_mm", lambda s: float(np.nanmax(s) - np.nanmin(s)) if len(s) else 0.0),
        n=("abs_error_mm", "size"),
    ).dropna()
    if stats.empty:
        return str(pred_df["STR_name"].iloc[0])
    q25, q75 = stats["mae"].quantile([0.25, 0.75])
    pool = stats[(stats["mae"] >= q25) & (stats["mae"] <= q75)].copy()
    if pool.empty:
        pool = stats.copy()
    if prefer_reversible and pool["rev_amp"].max() > 1e-9:
        return str(pool.sort_values(["rev_amp", "n"], ascending=[False, False]).index[0])
    median_mae = float(stats["mae"].median())
    pool["mae_distance"] = np.abs(pool["mae"] - median_mae)
    return str(pool.sort_values(["mae_distance", "n"], ascending=[True, False]).index[0])


def plot_uncertainty_curve(pred_df: pd.DataFrame, fig_dir: Path) -> None:
    # 选择误差中等的代表性结构，避免挑最好或最差。
    sid = _select_balanced_structure(pred_df, prefer_reversible=False)
    g = pred_df[pred_df["STR_name"] == sid].sort_values("Cum_Load_10k")

    plt.figure(figsize=(9, 5), dpi=300)
    x = g["Cum_Load_10k"].values
    lower = g["lower95_mm_clipped"].values if "lower95_mm_clipped" in g.columns else np.maximum(g["lower95_mm"].values, 0.0)
    upper = g["upper95_mm"].values
    plt.plot(x, g["y_true_mm"].values, marker="o", linewidth=1.8, label="Measured")
    plt.plot(x, g["y_pred_mm"].values, marker="s", linewidth=1.8, linestyle="--", label="Predicted mean")
    plt.fill_between(x, lower, upper, alpha=0.22, label="95% prediction interval")
    plt.xlabel("Cumulative equivalent axle loads (10k)")
    plt.ylabel("Rutting depth (mm)")
    plt.title(f"Rutting evolution with uncertainty: {sid}")
    plt.legend()
    save_paper_figure(fig_dir, "fig2_uncertainty_curve")

def plot_component_decomposition(pred_df: pd.DataFrame, fig_dir: Path) -> None:
    # 用可逆响应较清楚、但误差不过分极端的结构展示双状态分解。
    sid = _select_balanced_structure(pred_df, prefer_reversible=True)
    g = pred_df[pred_df["STR_name"] == sid].sort_values("Cum_Load_10k")

    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
    x = g["Cum_Load_10k"].values
    l1, = ax1.plot(x, g["y_pred_mm"].values, linewidth=1.8, label="Predicted rutting")
    l2, = ax1.plot(x, g["permanent_damage_mm"].values, linewidth=1.8, linestyle="--", label="Permanent damage state")
    ax1.set_xlabel("Cumulative equivalent axle loads (10k)")
    ax1.set_ylabel("Rutting / permanent state (mm)")

    ax2 = ax1.twinx()
    l3, = ax2.plot(x, g["reversible_response_mm"].values, linewidth=1.5, linestyle=":", label="Reversible thermal response")
    ax2.axhline(0, linewidth=1.0, linestyle="--", alpha=0.5)
    ax2.set_ylabel("Reversible response (mm)")

    lines = [l1, l2, l3]
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    plt.title(f"Dual-state decomposition: {sid}")
    save_paper_figure(fig_dir, "fig3_dual_state_decomposition")

def plot_structure_error(by_struct: pd.DataFrame, fig_dir: Path) -> None:
    df = by_struct.sort_values("STR_name", key=lambda s: s.str.replace("STR", "", regex=False).astype(int))
    plt.figure(figsize=(11, 4.8), dpi=300)
    plt.bar(df["STR_name"].values, df["MAE_mm"].values)
    plt.xlabel("Pavement structure")
    plt.ylabel("MAE (mm)")
    plt.title("Prediction error by unseen structure")
    plt.xticks(rotation=45)
    save_paper_figure(fig_dir, "fig4_structure_mae")


def plot_risk_heatmap_like(pred_df: pd.DataFrame, fig_dir: Path) -> None:
    # 不使用 seaborn，生成结构-荷载阶段的平均风险矩阵。
    tmp = pred_df.copy()
    try:
        tmp["Load_Bin"] = pd.qcut(tmp["Cum_Load_10k"], q=8, duplicates="drop")
    except Exception:
        return
    pivot = tmp.pivot_table(index="STR_name", columns="Load_Bin", values="risk_prob", aggfunc="mean")
    pivot = pivot.sort_index(key=lambda s: s.str.replace("STR", "", regex=False).astype(int))

    plt.figure(figsize=(11, 6), dpi=300)
    im = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(im, label="Mean exceedance risk probability")
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.xticks(np.arange(len(pivot.columns)), [f"{int(c.left)}-{int(c.right)}" for c in pivot.columns], rotation=45, ha="right")
    plt.xlabel("Cumulative load bin (10k)")
    plt.ylabel("Pavement structure")
    plt.title("Maintenance-threshold risk map")
    save_paper_figure(fig_dir, "fig5_risk_map")



def configure_paper_plot_style() -> None:
    """统一论文图件风格：高分辨率、弱网格、紧凑字体。"""
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_paper_figure(fig_dir: Path, name: str) -> None:
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}.png", bbox_inches="tight")
    plt.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close()


def plot_residual_vs_load(pred_df: pd.DataFrame, fig_dir: Path) -> None:
    df = pred_df.copy().sort_values("Cum_Load_10k")
    df["residual_mm"] = df["y_pred_mm"] - df["y_true_mm"]
    plt.figure(figsize=(8.2, 4.8), dpi=300)
    plt.scatter(df["Cum_Load_10k"], df["residual_mm"], s=12, alpha=0.45)
    plt.axhline(0, linestyle="--", linewidth=1.2)
    try:
        tmp = df.copy()
        tmp["Load_Bin"] = pd.qcut(tmp["Cum_Load_10k"], q=10, duplicates="drop")
        trend = tmp.groupby("Load_Bin", observed=False).agg(
            x=("Cum_Load_10k", "mean"),
            y=("residual_mm", "median"),
        ).dropna()
        plt.plot(trend["x"].values, trend["y"].values, marker="o", linewidth=1.6, label="Median residual by load bin")
        plt.legend(frameon=False)
    except Exception:
        pass
    plt.xlabel("Cumulative equivalent axle loads (10k)")
    plt.ylabel("Prediction residual (predicted - measured, mm)")
    plt.title("Residual diagnostic along loading history")
    save_paper_figure(fig_dir, "fig6_residual_vs_load")


def plot_uncertainty_calibration(pred_df: pd.DataFrame, fig_dir: Path) -> None:
    if "y_std_mm" not in pred_df.columns:
        return
    df = pred_df.copy()
    df["abs_residual_mm"] = np.abs(df["y_pred_mm"] - df["y_true_mm"])
    try:
        df["Unc_Bin"] = pd.qcut(df["y_std_mm"], q=8, duplicates="drop")
        cal = df.groupby("Unc_Bin", observed=False).agg(
            pred_std=("y_std_mm", "mean"),
            abs_res=("abs_residual_mm", "mean"),
            n=("abs_residual_mm", "size"),
        ).dropna()
    except Exception:
        return
    if cal.empty:
        return
    lo = min(float(cal["pred_std"].min()), float(cal["abs_res"].min()))
    hi = max(float(cal["pred_std"].max()), float(cal["abs_res"].max()))
    pad = 0.08 * (hi - lo + 1e-9)
    plt.figure(figsize=(5.6, 5.2), dpi=300)
    plt.scatter(cal["pred_std"], cal["abs_res"], s=np.clip(cal["n"].values, 20, 120), alpha=0.75)
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1.2)
    plt.xlabel("Mean predicted standard deviation (mm)")
    plt.ylabel("Mean absolute residual (mm)")
    plt.title("Uncertainty calibration by predicted-variance bins")
    save_paper_figure(fig_dir, "fig7_uncertainty_calibration")


def plot_material_attention_heatmap(pred_df: pd.DataFrame, fig_dir: Path) -> None:
    attn_cols = [f"mat_attn_{c}" for c in SEMANTIC_COLS if f"mat_attn_{c}" in pred_df.columns]
    if not attn_cols:
        return
    tmp = pred_df.copy()
    mat = tmp.groupby("Struct_Class")[attn_cols].mean()
    # 删除几乎全 0 的材料列，让论文图更紧凑。
    mat = mat.loc[:, mat.sum(axis=0) > 1e-8]
    if mat.empty:
        return
    labels = [c.replace("mat_attn_", "") for c in mat.columns]
    plt.figure(figsize=(max(8.0, 0.42 * len(labels)), 4.8), dpi=300)
    im = plt.imshow(mat.values, aspect="auto")
    plt.colorbar(im, label="Mean material attention")
    plt.yticks(np.arange(len(mat.index)), mat.index)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.xlabel("Material-layer token")
    plt.ylabel("Structure class")
    plt.title("Learned material-layer attention across pavement classes")
    save_paper_figure(fig_dir, "fig8_material_attention_heatmap")


def plot_temporal_attention_profile(pred_df: pd.DataFrame, fig_dir: Path) -> None:
    time_cols = [c for c in pred_df.columns if c.startswith("time_attn_tminus_")]
    if not time_cols:
        return
    def lag_num(c: str) -> int:
        return int(c.split("_")[-1])
    time_cols = sorted(time_cols, key=lag_num, reverse=True)
    means = pred_df[time_cols].mean().values
    x_labels = ["t" if lag_num(c) == 0 else f"t-{lag_num(c)}" for c in time_cols]
    plt.figure(figsize=(6.8, 4.2), dpi=300)
    plt.bar(x_labels, means)
    plt.xlabel("Time step in input window")
    plt.ylabel("Mean temporal attention")
    plt.title("Temporal attention profile of the sequence encoder")
    save_paper_figure(fig_dir, "fig9_temporal_attention_profile")


def plot_class_stage_error_heatmap(pred_df: pd.DataFrame, fig_dir: Path) -> None:
    tmp = pred_df.copy()
    try:
        tmp["Load_Stage_Plot"] = pd.qcut(tmp["Cum_Load_10k"], q=4, labels=["Early", "Middle", "Late", "Ultra-late"], duplicates="drop")
    except Exception:
        return
    pivot = tmp.pivot_table(index="Struct_Class", columns="Load_Stage_Plot", values="abs_error_mm", aggfunc="mean")
    if pivot.empty:
        return
    plt.figure(figsize=(7.5, 4.8), dpi=300)
    im = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(im, label="Mean absolute error (mm)")
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=0)
    plt.xlabel("Loading stage")
    plt.ylabel("Structure class")
    plt.title("Error landscape by structure class and loading stage")
    save_paper_figure(fig_dir, "fig10_class_stage_error_heatmap")


def plot_threshold_time_error_figure(pred_df: pd.DataFrame, fig_dir: Path, thresholds: Optional[List[float]] = None) -> None:
    thresholds = thresholds or [8.0, 10.0, 12.0]
    th = threshold_time_error(pred_df, thresholds)
    ok = th[th["Status"].eq("ok")].copy()
    if ok.empty:
        return
    agg = ok.groupby("Threshold_mm").agg(
        mean_err=("Abs_Time_Error_10k", "mean"),
        std_err=("Abs_Time_Error_10k", "std"),
        n=("Abs_Time_Error_10k", "size"),
    ).reset_index()
    plt.figure(figsize=(6.4, 4.4), dpi=300)
    plt.bar(agg["Threshold_mm"].astype(str), agg["mean_err"], yerr=agg["std_err"].fillna(0.0), capsize=3)
    plt.xlabel("Rutting threshold (mm)")
    plt.ylabel("Absolute arrival-time error (10k axle loads)")
    plt.title("Maintenance threshold arrival-time accuracy")
    save_paper_figure(fig_dir, "fig11_threshold_arrival_error")


def plot_ablation_summary(ablation_summary: pd.DataFrame, fig_dir: Path) -> None:
    if ablation_summary is None or ablation_summary.empty:
        return
    df = ablation_summary.copy()
    if "RMSE_mm_mean" not in df.columns:
        return
    df = df.sort_values("RMSE_mm_mean", ascending=True)
    labels = df["Model"].astype(str).str.replace("TDR-PRSN-", "", regex=False).values
    y = np.arange(len(df))
    plt.figure(figsize=(8.5, max(4.8, 0.42 * len(df))), dpi=300)
    xerr = df["RMSE_mm_std"].fillna(0.0).values if "RMSE_mm_std" in df.columns else None
    plt.barh(y, df["RMSE_mm_mean"].values, xerr=xerr, capsize=3)
    plt.yticks(y, labels)
    plt.xlabel("Cross-structure RMSE (mm)")
    plt.ylabel("Model variant")
    plt.title("Ablation study: contribution of each physics-guided module")
    save_paper_figure(fig_dir, "fig12_ablation_rmse")

    if "Delta_RMSE_vs_Full_mm" in df.columns:
        d2 = df[~df["Model"].eq("TDR-PRSN-Full")].sort_values("Delta_RMSE_vs_Full_mm", ascending=True)
        if not d2.empty:
            labels2 = d2["Model"].astype(str).values
            y2 = np.arange(len(d2))
            plt.figure(figsize=(8.5, max(4.8, 0.42 * len(d2))), dpi=300)
            vals = d2["Delta_RMSE_vs_Full_mm"].values
            plt.barh(y2, vals)
            plt.axvline(0, linestyle="--", linewidth=1.0)
            plt.yticks(y2, labels2)
            plt.xlabel("Delta RMSE after module removal (mm)")
            plt.ylabel("Ablation variant")
            plt.title("Change in RMSE after removing each module")
            for yy, vv in zip(y2, vals):
                ha = "left" if vv >= 0 else "right"
                offset = 0.01 if vv >= 0 else -0.01
                plt.text(vv + offset, yy, f"{vv:+.3f}", va="center", ha=ha, fontsize=8)
            save_paper_figure(fig_dir, "fig13_ablation_degradation")



def _model_order_for_comparison(models: List[str]) -> List[str]:
    priority = [
        "Persistence_prev_y",
        "XGBoost_flat_sequence",
        "XGBoost_last_step_state",
        "XGBoost_sequence_state",
        "TDR-PRSN-Full",
        "TDR-PRSN-RC",
    ]
    ordered = [m for m in priority if m in set(models)]
    ordered += [m for m in models if m not in set(ordered)]
    return ordered


def _safe_fig_name(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_").replace("+", "plus").replace(".", "_")


def plot_baseline_parity_grid(all_pred_df: Optional[pd.DataFrame], fig_dir: Path) -> None:
    """各模型实测-预测一致性子图，便于论文中公平比较基线模型与 TDR-PRSN。"""
    if all_pred_df is None or all_pred_df.empty or "Model" not in all_pred_df.columns:
        return
    df = all_pred_df.dropna(subset=["y_true_mm", "y_pred_mm"]).copy()
    if df.empty:
        return
    models = _model_order_for_comparison(sorted(df["Model"].dropna().astype(str).unique().tolist()))
    models = [m for m in models if m in set(df["Model"].astype(str))]
    if not models:
        return
    n = len(models)
    ncols = 3 if n >= 3 else n
    nrows = int(math.ceil(n / ncols))
    lo = min(float(df["y_true_mm"].min()), float(df["y_pred_mm"].min()))
    hi = max(float(df["y_true_mm"].max()), float(df["y_pred_mm"].max()))
    pad = 0.05 * (hi - lo + 1e-9)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.9 * nrows), dpi=300)
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, model in zip(axes, models):
        g = df[df["Model"].astype(str).eq(model)]
        ax.scatter(g["y_true_mm"], g["y_pred_mm"], s=10, alpha=0.42)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1.1)
        m = calc_metrics(g["y_true_mm"].values, g["y_pred_mm"].values)
        ax.set_title(f"{model}\nR2={m['R2']:.3f}, RMSE={m['RMSE_mm']:.3f}, MAE={m['MAE_mm']:.3f}")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("Measured rutting depth (mm)")
        ax.set_ylabel("Predicted rutting depth (mm)")
    for ax in axes[len(models):]:
        ax.axis("off")
    fig.suptitle("Baseline comparison: predicted vs. measured rutting depth", y=1.02)
    save_paper_figure(fig_dir, "fig14_baseline_parity_grid")

    # 单模型图也保存，便于后续排版自由组合。
    for model in models:
        g = df[df["Model"].astype(str).eq(model)]
        plt.figure(figsize=(6.0, 5.6), dpi=300)
        plt.scatter(g["y_true_mm"], g["y_pred_mm"], s=12, alpha=0.45)
        plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1.3)
        m = calc_metrics(g["y_true_mm"].values, g["y_pred_mm"].values)
        plt.xlim(lo - pad, hi + pad)
        plt.ylim(lo - pad, hi + pad)
        plt.xlabel("Measured rutting depth (mm)")
        plt.ylabel("Predicted rutting depth (mm)")
        plt.title(f"{model}: parity plot\nR2={m['R2']:.3f}, RMSE={m['RMSE_mm']:.3f} mm, MAE={m['MAE_mm']:.3f} mm")
        save_paper_figure(fig_dir, f"fig14_parity_{_safe_fig_name(model)}")


def _select_structure_for_model_comparison(all_pred_df: pd.DataFrame) -> str:
    if "STR1" in set(all_pred_df["STR_name"].astype(str)):
        return "STR1"
    try:
        counts = all_pred_df.groupby("STR_name").size().sort_values(ascending=False)
        return str(counts.index[0])
    except Exception:
        return str(all_pred_df["STR_name"].iloc[0])


def plot_structure_curve_compare(all_pred_df: Optional[pd.DataFrame], fig_dir: Path) -> None:
    """代表结构演化曲线：实测值与主要基线模型同图比较。"""
    if all_pred_df is None or all_pred_df.empty or "Model" not in all_pred_df.columns:
        return
    df = all_pred_df.dropna(subset=["STR_name", "Cum_Load_10k", "y_true_mm", "y_pred_mm"]).copy()
    if df.empty:
        return
    sid = _select_structure_for_model_comparison(df)
    g_all = df[df["STR_name"].astype(str).eq(sid)].copy()
    if g_all.empty:
        return
    model_priority = ["Persistence_prev_y", "XGBoost_sequence_state", "TDR-PRSN-Full", "TDR-PRSN-RC"]
    models = [m for m in model_priority if m in set(g_all["Model"].astype(str))]
    if not models:
        models = _model_order_for_comparison(sorted(g_all["Model"].astype(str).unique().tolist()))[:5]
    truth = g_all.sort_values("Cum_Load_10k").drop_duplicates(["STR_name", "Cum_Load_10k", "Observation_Order"])
    plt.figure(figsize=(9.4, 5.2), dpi=300)
    plt.plot(truth["Cum_Load_10k"], truth["y_true_mm"], marker="o", linewidth=2.0, label="Measured", zorder=5)
    for model in models:
        g = g_all[g_all["Model"].astype(str).eq(model)].sort_values("Cum_Load_10k")
        plt.plot(g["Cum_Load_10k"], g["y_pred_mm"], marker=".", linewidth=1.5, label=model)
        if model in {"TDR-PRSN-Full", "TDR-PRSN-RC"} and "lower95_mm_clipped" in g.columns and "upper95_mm" in g.columns:
            gg = g.dropna(subset=["lower95_mm_clipped", "upper95_mm"])
            if not gg.empty:
                plt.fill_between(gg["Cum_Load_10k"].values, gg["lower95_mm_clipped"].values, gg["upper95_mm"].values, alpha=0.10)
    plt.xlabel("Cumulative equivalent axle loads (10k)")
    plt.ylabel("Rutting depth (mm)")
    plt.title(f"Representative structure evolution comparison ({sid})")
    plt.legend(frameon=False, ncol=2)
    save_paper_figure(fig_dir, "fig15_structure_curve_model_comparison")


def plot_residual_boxplot_models(all_pred_df: Optional[pd.DataFrame], fig_dir: Path) -> None:
    """基线模型残差箱线图，比较误差分布稳定性和离群程度。"""
    if all_pred_df is None or all_pred_df.empty or "Model" not in all_pred_df.columns:
        return
    df = all_pred_df.dropna(subset=["y_true_mm", "y_pred_mm"]).copy()
    if df.empty:
        return
    df["residual_mm"] = df["y_pred_mm"] - df["y_true_mm"]
    models = _model_order_for_comparison(sorted(df["Model"].astype(str).unique().tolist()))
    data = [df.loc[df["Model"].astype(str).eq(m), "residual_mm"].values for m in models]
    plt.figure(figsize=(max(8.2, 1.15 * len(models)), 5.0), dpi=300)
    plt.boxplot(data, labels=models, showfliers=False)
    plt.axhline(0, linestyle="--", linewidth=1.1)
    plt.xticks(rotation=28, ha="right")
    plt.ylabel("Prediction residual (predicted - measured, mm)")
    plt.title("Residual distribution comparison among baseline and proposed models")
    save_paper_figure(fig_dir, "fig16_residual_boxplot_models")


def plot_metrics_bar_models(all_pred_df: Optional[pd.DataFrame], fig_dir: Path) -> None:
    """各模型 R2、RMSE、MAE 基础指标柱状图。"""
    if all_pred_df is None or all_pred_df.empty or "Model" not in all_pred_df.columns:
        return
    rows = []
    for model, g in all_pred_df.dropna(subset=["y_true_mm", "y_pred_mm"]).groupby("Model"):
        m = calc_metrics(g["y_true_mm"].values, g["y_pred_mm"].values)
        m["Model"] = str(model)
        rows.append(m)
    if not rows:
        return
    df = pd.DataFrame(rows)
    models = _model_order_for_comparison(df["Model"].tolist())
    df = df.set_index("Model").loc[models].reset_index()
    x = np.arange(len(df))
    width = 0.36
    plt.figure(figsize=(max(8.0, 1.1 * len(df)), 4.8), dpi=300)
    plt.bar(x - width / 2, df["RMSE_mm"], width, label="RMSE")
    plt.bar(x + width / 2, df["MAE_mm"], width, label="MAE")
    plt.xticks(x, df["Model"], rotation=28, ha="right")
    plt.ylabel("Error (mm)")
    plt.title("Point-prediction error comparison")
    plt.legend(frameon=False)
    save_paper_figure(fig_dir, "fig17_metrics_bar_rmse_mae")

    plt.figure(figsize=(max(8.0, 1.1 * len(df)), 4.2), dpi=300)
    plt.bar(x, df["R2"])
    plt.xticks(x, df["Model"], rotation=28, ha="right")
    plt.ylabel("R2")
    plt.ylim(max(0.0, float(df["R2"].min()) - 0.05), min(1.02, float(df["R2"].max()) + 0.02))
    plt.title("Coefficient of determination comparison")
    save_paper_figure(fig_dir, "fig18_metrics_bar_r2")


def plot_probability_metric_bars(all_pred_df: Optional[pd.DataFrame], fig_dir: Path) -> None:
    """只对具备 y_std 的概率模型绘制 PICP/MPIW/NLL，对确定性基线不做不公平比较。"""
    if all_pred_df is None or all_pred_df.empty or "Model" not in all_pred_df.columns or "y_std_mm" not in all_pred_df.columns:
        return
    rows = []
    for model, g in all_pred_df.dropna(subset=["y_true_mm", "y_pred_mm", "y_std_mm"]).groupby("Model"):
        if len(g) == 0:
            continue
        m = calc_metrics(g["y_true_mm"].values, g["y_pred_mm"].values, g["y_std_mm"].values)
        if "PICP_95" in m:
            m["Model"] = str(model)
            rows.append(m)
    if not rows:
        return
    df = pd.DataFrame(rows)
    models = _model_order_for_comparison(df["Model"].tolist())
    df = df.set_index("Model").loc[models].reset_index()
    x = np.arange(len(df))
    width = 0.28
    plt.figure(figsize=(max(7.0, 1.3 * len(df)), 4.8), dpi=300)
    plt.bar(x - width, df["PICP_95"], width, label="PICP95")
    plt.bar(x, df["MPIW_95_mm"], width, label="MPIW95")
    plt.bar(x + width, df["NLL_raw_proxy"], width, label="NLL")
    plt.xticks(x, df["Model"], rotation=25, ha="right")
    plt.ylabel("Metric value")
    plt.title("Probabilistic metrics of TDR-PRSN variants")
    plt.legend(frameon=False)
    save_paper_figure(fig_dir, "fig19_probability_metrics_bar")


def _write_main_figure_manifest(fig_dir: Path, figure_mode: str) -> None:
    manifest = {
        "main": [
            ("fig1_parity_plot", "跨结构真实-预测散点图：证明整体点预测精度。"),
            ("fig2_uncertainty_curve", "代表性结构车辙演化及 95% 预测区间：证明概率预测能力。"),
            ("fig3_dual_state_decomposition", "永久损伤状态与可逆温度响应分解：证明物理状态内涵。"),
            ("fig7_uncertainty_calibration", "预测标准差与实际误差校准图：证明不确定性输出有可靠性。"),
            ("fig13_ablation_degradation", "核心模块消融退化图：证明模块贡献。"),
            ("fig14_baseline_parity_grid", "主要基线与TDR-PRSN的实测-预测一致性对比图。"),
            ("fig15_structure_curve_model_comparison", "代表结构演化曲线：实测值、基线模型与TDR-PRSN同图对比。"),
            ("fig16_residual_boxplot_models", "各模型残差箱线图：比较误差分布稳定性。"),
            ("fig17_metrics_bar_rmse_mae", "各模型RMSE/MAE柱状图。"),
        ],
        "supplementary": [
            ("fig4_structure_mae", "不同未见结构的 MAE 诊断。"),
            ("fig5_risk_map", "养护阈值风险热力图。"),
            ("fig6_residual_vs_load", "残差随荷载演化诊断。"),
            ("fig8_material_attention_heatmap", "材料层位注意力热力图。"),
            ("fig9_temporal_attention_profile", "时序注意力分布。"),
            ("fig10_class_stage_error_heatmap", "结构类别-荷载阶段误差图。"),
            ("fig11_threshold_arrival_error", "阈值到达时间误差。"),
            ("fig12_ablation_rmse", "各消融模型 RMSE 对比。"),
        ],
    }
    lines = [f"Figure mode: {figure_mode}", "", "Main-paper figures:"]
    for name, desc in manifest["main"]:
        lines.append(f"- {name}.png/.pdf: {desc}")
    lines.append("")
    lines.append("Supplementary figures generated only when --figure-mode all:")
    for name, desc in manifest["supplementary"]:
        lines.append(f"- {name}.png/.pdf: {desc}")
    (fig_dir / "paper_figure_manifest.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_all(
    pred_df: pd.DataFrame,
    by_struct: pd.DataFrame,
    out_dir: Path,
    ablation_summary: Optional[pd.DataFrame] = None,
    figure_mode: str = "main",
    all_model_predictions: Optional[pd.DataFrame] = None,
) -> None:
    fig_dir = ensure_dir(out_dir / "figures")
    configure_paper_plot_style()
    figure_mode = str(figure_mode).lower().strip()
    if figure_mode not in {"main", "all"}:
        figure_mode = "main"

    # 主文核心图：精度、概率区间、状态分解、概率校准、核心消融。
    plot_parity(pred_df, fig_dir)
    plot_uncertainty_curve(pred_df, fig_dir)
    plot_component_decomposition(pred_df, fig_dir)
    plot_uncertainty_calibration(pred_df, fig_dir)
    plot_ablation_summary(ablation_summary, fig_dir)

    # 基线模型论文对比图：与 Persistence、XGBoost 等公平比较。
    plot_baseline_parity_grid(all_model_predictions, fig_dir)
    plot_structure_curve_compare(all_model_predictions, fig_dir)
    plot_residual_boxplot_models(all_model_predictions, fig_dir)
    plot_metrics_bar_models(all_model_predictions, fig_dir)
    plot_probability_metric_bars(all_model_predictions, fig_dir)

    # 补充材料/诊断图：只在需要完整诊断时输出，避免主文图件过多。
    if figure_mode == "all":
        plot_structure_error(by_struct, fig_dir)
        plot_risk_heatmap_like(pred_df, fig_dir)
        plot_residual_vs_load(pred_df, fig_dir)
        plot_material_attention_heatmap(pred_df, fig_dir)
        plot_temporal_attention_profile(pred_df, fig_dir)
        plot_class_stage_error_heatmap(pred_df, fig_dir)
        plot_threshold_time_error_figure(pred_df, fig_dir)

    _write_main_figure_manifest(fig_dir, figure_mode)
    log(f"[绘图] 图像已保存到: {fig_dir}；当前 figure_mode={figure_mode}")


# =============================================================================
# 9. 入口
# =============================================================================

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="TDR-PRSN full pipeline for full-scale track rutting prediction.")
    parser.add_argument("--input", type=str, required=True, help="原始宽表或处理后长表 Excel 文件路径。")
    parser.add_argument("--output-dir", type=str, default="outputs_tdr_prsn", help="输出目录。")
    parser.add_argument("--rutting-scale-to-mm", type=float, default=0.1, help="原始车辙数值换算为 mm 的比例。原始单位为0.1mm时填0.1；若已是mm填1.0。")
    parser.add_argument("--seq-len", type=int, default=6, help="时序窗口长度。")
    parser.add_argument("--folds", type=int, default=5, help="GroupKFold 折数。")
    parser.add_argument("--epochs", type=int, default=220, help="最大训练轮数。")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小。")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="学习率。")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减。")
    parser.add_argument("--d-model", type=int, default=64, help="隐空间维度。")
    parser.add_argument("--dropout", type=float, default=0.10, help="Dropout。")
    parser.add_argument("--patience", type=int, default=40, help="早停耐心轮数。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--risk-threshold-mm", type=float, default=10.0, help="风险分支训练阈值，单位 mm。")
    parser.add_argument("--threshold-list-mm", type=str, default="8,10,12", help="阈值到达时间误差计算阈值，逗号分隔，单位 mm。")
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda。")
    parser.add_argument("--no-xgb", action="store_true", help="不运行 XGBoost 基线。")
    parser.add_argument("--no-dl-baselines", action="store_true", help="不运行 LSTM/Transformer 深度学习基线。")
    parser.add_argument("--dl-baseline-epochs", type=int, default=160, help="LSTM/Transformer 深度学习基线最大训练轮数。")
    parser.add_argument("--no-excel", action="store_true", help="不输出 Excel 汇总，只输出 CSV。")
    parser.add_argument("--no-ablation", action="store_true", help="不运行结构化消融试验。")
    parser.add_argument("--ablation-epochs", type=int, default=120, help="每个消融变体的最大训练轮数。")
    parser.add_argument("--ablation-variants", type=str, default="A2_flat_thickness,A4_no_thermal_gate,A5_no_reversible,A7_no_obs_adapter", help="要运行的消融变体，逗号分隔。默认只保留论文主文最核心的4组消融；如需完整消融，可手动传入A1-A8。")
    parser.add_argument("--val-every", type=int, default=5, help="每隔多少个 epoch 做一次验证；设为1则恢复每轮验证。")
    parser.add_argument("--figure-mode", type=str, default="main", choices=["main", "all"], help="main只输出主文核心图；all额外输出补充诊断图。")
    parser.add_argument("--delta-loss-weight", type=float, default=0.25, help="车辙增量监督损失权重；设为0可关闭。")
    parser.add_argument("--delta-loss-scale-mm", type=float, default=8.0, help="增量监督的尺度归一化参数，单位mm。")
    parser.add_argument("--dual-state-loss-weight", type=float, default=0.05, help="双状态均值辅助监督权重；用于让永久+可逆状态直接解释观测车辙。")
    parser.add_argument("--reversible-aux-weight", type=float, default=0.03, help="可逆响应软目标监督权重；冷启动过滤后建议保留小权重。")
    parser.add_argument("--reversible-reg-weight", type=float, default=5e-4, help="可逆响应幅值正则权重；过大会压制A5贡献。")
    parser.add_argument("--min-physics-fusion-gate", type=float, default=0.15, help="观测适配中物理双状态路径的最低融合比例。")
    parser.add_argument("--no-residual-calibration", action="store_true", help="关闭 TDR-PRSN-RC 跨结构残差校正模型。")
    parser.add_argument("--residual-calibrator-strength", type=float, default=0.75, help="残差校正强度，0表示不修正，1表示完全采用校正器残差。")
    parser.add_argument("--residual-calibrator-estimators", type=int, default=260, help="残差校正器 XGBoost 树数量。")
    parser.add_argument("--no-rc-for-main-outputs", action="store_true", help="主文图仍使用未校正的 TDR-PRSN-Full，而不是 TDR-PRSN-RC。")
    parser.add_argument("--split-mode", type=str, default="material_aware", choices=["material_aware", "groupkfold"], help="跨结构验证划分方式。material_aware保证验证集材料均在训练集中出现；groupkfold为原始无约束结构GroupKFold。")
    parser.add_argument("--material-anchor-structures", type=str, default="auto", help="材料锚点结构。auto表示自动把含唯一出现材料的结构固定在训练集中；也可传入逗号分隔STR名称；none表示不手动锚定但仍执行覆盖修正。")
    parser.add_argument("--no-split-report", action="store_true", help="不输出材料覆盖划分报告。")
    parser.add_argument("--include-initial-observation-targets", action="store_true", help="将每个结构的首个观测点也纳入训练/评价。默认不纳入，以避免无上一期实测状态导致的冷启动低估。")
    args = parser.parse_args()

    return Config(
        input=args.input,
        output_dir=args.output_dir,
        rutting_scale_to_mm=args.rutting_scale_to_mm,
        seq_len=args.seq_len,
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        dropout=args.dropout,
        patience=args.patience,
        seed=args.seed,
        risk_threshold_mm=args.risk_threshold_mm,
        threshold_list_mm=args.threshold_list_mm,
        run_xgb=not args.no_xgb,
        run_dl_baselines=not args.no_dl_baselines,
        dl_baseline_epochs=args.dl_baseline_epochs,
        device=args.device,
        save_excel=not args.no_excel,
        run_ablation=not args.no_ablation,
        ablation_epochs=args.ablation_epochs,
        ablation_variants=args.ablation_variants,
        val_every=args.val_every,
        figure_mode=args.figure_mode,
        delta_loss_weight=args.delta_loss_weight,
        delta_loss_scale_mm=args.delta_loss_scale_mm,
        dual_state_loss_weight=args.dual_state_loss_weight,
        reversible_aux_weight=args.reversible_aux_weight,
        reversible_reg_weight=args.reversible_reg_weight,
        min_physics_fusion_gate=args.min_physics_fusion_gate,
        enable_residual_calibration=not args.no_residual_calibration,
        residual_calibrator_strength=args.residual_calibrator_strength,
        residual_calibrator_estimators=args.residual_calibrator_estimators,
        use_rc_for_main_outputs=not args.no_rc_for_main_outputs,
        split_mode=args.split_mode,
        material_anchor_structures=args.material_anchor_structures,
        save_split_report=not args.no_split_report,
        exclude_initial_observation_targets=not args.include_initial_observation_targets,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    out_dir = ensure_dir(cfg.output_dir)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    start = time.time()
    log("\n========== TDR-PRSN 全流程启动 ==========")
    log(f"输入文件: {cfg.input}")
    log(f"输出目录: {out_dir.resolve()}")

    panel_df = load_and_prepare_dataset(cfg.input, cfg.rutting_scale_to_mm)
    audit_panel_dataset(panel_df, cfg, out_dir)

    panel_path_csv = out_dir / "rutting_panel_dataset_v2.csv"
    panel_df.to_csv(panel_path_csv, index=False, encoding="utf-8-sig")
    if cfg.save_excel:
        try:
            panel_df.to_excel(out_dir / "rutting_panel_dataset_v2.xlsx", index=False)
        except Exception as e:
            log(f"[提醒] 面板数据 Excel 写入失败，CSV 已保存。原因: {e}")
    log(f"[数据] 面板数据已保存: {panel_path_csv}")

    pred_df, metrics_df, by_struct, all_model_predictions = run_cross_structure_training(panel_df, cfg, out_dir)
    ablation_summary, ablation_metrics, ablation_preds = run_ablation_study(panel_df, metrics_df, cfg, out_dir)
    plot_all(pred_df, by_struct, out_dir, ablation_summary=ablation_summary, figure_mode=cfg.figure_mode, all_model_predictions=all_model_predictions)

    log("\n========== 交叉验证汇总 ==========")
    summary = metrics_df.groupby("Model", dropna=False)[["R2", "RMSE_mm", "MAE_mm"]].mean(numeric_only=True)
    if "PICP_95" in metrics_df.columns:
        extra_cols = [c for c in ["PICP_95", "MPIW_95_mm", "NLL_raw_proxy"] if c in metrics_df.columns]
        extra = metrics_df.groupby("Model", dropna=False)[extra_cols].mean(numeric_only=True)
        summary = summary.join(extra, how="left")
    if "Persistence_prev_y" in summary.index:
        base_rmse = float(summary.loc["Persistence_prev_y", "RMSE_mm"])
        base_mae = float(summary.loc["Persistence_prev_y", "MAE_mm"])
        summary["RMSE_reduction_vs_Persistence_%"] = 100.0 * (base_rmse - summary["RMSE_mm"]) / max(abs(base_rmse), 1e-12)
        summary["MAE_reduction_vs_Persistence_%"] = 100.0 * (base_mae - summary["MAE_mm"]) / max(abs(base_mae), 1e-12)
    summary.to_csv(out_dir / "paper_table_main_metrics.csv", encoding="utf-8-sig")
    if ablation_summary is not None and not ablation_summary.empty:
        ablation_summary.to_csv(out_dir / "paper_table_ablation_core.csv", index=False, encoding="utf-8-sig")
    log(summary.to_string())
    log("==================================")

    elapsed = time.time() - start
    log(f"\n[完成] 总耗时: {elapsed:.1f} 秒。主要输出位于: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
