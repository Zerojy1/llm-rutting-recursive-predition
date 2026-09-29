from __future__ import annotations

from pathlib import Path
from typing import Any
import copy
import json
import pandas as pd

from v31_downstream_data import FEATURE_COLS
from v31_qwen_data import build_chronological_qwen_datasets
from v43_protocol import V43Fold


IDENTITY_KEYS = {"STR_name", "STR_num", "Struct_ID", "Struct_Class"}


def split_master_for_fold(master: pd.DataFrame, fold: V43Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "STR_name" not in master.columns:
        raise KeyError("master missing STR_name")
    names = master["STR_name"].astype(str)
    held = set(fold.held_out_structures)
    train = master.loc[~names.isin(held)].copy().reset_index(drop=True)
    test = master.loc[names.isin(held)].copy().reset_index(drop=True)
    if train.empty:
        raise ValueError(f"fold {fold.fold_id} training master is empty")
    if test.empty:
        raise ValueError(f"fold {fold.fold_id} held-out master is empty")
    return train, test


def audit_identity_free_features() -> dict:
    features = list(FEATURE_COLS)
    forbidden = sorted(IDENTITY_KEYS.intersection(features))
    return {
        "pass": not forbidden,
        "feature_cols": features,
        "forbidden_identity_features": forbidden,
        "note": "STR identity is permitted only as metadata/grouping, not as a numerical model feature.",
    }


def _strip_identity_recursive(value: Any):
    if isinstance(value, dict):
        return {k: _strip_identity_recursive(v) for k, v in value.items() if k not in IDENTITY_KEYS}
    if isinstance(value, list):
        return [_strip_identity_recursive(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_strip_identity_recursive(v) for v in value)
    return copy.deepcopy(value)


def strip_identity_from_condition(condition: dict) -> dict:
    return _strip_identity_recursive(condition)


def build_fold_qwen_datasets(
    master: pd.DataFrame,
    fold: V43Fold,
    out_dir: str | Path,
    history_steps: int = 4,
    future_steps: int = 6,
) -> dict[str, pd.DataFrame]:
    train_master, _ = split_master_for_fold(master, fold)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = build_chronological_qwen_datasets(
        train_master,
        out,
        history_steps=int(history_steps),
        future_steps=int(future_steps),
    )
    held = set(fold.held_out_structures)
    for name, frame in frames.items():
        if len(frame) and set(frame["STR_name"].astype(str)).intersection(held):
            raise RuntimeError(f"held-out structure leaked into Qwen {name} fold dataset")
    manifest_path = out / "v43_fold_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "fold_id": fold.fold_id,
                "group_name": fold.group_name,
                "held_out_structures": list(fold.held_out_structures),
                "train_structures": list(fold.train_structures),
                "identity_free_model_features": audit_identity_free_features(),
                "counts": {k: int(len(v)) for k, v in frames.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return frames


def _structures_from_artifact(frame: pd.DataFrame | None) -> set[str]:
    if frame is None or frame.empty or "STR_name" not in frame.columns:
        return set()
    return set(frame["STR_name"].dropna().astype(str))


def audit_fold_isolation(
    fold: V43Fold,
    train_master: pd.DataFrame | None = None,
    qwen_frames: dict[str, pd.DataFrame] | None = None,
    synthetic_update: pd.DataFrame | None = None,
    downstream_train_meta: pd.DataFrame | None = None,
    downstream_dev_meta: pd.DataFrame | None = None,
    test_predictions: pd.DataFrame | None = None,
) -> dict:
    held = set(fold.held_out_structures)
    leaks: dict[str, list[str]] = {}

    def check(name: str, structures: set[str]):
        bad = sorted(held.intersection(structures))
        if bad:
            leaks[name] = bad

    check("train_master", _structures_from_artifact(train_master))
    for name, frame in (qwen_frames or {}).items():
        check(f"qwen_{name}", _structures_from_artifact(frame))
    check("synthetic_update", _structures_from_artifact(synthetic_update))
    check("downstream_train_meta", _structures_from_artifact(downstream_train_meta))
    check("downstream_dev_meta", _structures_from_artifact(downstream_dev_meta))

    prediction_error = None
    if test_predictions is not None and len(test_predictions):
        predicted = _structures_from_artifact(test_predictions)
        unexpected = sorted(predicted - held)
        missing = sorted(held - predicted)
        if unexpected:
            prediction_error = {"unexpected_nonheldout": unexpected}
        # Some groups can contain a structure without enough test points for H=12;
        # absence is reported but does not alone make isolation fail.
        if missing:
            prediction_error = {**(prediction_error or {}), "heldout_without_predictions": missing}

    feature_audit = audit_identity_free_features()
    passed = not leaks and feature_audit["pass"] and not (prediction_error and "unexpected_nonheldout" in prediction_error)
    return {
        "pass": bool(passed),
        "fold_id": fold.fold_id,
        "group_name": fold.group_name,
        "held_out_structures": list(fold.held_out_structures),
        "train_structures": list(fold.train_structures),
        "leaks": leaks,
        "test_prediction_audit": prediction_error or {},
        "identity_feature_audit": feature_audit,
    }
