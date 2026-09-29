from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from v31_downstream_data import build_real_downstream_frame, build_sequence_arrays_simple
from v40_metrics import summarize_prediction_rows
from v40_openloop import EvalArrays, forecast_all_origins
from v41_fixed_origin import forecast_end2020_fixed_origin
from v43_protocol import V43Fold, V43RunKey, V43Protocol


def _heldout_master(master: pd.DataFrame, fold: V43Fold) -> pd.DataFrame:
    out = master.loc[master["STR_name"].astype(str).isin(fold.held_out_structures)].copy()
    if out.empty:
        raise ValueError(f"fold {fold.fold_id} held-out master is empty")
    return out


def build_ood_eval_arrays(master: pd.DataFrame, fold: V43Fold, seq_len: int = 6) -> EvalArrays:
    held = _heldout_master(master, fold)
    frame = build_real_downstream_frame(held)
    X, y, prev, peak, meta = build_sequence_arrays_simple(frame, seq_len=int(seq_len))
    years = pd.to_numeric(meta["Year"], errors="coerce")
    mask = years.isin([2020, 2021]).to_numpy()
    meta = meta.loc[mask].reset_index(drop=True)
    out = EvalArrays(
        np.asarray(X[mask], np.float32),
        np.asarray(y[mask], np.float32),
        np.asarray(prev[mask], np.float32),
        np.asarray(peak[mask], np.float32),
        meta,
    )
    if len(out.y) == 0:
        raise ValueError(f"fold {fold.fold_id} has no held-out 2020-2021 targets")
    predicted_structures = set(out.meta["STR_name"].astype(str))
    if not predicted_structures.issubset(set(fold.held_out_structures)):
        raise RuntimeError("non-heldout structure entered Stage 13 OOD evaluation arrays")
    if not set(pd.to_numeric(out.meta["Year"], errors="coerce").dropna().astype(int)).issubset({2020, 2021}):
        raise RuntimeError("Stage 13 OOD rolling arrays contain targets outside 2020-2021")
    return out


def _annotate_v43(frame: pd.DataFrame, fold: V43Fold, run_key: V43RunKey) -> pd.DataFrame:
    out = frame.copy()
    values = [
        ("fold_id", int(fold.fold_id)),
        ("group_name", str(fold.group_name)),
        ("seed", int(run_key.seed)),
        ("allocation", int(run_key.allocation)),
        ("synthetic_fraction", float(run_key.q)),
        ("method", f"v43|{run_key.condition}"),
        ("model", "tdr"),
        ("condition", str(run_key.condition)),
    ]
    for column, value in reversed(values):
        if column in out.columns:
            out[column] = value
        else:
            out.insert(0, column, value)
    return out


def _add_target_year(frame: pd.DataFrame, held_master: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    lookup = held_master[["STR_name", "Observation_Index", "Year"]].drop_duplicates().rename(
        columns={"Observation_Index": "target_observation_index", "Year": "target_year"}
    )
    out = frame.merge(lookup, on=["STR_name", "target_observation_index"], how="left", validate="many_to_one")
    if pd.to_numeric(out["target_year"], errors="coerce").isna().any():
        raise RuntimeError("could not map Stage 13 prediction target year")
    return out


def forecast_ood_horizon(
    fit,
    master: pd.DataFrame,
    fold: V43Fold,
    horizon: int,
    run_key: V43RunKey,
    predictor=None,
    seq_len: int | None = None,
) -> pd.DataFrame:
    p = V43Protocol()
    if int(horizon) not in set(p.horizons):
        raise ValueError(f"Stage 13 horizon must be one of {p.horizons}")
    arrays = build_ood_eval_arrays(master, fold, seq_len=int(seq_len or p.seq_len))
    rows = forecast_all_origins(fit, arrays, int(horizon), "tdr", predictor=predictor)
    if rows.empty:
        return rows
    held = _heldout_master(master, fold)
    rows = _add_target_year(rows, held)
    if not pd.to_numeric(rows["target_year"], errors="coerce").isin([2020, 2021]).all():
        raise RuntimeError("Stage 13 rolling OOD prediction target escaped 2020-2021")
    rows = _annotate_v43(rows, fold, run_key)
    return rows


def forecast_ood_fixed_origin(
    fit,
    master: pd.DataFrame,
    fold: V43Fold,
    run_key: V43RunKey,
    predictor=None,
    seq_len: int | None = None,
) -> pd.DataFrame:
    p = V43Protocol()
    held = _heldout_master(master, fold)
    rows = forecast_end2020_fixed_origin(
        fit,
        held,
        "tdr",
        predictor=predictor,
        seq_len=int(seq_len or p.seq_len),
    )
    if rows.empty:
        return rows
    rows = _add_target_year(rows, held)
    if not pd.to_numeric(rows["target_year"], errors="coerce").eq(p.chronological_evaluation_year).all():
        raise RuntimeError("Stage 13 fixed-origin predictions must target 2021 only")
    rows = _annotate_v43(rows, fold, run_key)
    return rows


def _safe_r2(y, pred):
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    return float(r2_score(y, pred)) if len(y) >= 2 and np.std(y) > 0 else np.nan


def _fixed_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    group_cols = [c for c in ["fold_id","group_name","model","condition","synthetic_fraction","allocation","seed"] if c in predictions.columns]
    rows=[]
    for keys,g in predictions.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple): keys=(keys,)
        rec=dict(zip(group_cols,keys))
        y=g["y_true"].to_numpy(float); pred=g["y_pred"].to_numpy(float); per=g["persistence_pred"].to_numpy(float)
        rmse=float(np.sqrt(mean_squared_error(y,pred))); prmse=float(np.sqrt(mean_squared_error(y,per)))
        rec.update({
            "rmse":rmse,
            "mae":float(mean_absolute_error(y,pred)),
            "r2":_safe_r2(y,pred),
            "persistence_rmse":prmse,
            "persistence_skill":float(1-rmse/prmse) if prmse>0 else np.nan,
            "n_rows":int(len(g)),
            "n_structures":int(g["STR_name"].nunique()),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_ood_predictions(horizon_predictions: pd.DataFrame, fixed_origin_predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if horizon_predictions.empty:
        horizon_metrics = pd.DataFrame()
        step_metrics = pd.DataFrame()
    else:
        s = summarize_prediction_rows(horizon_predictions)
        horizon_metrics = s["horizon_metrics"]
        step_metrics = s["step_metrics"]
        # v40 summary grouping omits fold/condition because they are not generic run cols;
        # each Stage-13 call is a single fold/run, so annotate them from the source frame.
        for table in (horizon_metrics, step_metrics):
            for col in ("fold_id", "group_name", "condition"):
                if col not in table.columns:
                    table.insert(0, col, horizon_predictions[col].iloc[0])
    return {
        "horizon_metrics": horizon_metrics,
        "step_metrics": step_metrics,
        "fixed_origin_metrics": _fixed_metrics(fixed_origin_predictions),
    }
