from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from v31_statistics import hierarchical_cluster_relative_bootstrap
from v40_metrics import PAIR_KEYS


V41_RUN_COLS = [
    "model",
    "condition",
    "real_fraction",
    "synthetic_fraction",
    "allocation",
    "seed",
]


def _required(frame: pd.DataFrame, columns) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")


def _trajectory_keys(frame: pd.DataFrame) -> list[str]:
    base = [column for column in V41_RUN_COLS if column in frame.columns]
    return base + ["STR_name", "origin_observation_index", "requested_horizon"]


def add_increment_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    _required(
        predictions,
        [
            "STR_name",
            "origin_observation_index",
            "requested_horizon",
            "step_h",
            "y_origin",
            "y_true",
            "y_pred",
        ],
    )
    out = predictions.copy()
    keys = _trajectory_keys(out)
    out = out.sort_values(keys + ["step_h"], kind="mergesort").reset_index(drop=True)
    grouped = out.groupby(keys, dropna=False, sort=False)
    previous_true = grouped["y_true"].shift(1)
    previous_predicted = grouped["y_pred"].shift(1)
    first = grouped.cumcount().eq(0)
    previous_true.loc[first] = pd.to_numeric(out.loc[first, "y_origin"], errors="coerce")
    previous_predicted.loc[first] = pd.to_numeric(out.loc[first, "y_origin"], errors="coerce")
    out["true_increment"] = pd.to_numeric(out["y_true"], errors="coerce") - previous_true
    out["predicted_increment"] = pd.to_numeric(out["y_pred"], errors="coerce") - previous_predicted
    out["increment_error"] = out["predicted_increment"] - out["true_increment"]
    out["increment_direction_correct"] = (
        np.sign(out["predicted_increment"].to_numpy(float))
        == np.sign(out["true_increment"].to_numpy(float))
    )
    return out


def _increment_record(group: pd.DataFrame) -> dict:
    true_increment = pd.to_numeric(group["true_increment"], errors="coerce").to_numpy(float)
    predicted_increment = pd.to_numeric(group["predicted_increment"], errors="coerce").to_numpy(float)
    finite = np.isfinite(true_increment) & np.isfinite(predicted_increment)
    true_increment = true_increment[finite]
    predicted_increment = predicted_increment[finite]
    if len(true_increment) == 0:
        return {
            "increment_rmse": np.nan,
            "increment_mae": np.nan,
            "increment_corr": np.nan,
            "direction_accuracy": np.nan,
            "n": 0,
        }
    corr = (
        float(np.corrcoef(true_increment, predicted_increment)[0, 1])
        if len(true_increment) > 1
        and np.std(true_increment) > 0
        and np.std(predicted_increment) > 0
        else np.nan
    )
    return {
        "increment_rmse": float(
            np.sqrt(mean_squared_error(true_increment, predicted_increment))
        ),
        "increment_mae": float(mean_absolute_error(true_increment, predicted_increment)),
        "increment_corr": corr,
        "direction_accuracy": float(
            np.mean(np.sign(true_increment) == np.sign(predicted_increment))
        ),
        "n": int(len(true_increment)),
    }


def summarize_increment_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    enriched = add_increment_columns(predictions)
    base = [column for column in V41_RUN_COLS if column in enriched.columns]
    rows = []
    for keys, group in enriched.groupby(base + ["requested_horizon"], dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(base + ["requested_horizon"], keys))
        rows.append({**record, "metric_scope": "trajectory", "step_h": np.nan, **_increment_record(group)})
    for keys, group in enriched.groupby(
        base + ["requested_horizon", "step_h"], dropna=False, sort=True
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(base + ["requested_horizon", "step_h"], keys))
        rows.append({**record, "metric_scope": "exact_step", **_increment_record(group)})
    return pd.DataFrame(rows)


def threshold_crossing_metrics(
    predictions: pd.DataFrame, thresholds=(8.0, 10.0)
) -> pd.DataFrame:
    enriched = predictions.copy()
    _required(
        enriched,
        [
            "STR_name",
            "origin_observation_index",
            "requested_horizon",
            "step_h",
            "y_origin",
            "y_true",
            "y_pred",
        ],
    )
    keys = _trajectory_keys(enriched)
    rows = []
    for group_key, group in enriched.groupby(keys, dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        identity = dict(zip(keys, group_key))
        group = group.sort_values("step_h", kind="mergesort")
        origin = float(pd.to_numeric(group["y_origin"], errors="coerce").iloc[0])
        for threshold in thresholds:
            threshold = float(threshold)
            if np.isfinite(origin) and origin >= threshold:
                rows.append(
                    {
                        **identity,
                        "threshold_mm": threshold,
                        "true_cross_step": 0,
                        "predicted_cross_step": 0,
                        "abs_step_error": 0.0,
                        "status": "origin_already_reached",
                    }
                )
                continue
            true_hits = group.loc[
                pd.to_numeric(group["y_true"], errors="coerce").ge(threshold), "step_h"
            ]
            predicted_hits = group.loc[
                pd.to_numeric(group["y_pred"], errors="coerce").ge(threshold), "step_h"
            ]
            true_step = int(true_hits.iloc[0]) if len(true_hits) else np.nan
            predicted_step = int(predicted_hits.iloc[0]) if len(predicted_hits) else np.nan
            if np.isfinite(true_step) and np.isfinite(predicted_step):
                status = "crossed"
                error = abs(float(predicted_step) - float(true_step))
            elif np.isfinite(true_step):
                status = "missed"
                error = np.nan
            elif np.isfinite(predicted_step):
                status = "false_alarm"
                error = np.nan
            else:
                status = "both_not_reached"
                error = np.nan
            rows.append(
                {
                    **identity,
                    "threshold_mm": threshold,
                    "true_cross_step": true_step,
                    "predicted_cross_step": predicted_step,
                    "abs_step_error": error,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def paired_augmentation_superiority(
    control_rows: pd.DataFrame,
    augmented_rows: pd.DataFrame,
    n_boot: int = 10000,
    seed: int = 20260817,
) -> pd.DataFrame:
    _required(control_rows, ["allocation", "seed", "requested_horizon", "y_true", "y_pred"])
    _required(augmented_rows, ["allocation", "seed", "requested_horizon", "y_true", "y_pred"])
    models = sorted(
        set(control_rows.get("model", pd.Series(["all"])).astype(str))
        & set(augmented_rows.get("model", pd.Series(["all"])).astype(str))
    )
    records = []
    for model in models:
        control_model = control_rows if "model" not in control_rows else control_rows.loc[control_rows.model.astype(str).eq(model)]
        augmented_model = augmented_rows if "model" not in augmented_rows else augmented_rows.loc[augmented_rows.model.astype(str).eq(model)]
        horizons = sorted(pd.to_numeric(augmented_model["requested_horizon"], errors="coerce").dropna().astype(int).unique())
        for horizon in horizons:
            frames = []
            current = augmented_model.loc[pd.to_numeric(augmented_model.requested_horizon).eq(horizon)]
            for (allocation, neural_seed), augmented_run in current.groupby(["allocation", "seed"], sort=True):
                control_run = control_model.loc[
                    pd.to_numeric(control_model.allocation).eq(int(allocation))
                    & pd.to_numeric(control_model.seed).eq(int(neural_seed))
                    & pd.to_numeric(control_model.requested_horizon).eq(int(horizon))
                ]
                if control_run.empty:
                    continue
                keys = [key for key in PAIR_KEYS if key in control_run.columns and key in augmented_run.columns]
                paired = control_run[keys + ["y_true", "y_pred"]].merge(
                    augmented_run[keys + ["y_true", "y_pred"]],
                    on=keys,
                    how="inner",
                    suffixes=("_control", "_augmented"),
                    validate="one_to_one",
                )
                if paired.empty:
                    continue
                if not np.allclose(paired.y_true_control, paired.y_true_augmented, atol=1e-9, rtol=0):
                    raise RuntimeError("paired augmentation target mismatch")
                frames.append(
                    pd.DataFrame(
                        {
                            "STR_name": paired["STR_name"].astype(str),
                            "y_true": paired.y_true_control.astype(float),
                            "real_pred": paired.y_pred_control.astype(float),
                            "mix_pred": paired.y_pred_augmented.astype(float),
                        }
                    )
                )
            if not frames:
                continue
            boot = hierarchical_cluster_relative_bootstrap(
                frames, n_boot=int(n_boot), seed=int(seed) + int(horizon) * 100
            )
            records.append(
                {
                    "model": model,
                    "requested_horizon": int(horizon),
                    "n_paired_runs": int(boot["n_runs"]),
                    "n_clusters": int(boot["n_clusters"]),
                    "mean_relative_degradation": float(boot["mean_relative_degradation"]),
                    "lower_ci_relative_degradation": float(boot["ci_low"]),
                    "upper_ci_relative_degradation": float(boot["ci_high"]),
                    "superior": bool(float(boot["ci_high"]) < 0.0),
                }
            )
    return pd.DataFrame(records)


def recovery_ratio_table(
    r50_metrics: pd.DataFrame,
    r50_q50_metrics: pd.DataFrame,
    r100_metrics: pd.DataFrame,
) -> pd.DataFrame:
    required = ["model", "seed", "requested_horizon", "trajectory_rmse"]
    _required(r50_metrics, required + ["allocation"])
    _required(r50_q50_metrics, required + ["allocation"])
    _required(r100_metrics, required)
    paired_keys = ["model", "allocation", "seed", "requested_horizon"]
    paired = r50_metrics[paired_keys + ["trajectory_rmse"]].merge(
        r50_q50_metrics[paired_keys + ["trajectory_rmse"]],
        on=paired_keys,
        how="inner",
        suffixes=("_r50", "_r50_q50"),
        validate="one_to_one",
    )
    reference_keys = ["model", "seed", "requested_horizon"]
    reference = (
        r100_metrics[reference_keys + ["trajectory_rmse"]]
        .drop_duplicates(reference_keys)
        .rename(columns={"trajectory_rmse": "trajectory_rmse_r100"})
    )
    out = paired.merge(reference, on=reference_keys, how="left", validate="many_to_one")
    denominator = out["trajectory_rmse_r50"] - out["trajectory_rmse_r100"]
    numerator = out["trajectory_rmse_r50"] - out["trajectory_rmse_r50_q50"]
    defined = np.isfinite(denominator) & np.isfinite(numerator) & denominator.gt(0)
    out["defined"] = defined
    out["recovery_ratio"] = np.where(defined, numerator / denominator, np.nan)
    return out
