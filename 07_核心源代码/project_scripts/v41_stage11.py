from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import os
import re
import time

import pandas as pd

from v31_experiment_engine import fixed_real_evaluation_arrays
from v31_gru import predict_gru
from v31_tdr import predict_tdr
from v40_audit import audit_one_step_reproduction, run_openloop_leakage_audit
from v40_checkpoint import load_fit_checkpoint, save_fit_checkpoint
from v40_metrics import (
    compute_error_growth,
    horizon_degradation_table,
    paired_noninferiority_by_horizon,
    summarize_prediction_rows,
)
from v40_openloop import build_eval_arrays, forecast_all_origins
from v40_protocol import V40Protocol
from v40_refit import load_frozen_model_configs
from v41_fixed_origin import forecast_end2020_fixed_origin
from v41_metrics import (
    paired_augmentation_superiority,
    recovery_ratio_table,
    summarize_increment_metrics,
    threshold_crossing_metrics,
)
from v41_protocol import (
    V41Protocol,
    V41RunKey,
    checkpoint_dir_v41,
    protocol_signature_v41,
    stage11_run_plan,
)
from v41_refit import fit_stage11_run


def _json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True, default=str),
        encoding="utf-8",
    )


def formal_run_allowed_v41(max_steps: int) -> bool:
    return int(max_steps) == V41Protocol().max_optimizer_steps


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _manifest_is_formal(path: Path) -> bool:
    manifest = _read_json(path)
    return bool(manifest.get("formal")) and int(manifest.get("max_steps", -1)) == int(
        V41Protocol().max_optimizer_steps
    )


def _audit_is_accepted(path: Path, require_formal: bool = False) -> bool:
    status = _read_json(path).get("status")
    allowed = {"PASS"} if require_formal else {"PASS", "SMOKE_ONLY"}
    return status in allowed


def is_run_complete_v41(
    root: str | Path, run_key: V41RunKey, require_formal: bool = False
) -> bool:
    directory = checkpoint_dir_v41(root, run_key)
    required = [
        directory / "checkpoint.pt",
        directory / "run_manifest.json",
        directory / "checkpoint_audit.json",
        directory / "one_step_reference.csv",
        directory / "training_counts.json",
        directory / "metrics.json",
        directory / "fixed_origin_predictions.csv",
    ]
    required.extend(
        directory / f"pred_H{horizon:03d}.csv"
        for horizon in V41Protocol().horizons
    )
    if not all(path.exists() for path in required):
        return False
    if require_formal and not _manifest_is_formal(directory / "run_manifest.json"):
        return False
    return _audit_is_accepted(
        directory / "checkpoint_audit.json", require_formal=require_formal
    )


def _split_models(raw: str) -> set[str]:
    return {part for part in re.split(r"[\s,;]+", raw.strip().lower()) if part}


def _interleave_conditions(plan: list[V41RunKey]) -> list[V41RunKey]:
    """Keep short smoke plans scientifically useful by including both conditions."""
    ordered: list[V41RunKey] = []
    model_order = list(dict.fromkeys(run.model for run in plan))
    for model in model_order:
        by_condition = {
            condition: [
                run for run in plan if run.model == model and run.condition == condition
            ]
            for condition in ("real_reduced", "qwen_additive")
        }
        width = max((len(rows) for rows in by_condition.values()), default=0)
        for index in range(width):
            for condition in ("real_reduced", "qwen_additive"):
                rows = by_condition[condition]
                if index < len(rows):
                    ordered.append(rows[index])
    return ordered


def _filter_plan_v41(plan: list[V41RunKey]) -> list[V41RunKey]:
    selected = list(plan)
    raw_models = os.environ.get("V41_MODELS", "").strip()
    if raw_models:
        models = _split_models(raw_models)
        unknown = models - {"gru", "tdr"}
        if unknown:
            raise ValueError(f"unsupported V41_MODELS: {sorted(unknown)}")
        selected = [run for run in selected if run.model in models]
    selected = _interleave_conditions(selected)
    raw_limit = os.environ.get("V41_MAX_RUNS", "").strip()
    if raw_limit:
        limit = int(raw_limit)
        if limit < 0:
            raise ValueError("V41_MAX_RUNS must be non-negative")
        selected = selected[:limit]
    return selected


def _annotate_prediction_rows(
    predictions: pd.DataFrame, run_key: V41RunKey
) -> pd.DataFrame:
    out = predictions.copy()
    values = (
        ("seed", int(run_key.seed)),
        ("allocation", int(run_key.allocation)),
        ("synthetic_fraction", float(run_key.synthetic_fraction)),
        ("real_fraction", float(run_key.real_fraction)),
        ("condition", str(run_key.condition)),
        ("model", str(run_key.model)),
    )
    for column, value in values:
        if column in out.columns:
            out[column] = value
        else:
            out.insert(0, column, value)
    return out


def _one_step_predictions(master: pd.DataFrame, fit: dict, model: str) -> pd.DataFrame:
    arrays = fixed_real_evaluation_arrays(master, seq_len=V41Protocol().seq_len)[
        "eval2021"
    ]
    X, y, previous, peak, meta = arrays
    if model == "gru":
        predictions = predict_gru(fit, X, previous)
    elif model == "tdr":
        predictions = predict_tdr(fit, X, previous, peak)
    else:
        raise ValueError(f"unsupported model: {model}")
    return pd.DataFrame(
        {
            "STR_name": meta["STR_name"].astype(str).to_numpy(),
            "Observation_Index": pd.to_numeric(meta["Observation_Order"]).astype(int),
            "y_true": y,
            "y_pred": predictions,
            "prev": previous.reshape(-1),
        }
    )


def _save_new_checkpoint(
    root: Path,
    run_key: V41RunKey,
    result: dict,
    max_steps: int,
) -> tuple[dict, dict]:
    directory = checkpoint_dir_v41(root, run_key)
    directory.mkdir(parents=True, exist_ok=True)
    configs = load_frozen_model_configs(root)
    metadata = {
        "run_stem": run_key.stem,
        "model": run_key.model,
        "condition": run_key.condition,
        "real_fraction": float(run_key.real_fraction),
        "synthetic_fraction": float(run_key.synthetic_fraction),
        "allocation": int(run_key.allocation),
        "seed": int(run_key.seed),
        "max_steps": int(max_steps),
        "formal": formal_run_allowed_v41(max_steps),
        "v41_protocol_signature": protocol_signature_v41(),
        "selected_dev_rmse": result.get("dev_metrics", {}).get("rmse"),
        "steps_completed": result.get("fit", {}).get("steps_completed"),
    }
    if run_key.model == "gru":
        model_config = asdict(configs["gru"])
    else:
        model_config = {
            "input_dim": 23,
            "seq_len": V41Protocol().seq_len,
            "batch_size": 64,
            "stable": asdict(configs["tdr"]),
        }
    save_fit_checkpoint(
        directory / "checkpoint.pt",
        run_key.model,
        result["fit"],
        model_config,
        metadata,
    )
    _json_dump(directory / "run_manifest.json", metadata)
    _json_dump(directory / "training_counts.json", result["training_counts"])
    return result["fit"], metadata


def _ensure_checkpoint_audit(
    directory: Path,
    master: pd.DataFrame,
    fit: dict,
    model: str,
    max_steps: int,
) -> dict:
    audit_path = directory / "checkpoint_audit.json"
    formal = formal_run_allowed_v41(max_steps)
    if _audit_is_accepted(audit_path, require_formal=formal):
        return _read_json(audit_path)
    reference_path = directory / "one_step_reference.csv"
    if not reference_path.exists():
        recovered = _one_step_predictions(master, fit, model)
        recovered.to_csv(reference_path, index=False)
    try:
        reference = pd.read_csv(reference_path)
        checkpoint_predictions = _one_step_predictions(master, fit, model)
        audit = audit_one_step_reproduction(
            checkpoint_predictions, reference, V40Protocol()
        )
    except Exception as exc:
        audit = {"status": "FAIL", "error": repr(exc)}
    if not formal:
        observed_status = audit.get("status")
        audit = {
            **audit,
            "observed_audit_status": observed_status,
            "status": "SMOKE_ONLY",
            "formal_reproduction_required": False,
            "note": (
                "This is a non-2400-step smoke run. It verifies the pipeline only "
                "and must not be used for scientific conclusions."
            ),
        }
    _json_dump(audit_path, audit)
    if formal and audit.get("status") != "PASS":
        raise RuntimeError(f"Stage 11 checkpoint audit failed: {audit}")
    return audit


def _attach_condition_to_summary(
    frame: pd.DataFrame, run_key: V41RunKey
) -> pd.DataFrame:
    out = frame.copy()
    if "condition" not in out.columns:
        out.insert(1 if "model" in out.columns else 0, "condition", run_key.condition)
    if "real_fraction" not in out.columns:
        position = 2 if "model" in out.columns else 1
        out.insert(position, "real_fraction", float(run_key.real_fraction))
    return out


def _write_run_metrics(directory: Path, run_key: V41RunKey) -> dict:
    horizon_frames = [
        pd.read_csv(directory / f"pred_H{horizon:03d}.csv")
        for horizon in V41Protocol().horizons
        if (directory / f"pred_H{horizon:03d}.csv").exists()
    ]
    if not horizon_frames:
        raise RuntimeError("no Stage 11 multi-horizon predictions to summarize")
    predictions = pd.concat(horizon_frames, ignore_index=True)
    standard = summarize_prediction_rows(predictions)
    horizon_metrics = _attach_condition_to_summary(
        standard["horizon_metrics"], run_key
    )
    step_metrics = _attach_condition_to_summary(standard["step_metrics"], run_key)
    error_growth = compute_error_growth(step_metrics)
    horizon_degradation = horizon_degradation_table(horizon_metrics)
    increment_metrics = summarize_increment_metrics(predictions)
    threshold_metrics = threshold_crossing_metrics(
        predictions, thresholds=V41Protocol().thresholds_mm
    )

    fixed_predictions = pd.read_csv(directory / "fixed_origin_predictions.csv")
    fixed_metrics = _attach_condition_to_summary(
        summarize_prediction_rows(fixed_predictions)["horizon_metrics"], run_key
    )

    tables = {
        "horizon_metrics.csv": horizon_metrics,
        "step_metrics.csv": step_metrics,
        "error_growth.csv": error_growth,
        "horizon_degradation.csv": horizon_degradation,
        "increment_metrics.csv": increment_metrics,
        "threshold_crossing_metrics.csv": threshold_metrics,
        "fixed_origin_metrics.csv": fixed_metrics,
    }
    for filename, frame in tables.items():
        frame.to_csv(directory / filename, index=False)
    payload = {
        "horizon_metrics": horizon_metrics.to_dict("records"),
        "step_metrics": step_metrics.to_dict("records"),
        "error_growth": error_growth.to_dict("records"),
        "horizon_degradation": horizon_degradation.to_dict("records"),
        "increment_metrics": increment_metrics.to_dict("records"),
        "fixed_origin_metrics": fixed_metrics.to_dict("records"),
        "threshold_crossing_row_count": int(len(threshold_metrics)),
        "primary_horizon": V41Protocol().primary_horizon,
    }
    _json_dump(directory / "metrics.json", payload)
    return payload


def run_one_v41_condition(
    root: str | Path,
    run_key: V41RunKey,
    master: pd.DataFrame | None = None,
    max_steps: int | None = None,
) -> dict:
    root = Path(root)
    protocol = V41Protocol()
    steps = int(protocol.max_optimizer_steps if max_steps is None else max_steps)
    directory = checkpoint_dir_v41(root, run_key)
    directory.mkdir(parents=True, exist_ok=True)
    if master is None:
        master = pd.read_csv(root / "data/processed/rutting_master_v31.csv")

    checkpoint_path = directory / "checkpoint.pt"
    manifest_path = directory / "run_manifest.json"
    if checkpoint_path.exists():
        if formal_run_allowed_v41(steps) and not _manifest_is_formal(manifest_path):
            raise RuntimeError(
                "A smoke checkpoint cannot be reused as a formal 2400-step result. "
                f"Use a clean output directory for {run_key.stem}."
            )
        fit = load_fit_checkpoint(checkpoint_path, "auto")
        if not manifest_path.exists():
            _json_dump(
                manifest_path,
                {
                    "run_stem": run_key.stem,
                    "formal": False,
                    "max_steps": steps,
                    "recovered_manifest": True,
                },
            )
    else:
        result = fit_stage11_run(root, master, run_key, max_steps=steps)
        result["eval_predictions"].to_csv(
            directory / "one_step_reference.csv", index=False
        )
        fit, _ = _save_new_checkpoint(root, run_key, result, steps)

    _ensure_checkpoint_audit(directory, master, fit, run_key.model, steps)

    missing_horizons = [
        horizon
        for horizon in protocol.horizons
        if not (directory / f"pred_H{horizon:03d}.csv").exists()
    ]
    if missing_horizons:
        evaluation_arrays = build_eval_arrays(master, protocol.seq_len)
        for horizon in missing_horizons:
            prediction = forecast_all_origins(
                fit, evaluation_arrays, horizon, run_key.model
            )
            _annotate_prediction_rows(prediction, run_key).to_csv(
                directory / f"pred_H{horizon:03d}.csv", index=False
            )

    fixed_path = directory / "fixed_origin_predictions.csv"
    if not fixed_path.exists():
        fixed = forecast_end2020_fixed_origin(
            fit,
            master,
            run_key.model,
            run_key=run_key,
            seq_len=protocol.seq_len,
        )
        fixed.to_csv(fixed_path, index=False)

    metrics = _write_run_metrics(directory, run_key)
    return {
        "run_key": run_key,
        "directory": directory,
        "metrics": metrics,
        "complete": is_run_complete_v41(
            root, run_key, require_formal=formal_run_allowed_v41(steps)
        ),
    }


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _concatenate(frames: list[pd.DataFrame]) -> pd.DataFrame:
    usable = [frame for frame in frames if len(frame)]
    return pd.concat(usable, ignore_index=True) if usable else pd.DataFrame()


def _aggregate_v41(root: Path) -> dict[str, object]:
    checkpoint_root = root / "checkpoints/v41"
    directories = (
        sorted({path.parent for path in checkpoint_root.glob("**/run_manifest.json")})
        if checkpoint_root.exists()
        else []
    )
    buckets: dict[str, list[pd.DataFrame]] = {
        "horizon_metrics": [],
        "increment_metrics": [],
        "threshold_metrics": [],
        "fixed_metrics": [],
        "predictions": [],
    }
    counts = []
    audits = []
    for directory in directories:
        manifest = _read_json(directory / "run_manifest.json")
        count = _read_json(directory / "training_counts.json")
        if count:
            counts.append(
                {
                    **{
                        key: manifest.get(key)
                        for key in (
                            "run_stem",
                            "model",
                            "condition",
                            "real_fraction",
                            "synthetic_fraction",
                            "allocation",
                            "seed",
                            "formal",
                            "max_steps",
                        )
                    },
                    **count,
                }
            )
        audit = _read_json(directory / "checkpoint_audit.json")
        if audit:
            audits.append(
                {
                    **{
                        key: manifest.get(key)
                        for key in ("run_stem", "model", "condition", "formal")
                    },
                    **audit,
                }
            )
        mapping = {
            "horizon_metrics.csv": "horizon_metrics",
            "increment_metrics.csv": "increment_metrics",
            "threshold_crossing_metrics.csv": "threshold_metrics",
            "fixed_origin_metrics.csv": "fixed_metrics",
        }
        for filename, bucket in mapping.items():
            frame = _read_csv(directory / filename)
            if len(frame):
                buckets[bucket].append(frame)
        for horizon in V41Protocol().horizons:
            frame = _read_csv(directory / f"pred_H{horizon:03d}.csv")
            if len(frame):
                buckets["predictions"].append(frame)
    return {
        "directories": directories,
        "counts": pd.DataFrame(counts),
        "audits": pd.DataFrame(audits),
        **{key: _concatenate(value) for key, value in buckets.items()},
    }


def _aggregate_v40(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    checkpoint_root = root / "checkpoints/v40"
    if not checkpoint_root.exists():
        return pd.DataFrame(), pd.DataFrame()
    predictions = []
    metrics = []
    for path in checkpoint_root.glob("**/pred_H*.csv"):
        frame = _read_csv(path)
        if len(frame):
            predictions.append(frame)
    for path in checkpoint_root.glob("**/horizon_metrics.csv"):
        frame = _read_csv(path)
        if len(frame):
            metrics.append(frame)
    all_predictions = _concatenate(predictions)
    all_metrics = _concatenate(metrics)
    if all_metrics.empty and len(all_predictions):
        all_metrics = summarize_prediction_rows(all_predictions)["horizon_metrics"]
    return all_predictions, all_metrics


def _empty_or_compute_augmentation(
    v41_predictions: pd.DataFrame, v40_predictions: pd.DataFrame, n_boot: int
) -> pd.DataFrame:
    columns = [
        "model",
        "requested_horizon",
        "n_paired_runs",
        "n_clusters",
        "mean_relative_degradation",
        "lower_ci_relative_degradation",
        "upper_ci_relative_degradation",
        "superior",
    ]
    needed_v41 = {"condition", "model", "allocation", "seed"}
    needed_v40 = {"method", "model", "synthetic_fraction", "allocation", "seed"}
    if not needed_v41.issubset(v41_predictions.columns) or not needed_v40.issubset(
        v40_predictions.columns
    ):
        return pd.DataFrame(columns=columns)
    reduced = v41_predictions.loc[
        v41_predictions["condition"].astype(str).eq("real_reduced")
    ]
    qwen = v40_predictions.loc[
        v40_predictions["method"].astype(str).eq("qwen")
        & pd.to_numeric(v40_predictions["synthetic_fraction"], errors="coerce").eq(0.5)
    ]
    if reduced.empty or qwen.empty:
        return pd.DataFrame(columns=columns)
    return paired_augmentation_superiority(
        reduced, qwen, n_boot=n_boot, seed=20260817
    )


def _empty_or_compute_additive(
    v41_predictions: pd.DataFrame, v40_predictions: pd.DataFrame, n_boot: int
) -> pd.DataFrame:
    columns = [
        "model",
        "requested_horizon",
        "synthetic_fraction",
        "margin",
        "n_paired_runs",
        "n_clusters",
        "mean_relative_degradation",
        "lower_ci_relative_degradation",
        "upper_ci_relative_degradation",
        "superior",
    ]
    if "condition" not in v41_predictions or "method" not in v40_predictions:
        return pd.DataFrame(columns=columns)
    pieces = []
    for model in ("gru", "tdr"):
        full_real = v40_predictions.loc[
            v40_predictions["model"].astype(str).eq(model)
            & v40_predictions["method"].astype(str).eq("full_real")
        ]
        additive = v41_predictions.loc[
            v41_predictions["model"].astype(str).eq(model)
            & v41_predictions["condition"].astype(str).eq("qwen_additive")
        ]
        if full_real.empty or additive.empty:
            continue
        result = paired_noninferiority_by_horizon(
            full_real,
            additive,
            margin=0.0,
            n_boot=n_boot,
            seed=20260817,
        )
        if len(result):
            result.insert(0, "model", model)
            result = result.rename(columns={"noninferior": "superior"})
            pieces.append(result)
    return _concatenate(pieces) if pieces else pd.DataFrame(columns=columns)


def _empty_or_compute_recovery(
    v41_metrics: pd.DataFrame, v40_metrics: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        "model",
        "allocation",
        "seed",
        "requested_horizon",
        "trajectory_rmse_r50",
        "trajectory_rmse_r50_q50",
        "trajectory_rmse_r100",
        "defined",
        "recovery_ratio",
    ]
    if "condition" not in v41_metrics or "method" not in v40_metrics:
        return pd.DataFrame(columns=columns)
    r50 = v41_metrics.loc[
        v41_metrics["condition"].astype(str).eq("real_reduced")
    ]
    qwen = v40_metrics.loc[
        v40_metrics["method"].astype(str).eq("qwen")
        & pd.to_numeric(v40_metrics["synthetic_fraction"], errors="coerce").eq(0.5)
    ]
    full = v40_metrics.loc[
        v40_metrics["method"].astype(str).eq("full_real")
    ]
    if r50.empty or qwen.empty or full.empty:
        return pd.DataFrame(columns=columns)
    return recovery_ratio_table(r50, qwen, full)


def _write_frame(path: Path, frame: pd.DataFrame, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(frame):
        frame.to_csv(path, index=False)
    else:
        pd.DataFrame(columns=columns).to_csv(path, index=False)


def write_v41_reports(root: str | Path) -> Path:
    root = Path(root)
    output = root / "reports/v41"
    output.mkdir(parents=True, exist_ok=True)
    aggregate = _aggregate_v41(root)
    v40_predictions, v40_metrics = _aggregate_v40(root)
    n_boot = int(os.environ.get("V41_BOOTSTRAP_N", "10000"))

    _write_frame(
        output / "metrics/multi_horizon_run_metrics.csv",
        aggregate["horizon_metrics"],
        [
            "model",
            "condition",
            "real_fraction",
            "synthetic_fraction",
            "allocation",
            "seed",
            "requested_horizon",
            "trajectory_rmse",
            "trajectory_mae",
            "trajectory_r2",
            "persistence_rmse",
            "persistence_skill",
        ],
    )
    _write_frame(
        output / "metrics/increment_metrics.csv",
        aggregate["increment_metrics"],
        [
            "model",
            "condition",
            "real_fraction",
            "synthetic_fraction",
            "allocation",
            "seed",
            "requested_horizon",
            "metric_scope",
            "step_h",
            "increment_rmse",
            "increment_mae",
            "increment_corr",
            "direction_accuracy",
            "n",
        ],
    )
    _write_frame(
        output / "metrics/threshold_crossing_metrics.csv",
        aggregate["threshold_metrics"],
        [
            "model",
            "condition",
            "real_fraction",
            "synthetic_fraction",
            "allocation",
            "seed",
            "STR_name",
            "origin_observation_index",
            "requested_horizon",
            "threshold_mm",
            "true_cross_step",
            "predicted_cross_step",
            "abs_step_error",
            "status",
        ],
    )
    _write_frame(
        output / "metrics/fixed_origin_metrics.csv",
        aggregate["fixed_metrics"],
        [
            "model",
            "condition",
            "real_fraction",
            "synthetic_fraction",
            "allocation",
            "seed",
            "requested_horizon",
            "trajectory_rmse",
            "trajectory_mae",
            "trajectory_r2",
            "persistence_rmse",
            "persistence_skill",
        ],
    )

    augmentation = _empty_or_compute_augmentation(
        aggregate["predictions"], v40_predictions, n_boot
    )
    additive = _empty_or_compute_additive(
        aggregate["predictions"], v40_predictions, n_boot
    )
    recovery = _empty_or_compute_recovery(aggregate["horizon_metrics"], v40_metrics)
    _write_frame(
        output / "augmentation/augmentation_superiority.csv",
        augmentation,
        [
            "model",
            "requested_horizon",
            "n_paired_runs",
            "n_clusters",
            "mean_relative_degradation",
            "lower_ci_relative_degradation",
            "upper_ci_relative_degradation",
            "superior",
        ],
    )
    _write_frame(
        output / "augmentation/additive_superiority.csv",
        additive,
        [
            "model",
            "requested_horizon",
            "synthetic_fraction",
            "margin",
            "n_paired_runs",
            "n_clusters",
            "mean_relative_degradation",
            "lower_ci_relative_degradation",
            "upper_ci_relative_degradation",
            "superior",
        ],
    )
    _write_frame(
        output / "augmentation/recovery_ratio.csv",
        recovery,
        [
            "model",
            "allocation",
            "seed",
            "requested_horizon",
            "trajectory_rmse_r50",
            "trajectory_rmse_r50_q50",
            "trajectory_rmse_r100",
            "defined",
            "recovery_ratio",
        ],
    )
    _write_frame(
        output / "audit/training_counts.csv",
        aggregate["counts"],
        [
            "run_stem",
            "model",
            "condition",
            "real_fraction",
            "synthetic_fraction",
            "allocation",
            "seed",
            "formal",
            "max_steps",
            "n_historical_real",
            "n_2019_real",
            "n_full_real",
            "n_synthetic",
            "n_total",
        ],
    )
    _write_frame(
        output / "audit/checkpoint_audit_summary.csv",
        aggregate["audits"],
        ["run_stem", "model", "condition", "formal", "status"],
    )

    master_path = root / "data/processed/rutting_master_v31.csv"
    if master_path.exists():
        try:
            master = pd.read_csv(master_path)
            evaluation = build_eval_arrays(master, V41Protocol().seq_len)
            gru_audit = run_openloop_leakage_audit(evaluation, "gru", horizon=3)
            tdr_audit = run_openloop_leakage_audit(evaluation, "tdr", horizon=3)
            leakage = {
                "gru": gru_audit,
                "tdr": tdr_audit,
                "status": (
                    "PASS"
                    if gru_audit.get("status") == "PASS"
                    and tdr_audit.get("status") == "PASS"
                    else "FAIL"
                ),
            }
        except Exception as exc:
            leakage = {"status": "FAIL", "error": repr(exc)}
    else:
        leakage = {"status": "NOT_RUN", "reason": "master data unavailable"}
    _json_dump(output / "audit/leakage_test_results.json", leakage)

    directories = aggregate["directories"]
    formal_runs = sum(
        bool(_read_json(directory / "run_manifest.json").get("formal"))
        for directory in directories
    )
    complete_runs = 0
    for directory in directories:
        manifest = _read_json(directory / "run_manifest.json")
        try:
            key = V41RunKey(
                manifest["model"],
                manifest["condition"],
                float(manifest["real_fraction"]),
                float(manifest["synthetic_fraction"]),
                int(manifest["allocation"]),
                int(manifest["seed"]),
            )
            complete_runs += int(is_run_complete_v41(root, key))
        except Exception:
            continue
    protocol = V41Protocol()
    manifest = {
        "stage": "V4.1 Stage 11",
        "v41_protocol_signature": protocol_signature_v41(),
        "planned_runs": len(stage11_run_plan(protocol)),
        "conditions": ["real_reduced", "qwen_additive"],
        "condition_definitions": {
            "real_reduced": "R50: 50% of real 2019 updates; no synthetic rows",
            "qwen_additive": "R100+Q50: all real updates plus Qwen synthetic rows",
        },
        "primary_horizon": protocol.primary_horizon,
        "horizons": list(protocol.horizons),
        "thresholds_mm": list(protocol.thresholds_mm),
        "max_optimizer_steps": protocol.max_optimizer_steps,
        "run_directories_seen": len(directories),
        "completed_run_directories": int(complete_runs),
        "formal_run_directories": int(formal_runs),
        "smoke_run_directories": int(len(directories) - formal_runs),
        "leakage_audit_status": leakage.get("status"),
        "comparison_outputs": {
            "augmentation_superiority_rows": int(len(augmentation)),
            "additive_superiority_rows": int(len(additive)),
            "recovery_ratio_rows": int(len(recovery)),
        },
        "interpretation_rule": (
            "Formal conclusions require 2400-step runs, PASS checkpoint audits, "
            "the frozen 2021 chronological evaluation, and paired uncertainty intervals."
        ),
    }
    _json_dump(output / "audit/stage11_manifest.json", manifest)
    return output


def stage11(root: str | Path) -> list[str]:
    root = Path(root)
    plan = _filter_plan_v41(stage11_run_plan())
    raw_steps = os.environ.get("V41_MAX_STEPS", "").strip()
    max_steps = int(raw_steps) if raw_steps else V41Protocol().max_optimizer_steps
    master = pd.read_csv(root / "data/processed/rutting_master_v31.csv")
    require_formal = formal_run_allowed_v41(max_steps)
    start = time.time()
    completed = []
    for index, run_key in enumerate(plan, 1):
        if is_run_complete_v41(root, run_key, require_formal=require_formal):
            state = "complete"
        else:
            run_one_v41_condition(
                root, run_key, master=master, max_steps=max_steps
            )
            state = "done"
        elapsed = time.time() - start
        eta = (elapsed / max(index, 1)) * (len(plan) - index)
        print(
            f"[Stage11] {index:03d}/{len(plan):03d} {run_key.stem} "
            f"state={state} elapsed={elapsed / 60:.1f}m ETA={eta / 60:.1f}m"
        )
        completed.append(run_key.stem)
    write_v41_reports(root)
    return completed

