from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from v31_downstream_data import build_real_downstream_frame, build_sequence_arrays_simple
from v31_experiment_engine import (
    evaluate_predictions,
    fixed_real_evaluation_arrays,
    run_gru_condition,
    run_tdr_condition,
    training_frame_with_update,
)
from v31_gru import fit_gru_step_budget, predict_gru
from v31_tdr import fit_tdr_step_budget, predict_tdr
from v40_refit import load_frozen_model_configs
from v41_data import build_additive_training_arrays, build_reduced_real_update
from v41_protocol import V41Protocol, V41RunKey


def _allocation_seed(run_key: V41RunKey, protocol: V41Protocol | None = None) -> int:
    p = protocol or V41Protocol()
    return int(p.allocation_seeds[int(run_key.allocation) - 1])


def load_stage11_update(
    root: str | Path,
    master: pd.DataFrame,
    run_key: V41RunKey,
    protocol: V41Protocol | None = None,
) -> pd.DataFrame:
    root = Path(root)
    p = protocol or V41Protocol()
    if run_key.condition == "real_reduced":
        return build_reduced_real_update(
            master,
            real_fraction=run_key.real_fraction,
            allocation_seed=_allocation_seed(run_key, p),
            year=2019,
        )
    if run_key.condition == "qwen_additive":
        qtag = int(round(float(run_key.synthetic_fraction) * 100))
        path = (
            root
            / "data"
            / "v31_updates"
            / f"qwen_Q{qtag:03d}_A{int(run_key.allocation)}.csv"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_csv(path)
    raise ValueError(f"unsupported Stage 11 condition: {run_key.condition}")


def _prediction_frame(meta, y, pred, prev) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "STR_name": meta["STR_name"].astype(str).to_numpy(),
            "Observation_Index": pd.to_numeric(meta["Observation_Order"]).astype(int).to_numpy(),
            "y_true": np.asarray(y, dtype=float).reshape(-1),
            "y_pred": np.asarray(pred, dtype=float).reshape(-1),
            "prev": np.asarray(prev, dtype=float).reshape(-1),
        }
    )


def _reduced_training_counts(master, update, seq_len: int) -> dict:
    panel = training_frame_with_update(master, update, year=2019)
    frame = build_real_downstream_frame(panel)
    _, y, _, _, meta = build_sequence_arrays_simple(frame, seq_len=seq_len)
    years = pd.to_numeric(meta["Year"], errors="coerce")
    keep = years.le(2019).to_numpy()
    years = years.loc[keep]
    n_historical = int(years.le(2018).sum())
    n_2019 = int(years.eq(2019).sum())
    return {
        "n_historical_real": n_historical,
        "n_2019_real": n_2019,
        "n_full_real": n_historical + n_2019,
        "n_synthetic": 0,
        "n_total": int(np.asarray(y)[keep].size),
    }


def fit_stage11_run(
    root: str | Path,
    master: pd.DataFrame,
    run_key: V41RunKey,
    max_steps: int | None = None,
    device: str = "auto",
) -> dict:
    p = V41Protocol()
    configs = load_frozen_model_configs(root)
    steps = int(p.max_optimizer_steps if max_steps is None else max_steps)
    warmup = p.warmup_steps if steps >= p.warmup_steps else max(1, steps // 4)
    val_every = (
        p.validation_every_steps
        if steps >= p.validation_every_steps
        else max(1, steps // 2)
    )
    common = {
        "max_steps": steps,
        "warmup_steps": warmup,
        "val_every": val_every,
        "patience_checks": p.patience_checks,
        "seq_len": p.seq_len,
    }
    update = load_stage11_update(root, master, run_key, p)

    if run_key.condition == "real_reduced":
        if run_key.model == "gru":
            result = run_gru_condition(
                master, update, configs["gru"], seed=run_key.seed, **common
            )
        else:
            result = run_tdr_condition(
                master, update, configs["tdr"], seed=run_key.seed, **common
            )
        result["training_counts"] = _reduced_training_counts(master, update, p.seq_len)
        return result

    arrays = build_additive_training_arrays(master, update, seq_len=p.seq_len, year=2019)
    fixed = fixed_real_evaluation_arrays(master, seq_len=p.seq_len)
    Xv, yv, pv, pkv, mv = fixed["dev2020"]
    Xe, ye, pe, pke, me = fixed["eval2021"]
    if run_key.model == "gru":
        fit = fit_gru_step_budget(
            arrays["X"],
            arrays["y"],
            arrays["prev"],
            Xv,
            yv,
            pv,
            configs["gru"],
            max_steps=steps,
            warmup_steps=warmup,
            val_every=val_every,
            patience_checks=p.patience_checks,
            seed=run_key.seed,
            device=device,
        )
        pred_val = predict_gru(fit, Xv, pv)
        pred_eval = predict_gru(fit, Xe, pe)
    else:
        fit = fit_tdr_step_budget(
            arrays["X"],
            arrays["y"],
            arrays["prev"],
            arrays["peak"],
            Xv,
            yv,
            pv,
            pkv,
            configs["tdr"],
            max_steps=steps,
            warmup_steps=warmup,
            val_every=val_every,
            patience_checks=p.patience_checks,
            batch_size=64,
            seed=run_key.seed,
            device=device,
        )
        pred_val = predict_tdr(fit, Xv, pv, pkv)
        pred_eval = predict_tdr(fit, Xe, pe, pke)
    return {
        "fit": fit,
        "dev_metrics": evaluate_predictions(yv, pred_val, pv),
        "eval_metrics": evaluate_predictions(ye, pred_eval, pe),
        "dev_predictions": _prediction_frame(mv, yv, pred_val, pv),
        "eval_predictions": _prediction_frame(me, ye, pred_eval, pe),
        "n_train": int(len(arrays["y"])),
        "training_counts": arrays["counts"],
    }
