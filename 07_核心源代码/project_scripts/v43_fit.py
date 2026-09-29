from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from v31_downstream_data import build_real_downstream_frame, build_sequence_arrays_simple
from v31_experiment_engine import evaluate_predictions
from v31_tdr import fit_tdr_step_budget, predict_tdr
from v40_refit import load_frozen_model_configs
from v41_data import build_additive_training_arrays
from v43_data import split_master_for_fold, audit_fold_isolation
from v43_protocol import V43Fold, V43Protocol, V43RunKey, get_fold_v43
from v43_qwen import fold_qwen_paths


def _arrays_for_year(frame: pd.DataFrame, year: int, seq_len: int) -> dict:
    X, y, prev, peak, meta = build_sequence_arrays_simple(frame, seq_len=int(seq_len))
    years = pd.to_numeric(meta["Year"], errors="coerce")
    mask = years.eq(int(year)).to_numpy()
    return {
        "X": X[mask],
        "y": y[mask],
        "prev": prev[mask],
        "peak": peak[mask],
        "meta": meta.loc[mask].reset_index(drop=True),
    }


def _full_real_arrays(train_master: pd.DataFrame, seq_len: int, year_max: int = 2019) -> dict:
    panel = train_master.loc[pd.to_numeric(train_master["Year"], errors="coerce").le(int(year_max))].copy()
    frame = build_real_downstream_frame(panel)
    X, y, prev, peak, meta = build_sequence_arrays_simple(frame, seq_len=int(seq_len))
    years = pd.to_numeric(meta["Year"], errors="coerce")
    keep = years.le(int(year_max)).to_numpy()
    X, y, prev, peak = X[keep], y[keep], prev[keep], peak[keep]
    meta = meta.loc[keep].reset_index(drop=True)
    years = pd.to_numeric(meta["Year"], errors="coerce")
    return {
        "X": X, "y": y, "prev": prev, "peak": peak, "meta": meta,
        "counts": {
            "n_historical_real": int(years.le(2018).sum()),
            "n_2019_real": int(years.eq(2019).sum()),
            "n_full_real": int(len(y)),
            "n_synthetic": 0,
            "n_total": int(len(y)),
        },
    }


def build_v43_training_arrays(
    master: pd.DataFrame,
    fold: V43Fold,
    condition: str,
    qwen_update: pd.DataFrame | None = None,
    seq_len: int | None = None,
) -> dict:
    p = V43Protocol()
    seq_len = int(seq_len or p.seq_len)
    train_master, _ = split_master_for_fold(master, fold)
    if condition == "full_real":
        arrays = _full_real_arrays(train_master, seq_len, p.downstream_train_year_max)
    elif condition == "qwen_additive":
        if qwen_update is None:
            raise ValueError("qwen_additive requires a fold-specific Qwen update")
        bad = set(qwen_update.get("STR_name", pd.Series(dtype=str)).astype(str)) - set(fold.train_structures)
        if bad:
            raise RuntimeError(f"held-out/unexpected structures in additive update: {sorted(bad)}")
        arrays = build_additive_training_arrays(
            train_master,
            qwen_update,
            seq_len=seq_len,
            year=p.downstream_train_year_max,
        )
    else:
        raise ValueError(f"unsupported Stage 13 condition: {condition}")

    years = pd.to_numeric(arrays["meta"]["Year"], errors="coerce")
    structures = set(arrays["meta"]["STR_name"].astype(str))
    if years.isna().any() or int(years.max()) > p.downstream_train_year_max:
        raise RuntimeError("Stage 13 downstream training contains post-2019 targets")
    if not structures.issubset(set(fold.train_structures)):
        raise RuntimeError("Stage 13 downstream training contains held-out structures")
    return arrays


def build_v43_dev_arrays(master: pd.DataFrame, fold: V43Fold, seq_len: int | None = None) -> dict:
    p = V43Protocol()
    seq_len = int(seq_len or p.seq_len)
    train_master, _ = split_master_for_fold(master, fold)
    frame = build_real_downstream_frame(train_master)
    out = _arrays_for_year(frame, p.downstream_dev_year, seq_len)
    if len(out["y"]) == 0:
        raise ValueError(f"fold {fold.fold_id} has no allowed-structure 2020 development targets")
    if set(out["meta"]["STR_name"].astype(str)).intersection(fold.held_out_structures):
        raise RuntimeError("held-out structure leaked into Stage 13 dev arrays")
    return out


def fit_v43_run(
    root: str | Path,
    master: pd.DataFrame,
    run_key: V43RunKey,
    max_steps: int | None = None,
    device: str = "auto",
    fit_fn=fit_tdr_step_budget,
    predict_fn=predict_tdr,
    stable_config=None,
) -> dict:
    root = Path(root)
    p = V43Protocol()
    fold = get_fold_v43(run_key.fold_id)
    steps = int(p.max_optimizer_steps if max_steps is None else max_steps)
    warmup = p.warmup_steps if steps >= p.warmup_steps else max(1, steps // 4)
    val_every = p.validation_every_steps if steps >= p.validation_every_steps else max(1, steps // 2)

    qwen_update = None
    if run_key.condition == "qwen_additive":
        update_path = fold_qwen_paths(root, fold.fold_id)["updates_dir"] / f"qwen_Q050_A{int(run_key.allocation)}.csv"
        if not update_path.exists():
            raise FileNotFoundError(update_path)
        qwen_update = pd.read_csv(update_path)
    arrays = build_v43_training_arrays(master, fold, run_key.condition, qwen_update=qwen_update, seq_len=p.seq_len)
    dev = build_v43_dev_arrays(master, fold, seq_len=p.seq_len)

    isolation = audit_fold_isolation(
        fold,
        train_master=split_master_for_fold(master, fold)[0],
        synthetic_update=qwen_update,
        downstream_train_meta=arrays["meta"],
        downstream_dev_meta=dev["meta"],
    )
    if not isolation["pass"]:
        raise RuntimeError(f"Stage 13 pre-fit isolation audit failed: {isolation}")

    stable = stable_config if stable_config is not None else load_frozen_model_configs(root)["tdr"]
    fit = fit_fn(
        arrays["X"], arrays["y"], arrays["prev"], arrays["peak"],
        dev["X"], dev["y"], dev["prev"], dev["peak"], stable,
        max_steps=steps, warmup_steps=warmup, val_every=val_every,
        patience_checks=p.patience_checks, batch_size=64, seed=run_key.seed,
        device=device, variant="full",
    )
    pred = np.asarray(predict_fn(fit, dev["X"], dev["prev"], dev["peak"]), dtype=float).reshape(-1)
    return {
        "fit": fit,
        "dev_metrics": evaluate_predictions(dev["y"], pred, dev["prev"]),
        "training_counts": arrays["counts"],
        "train_meta": arrays["meta"],
        "dev_meta": dev["meta"],
        "model_config": stable,
        "isolation_audit": isolation,
    }
