from __future__ import annotations

import numpy as np
import pandas as pd

from v31_downstream_data import build_real_downstream_frame, build_sequence_arrays_simple
from v31_experiment_engine import training_frame_with_update
from v31_substitution import _stratified_take


KEY_COLUMNS = ["STR_name", "Observation_Index"]


def _observed_real_pool(master: pd.DataFrame, year: int) -> pd.DataFrame:
    required = set(KEY_COLUMNS + ["Year", "Rutting_Observed", "Rutting_mm"])
    missing = required - set(master.columns)
    if missing:
        raise KeyError(f"master missing columns: {sorted(missing)}")
    mask = (
        pd.to_numeric(master["Year"], errors="coerce").eq(int(year))
        & pd.to_numeric(master["Rutting_Observed"], errors="coerce").fillna(0).astype(int).eq(1)
        & np.isfinite(pd.to_numeric(master["Rutting_mm"], errors="coerce"))
    )
    pool = master.loc[mask].copy()
    if pool.empty:
        raise ValueError(f"no observed real update states for year {year}")
    if pool.duplicated(KEY_COLUMNS).any():
        raise ValueError("duplicate real update keys")
    return pool


def build_reduced_real_update(
    master: pd.DataFrame,
    real_fraction: float,
    allocation_seed: int,
    year: int = 2019,
) -> pd.DataFrame:
    real_fraction = float(real_fraction)
    if not 0.0 < real_fraction <= 1.0:
        raise ValueError("real_fraction must be in (0, 1]")
    pool = _observed_real_pool(master, year)
    expected = int(round(len(pool) * real_fraction))
    counts = pool.groupby("STR_name", sort=True).size()
    raw = counts / counts.sum() * expected
    quotas = np.floor(raw).astype(int)
    remainder = expected - int(quotas.sum())
    if remainder:
        rng = np.random.default_rng(int(allocation_seed))
        tie = pd.Series(rng.random(len(counts)), index=counts.index)
        order = pd.DataFrame({"fraction": raw - quotas, "tie": tie}).sort_values(
            ["fraction", "tie"], ascending=[False, True]
        )
        for structure in order.index[:remainder]:
            quotas.loc[structure] += 1
    selected = []
    for offset, (structure, group) in enumerate(pool.groupby("STR_name", sort=True)):
        quota = int(quotas.loc[structure])
        if quota:
            selected.append(
                _stratified_take(group, quota, int(allocation_seed) + offset * 1009)
            )
    out = pd.concat(selected, ignore_index=True) if selected else pool.iloc[0:0].copy()
    if len(out) != expected:
        raise RuntimeError(
            f"reduced-real allocation size mismatch: expected {expected}, got {len(out)}"
        )
    out["source_type"] = "real_reduced"
    return out.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)


def _arrays_for_years(frame: pd.DataFrame, years, seq_len: int):
    X, y, prev, peak, meta = build_sequence_arrays_simple(frame, seq_len=seq_len)
    mask = pd.to_numeric(meta["Year"], errors="coerce").isin(list(years)).to_numpy()
    return X[mask], y[mask], prev[mask], peak[mask], meta.loc[mask].reset_index(drop=True)


def _validate_synthetic_update(qwen_update: pd.DataFrame) -> pd.DataFrame:
    required = set(
        KEY_COLUMNS
        + [
            "source_type",
            "Rutting_Observed",
            "Rutting_mm",
            "Previous_Rutting_mm",
            "Previous_Peak_mm",
        ]
    )
    missing = required - set(qwen_update.columns)
    if missing:
        raise KeyError(f"Qwen update missing columns: {sorted(missing)}")
    synthetic = qwen_update.loc[
        qwen_update["source_type"].astype(str).str.lower().eq("synthetic")
    ].copy()
    if synthetic.empty:
        raise ValueError("Qwen update contains no synthetic supervision")
    if synthetic.duplicated(KEY_COLUMNS).any():
        raise ValueError("duplicate synthetic update keys")
    finite_columns = ["Rutting_mm", "Previous_Rutting_mm", "Previous_Peak_mm"]
    finite = np.column_stack(
        [
            np.isfinite(pd.to_numeric(synthetic[column], errors="coerce").to_numpy())
            for column in finite_columns
        ]
    )
    if not finite.all():
        raise ValueError("synthetic supervision must contain finite rutting states")
    if not pd.to_numeric(synthetic["Rutting_Observed"], errors="coerce").fillna(0).astype(int).eq(1).all():
        raise ValueError("synthetic supervision must be marked observed")
    return synthetic.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)


def build_additive_training_arrays(
    master: pd.DataFrame,
    qwen_update: pd.DataFrame,
    seq_len: int = 6,
    year: int = 2019,
) -> dict:
    synthetic = _validate_synthetic_update(qwen_update)

    real_panel = master.loc[pd.to_numeric(master["Year"], errors="coerce").le(int(year))].copy()
    real_frame = build_real_downstream_frame(real_panel)
    X_real, y_real, prev_real, peak_real, meta_real = _arrays_for_years(
        real_frame, range(2016, int(year) + 1), int(seq_len)
    )
    if len(y_real) == 0:
        raise ValueError("full-real training arrays are empty")
    meta_real = meta_real.copy()
    meta_real["training_source"] = "full_real"

    synthetic_panel = training_frame_with_update(master, synthetic, year=int(year))
    synthetic_frame = build_real_downstream_frame(synthetic_panel)
    X_syn, y_syn, prev_syn, peak_syn, meta_syn = _arrays_for_years(
        synthetic_frame, [int(year)], int(seq_len)
    )
    if len(y_syn) != len(synthetic):
        raise ValueError(
            "synthetic supervision did not produce one finite training array per input state"
        )
    arrays = (X_syn, y_syn, prev_syn, peak_syn)
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("synthetic training arrays must be finite")
    meta_syn = meta_syn.copy()
    meta_syn["training_source"] = "synthetic"

    X = np.concatenate([X_real, X_syn], axis=0)
    y = np.concatenate([y_real, y_syn], axis=0)
    prev = np.concatenate([prev_real, prev_syn], axis=0)
    peak = np.concatenate([peak_real, peak_syn], axis=0)
    meta = pd.concat([meta_real, meta_syn], ignore_index=True)
    counts = {
        "n_historical_real": int(pd.to_numeric(meta_real["Year"]).le(2018).sum()),
        "n_2019_real": int(pd.to_numeric(meta_real["Year"]).eq(int(year)).sum()),
        "n_full_real": int(len(y_real)),
        "n_synthetic": int(len(y_syn)),
        "n_total": int(len(y)),
    }
    return {
        "X": X,
        "y": y,
        "prev": prev,
        "peak": peak,
        "meta": meta,
        "counts": counts,
    }
