from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from v31_downstream_data import build_real_downstream_frame, build_sequence_arrays_simple
from v31_experiment_engine import fixed_real_evaluation_arrays, evaluate_predictions
from v31_gru import GRUConfig, fit_gru_step_budget, predict_gru
from v31_tdr import fit_tdr_step_budget, predict_tdr
from v40_refit import load_frozen_model_configs
from v41_data import build_additive_training_arrays
from v42_protocol import V42Protocol, V42RunKey


def _full_real_arrays(master: pd.DataFrame, seq_len: int) -> dict:
    panel = master.loc[pd.to_numeric(master['Year'], errors='coerce').le(2019)].copy()
    frame = build_real_downstream_frame(panel)
    X, y, prev, peak, meta = build_sequence_arrays_simple(frame, seq_len=seq_len)
    years = pd.to_numeric(meta['Year'], errors='coerce')
    keep = years.le(2019).to_numpy()
    X = X[keep]; y = y[keep]; prev = prev[keep]; peak = peak[keep]
    meta = meta.loc[keep].reset_index(drop=True)
    years = pd.to_numeric(meta['Year'], errors='coerce')
    counts = {
        'n_historical_real': int(years.le(2018).sum()),
        'n_2019_real': int(years.eq(2019).sum()),
        'n_full_real': int(len(y)),
        'n_synthetic': 0,
        'n_total': int(len(y)),
    }
    return {'X': X, 'y': y, 'prev': prev, 'peak': peak, 'meta': meta, 'counts': counts}


def load_v42_training_arrays(root: str | Path, master: pd.DataFrame, run_key: V42RunKey) -> dict:
    root = Path(root)
    p = V42Protocol()
    if run_key.condition == 'full_real':
        arrays = _full_real_arrays(master, p.seq_len)
    elif run_key.condition == 'qwen_additive':
        path = root / 'data/v31_updates' / f'qwen_Q050_A{int(run_key.allocation)}.csv'
        if not path.exists():
            raise FileNotFoundError(path)
        arrays = build_additive_training_arrays(master, pd.read_csv(path), seq_len=p.seq_len, year=2019)
    else:
        raise ValueError(f'unsupported Stage 12 condition: {run_key.condition}')
    years = pd.to_numeric(arrays['meta']['Year'], errors='coerce')
    if years.isna().any() or int(years.max()) > 2019:
        raise RuntimeError('Stage 12 training arrays contain post-2019 targets')
    return arrays


def _gru_config_for_run(root: str | Path, run_key: V42RunKey) -> GRUConfig:
    if run_key.variant == 'Large':
        return GRUConfig(input_dim=23, hidden_dim=64, num_layers=3, dropout=0.15)
    if run_key.variant == 'frozen_gru':
        return load_frozen_model_configs(root)['gru']
    raise ValueError(f'unsupported GRU Stage 12 variant: {run_key.variant}')


def fit_v42_run(
    root: str | Path,
    master: pd.DataFrame,
    run_key: V42RunKey,
    max_steps: int | None = None,
    device: str = 'auto',
) -> dict:
    p = V42Protocol()
    steps = int(p.max_optimizer_steps if max_steps is None else max_steps)
    warmup = p.warmup_steps if steps >= p.warmup_steps else max(1, steps // 4)
    val_every = p.validation_every_steps if steps >= p.validation_every_steps else max(1, steps // 2)
    arrays = load_v42_training_arrays(root, master, run_key)
    Xv, yv, pv, pkv, _ = fixed_real_evaluation_arrays(master, seq_len=p.seq_len)['dev2020']
    if run_key.model == 'gru':
        cfg = _gru_config_for_run(root, run_key)
        fit = fit_gru_step_budget(
            arrays['X'], arrays['y'], arrays['prev'], Xv, yv, pv, cfg,
            max_steps=steps, warmup_steps=warmup, val_every=val_every,
            patience_checks=p.patience_checks, batch_size=64, seed=run_key.seed,
            device=device, learning_rate=8e-4, weight_decay=5e-4,
        )
        pred = predict_gru(fit, Xv, pv)
        model_config = cfg
    else:
        stable = load_frozen_model_configs(root)['tdr']
        fit = fit_tdr_step_budget(
            arrays['X'], arrays['y'], arrays['prev'], arrays['peak'],
            Xv, yv, pv, pkv, stable,
            max_steps=steps, warmup_steps=warmup, val_every=val_every,
            patience_checks=p.patience_checks, batch_size=64, seed=run_key.seed,
            device=device, variant=run_key.variant,
        )
        pred = predict_tdr(fit, Xv, pv, pkv)
        model_config = stable
    return {
        'fit': fit,
        'dev_metrics': evaluate_predictions(yv, pred, pv),
        'training_counts': arrays['counts'],
        'model_config': model_config,
    }
