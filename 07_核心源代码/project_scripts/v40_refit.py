from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from v31_gru import GRUConfig
from v31_tdr import TDRStableConfig
from v31_experiment_engine import (
    run_gru_condition, run_tdr_condition,
    run_gru_real_bootstrap, run_tdr_real_bootstrap,
    select_retained_indices,
)
from v40_protocol import RunKey, V40Protocol


def load_frozen_model_configs(root) -> dict:
    root = Path(root)
    p = root / 'reports/v31/tdr_stability/selected_tdr.json'
    if not p.exists():
        raise FileNotFoundError(p)
    selected = json.loads(p.read_text(encoding='utf-8'))
    cfg = selected.get('config', {})
    expected = {'d_model':48, 'dropout':0.15, 'weight_decay':0.0005, 'lr':0.0008}
    if selected.get('selected') != 'TDR-B' or selected.get('selection_uses_2020_only') is not True:
        raise RuntimeError('V4.0 requires frozen TDR-B selected on 2020 only')
    for k, v in expected.items():
        if abs(float(cfg.get(k, float('nan'))) - float(v)) > 1e-12:
            raise RuntimeError(f'frozen TDR-B config mismatch: {k}')
    return {
        'gru': GRUConfig(input_dim=23, hidden_dim=32, num_layers=2, dropout=0.15),
        'tdr': TDRStableConfig(**expected),
    }


def _qtag(q: float) -> str:
    return f'Q{int(round(float(q)*100)):03d}'


def _allocation_seed(run_key: RunKey, p: V40Protocol | None = None) -> int:
    p = p or V40Protocol()
    if run_key.allocation < 1 or run_key.allocation > len(p.allocation_seeds):
        raise ValueError(f'allocation {run_key.allocation} has no frozen allocation seed')
    return int(p.allocation_seeds[run_key.allocation - 1])


def load_update_for_run(root, master: pd.DataFrame, run_key: RunKey, protocol: V40Protocol | None = None):
    root = Path(root); p = protocol or V40Protocol()
    if run_key.method == 'full_real':
        return master[(pd.to_numeric(master['Year']) == 2019) & (pd.to_numeric(master['Rutting_Observed']).fillna(0).astype(int) == 1)].copy()
    if run_key.method == 'historical_only':
        return master.iloc[0:0].copy()
    if run_key.method == 'real_bootstrap':
        return select_retained_indices(master, run_key.q, _allocation_seed(run_key, p), year=2019)
    if run_key.method in {'qwen', 'timegan', 'timeweaver'}:
        path = root / 'data/v31_updates' / f'{run_key.method}_{_qtag(run_key.q)}_A{run_key.allocation}.csv'
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_csv(path)
    raise ValueError(f'unsupported method: {run_key.method}')


def fit_frozen_run(root, master: pd.DataFrame, run_key: RunKey, max_steps: int | None = None, device='auto') -> dict:
    p = V40Protocol(); cfgs = load_frozen_model_configs(root)
    steps = int(p.max_optimizer_steps if max_steps is None else max_steps)
    common = dict(max_steps=steps, warmup_steps=p.warmup_steps if steps >= p.warmup_steps else max(1, steps//4),
                  val_every=p.validation_every_steps if steps >= p.validation_every_steps else max(1, steps//2),
                  patience_checks=p.patience_checks, seq_len=p.seq_len)
    update = load_update_for_run(root, master, run_key, p)
    if run_key.model == 'gru':
        if run_key.method == 'real_bootstrap':
            return run_gru_real_bootstrap(master, update, cfgs['gru'], seed=run_key.seed,
                                          allocation_seed=_allocation_seed(run_key, p), **common)
        return run_gru_condition(master, update, cfgs['gru'], seed=run_key.seed, **common)
    if run_key.model == 'tdr':
        if run_key.method == 'real_bootstrap':
            return run_tdr_real_bootstrap(master, update, cfgs['tdr'], seed=run_key.seed,
                                          allocation_seed=_allocation_seed(run_key, p), **common)
        return run_tdr_condition(master, update, cfgs['tdr'], seed=run_key.seed, **common)
    raise ValueError(f'unsupported model: {run_key.model}')



def validate_full_real_reference_consistency(root, model: str, seed: int) -> Path:
    """Require the five duplicated V3.2 full-real allocation references to agree exactly."""
    root=Path(root); base=root/'reports/v31/substitution_runs'
    paths=[base/f'{model}_full_real_Q000_A{a}_S{int(seed)}_pred2021.csv' for a in range(1,6)]
    missing=[str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError('missing V3.2 full-real reference allocation(s): '+', '.join(missing))
    keys=['STR_name','Observation_Index']
    ref=pd.read_csv(paths[0]).sort_values(keys,kind='mergesort').reset_index(drop=True)
    needed=set(keys+['y_true','y_pred'])
    if not needed.issubset(ref.columns):
        raise RuntimeError(f'V3.2 full-real reference missing columns: {sorted(needed-set(ref.columns))}')
    for path in paths[1:]:
        cur=pd.read_csv(path).sort_values(keys,kind='mergesort').reset_index(drop=True)
        if len(cur)!=len(ref) or not needed.issubset(cur.columns):
            raise RuntimeError(f'V3.2 full-real allocation mismatch: {path.name}')
        if not ref[keys].astype(str).equals(cur[keys].astype(str)):
            raise RuntimeError(f'V3.2 full-real key mismatch: {path.name}')
        for col in ('y_true','y_pred'):
            a=pd.to_numeric(ref[col],errors='coerce').to_numpy(float)
            b=pd.to_numeric(cur[col],errors='coerce').to_numpy(float)
            if not __import__('numpy').allclose(a,b,rtol=0.0,atol=1e-12,equal_nan=True):
                raise RuntimeError(f'V3.2 full-real prediction mismatch: {path.name} column={col}')
    return paths[0]

def find_v31_reference_prediction(root, run_key: RunKey) -> Path:
    root = Path(root)
    if run_key.method == 'full_real' and run_key.allocation == 0:
        return validate_full_real_reference_consistency(root, run_key.model, run_key.seed)
    allocation = run_key.allocation
    stem = f'{run_key.model}_{run_key.method}_{_qtag(run_key.q)}_A{allocation}_S{run_key.seed}'
    path = root / 'reports/v31/substitution_runs' / f'{stem}_pred2021.csv'
    if not path.exists():
        raise FileNotFoundError(path)
    return path
