from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


def horizon_error_accumulation(frame: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Fit RMSE(H)=a+bH for each run and compute normalized H1→Hmax growth."""
    f = frame.copy()
    needed = {'requested_horizon', 'trajectory_rmse'}
    missing = needed - set(f.columns)
    if missing:
        raise KeyError(f'missing columns: {sorted(missing)}')
    if group_cols is None:
        group_cols = [c for c in ['model', 'condition', 'method', 'variant', 'allocation', 'seed'] if c in f.columns]
    rows = []
    groups = f.groupby(group_cols, dropna=False, sort=True) if group_cols else [((), f)]
    for keys, g in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        g = g[['requested_horizon', 'trajectory_rmse']].dropna().sort_values('requested_horizon')
        x = pd.to_numeric(g['requested_horizon'], errors='coerce').to_numpy(float)
        y = pd.to_numeric(g['trajectory_rmse'], errors='coerce').to_numpy(float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]; y = y[mask]
        if len(x) < 2:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        hmin = int(x.min()); hmax = int(x.max())
        ymin = float(y[np.argmin(x)]); ymax = float(y[np.argmax(x)])
        rec.update({
            'horizon_min': hmin,
            'horizon_max': hmax,
            'rmse_hmin': ymin,
            'rmse_hmax': ymax,
            'rmse_per_horizon_step_slope': float(slope),
            'rmse_intercept': float(intercept),
            'normalized_h1_h12_growth': float((ymax - ymin) / ymin) if ymin != 0 else np.nan,
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def paired_horizon_slope_change(full_real_metrics: pd.DataFrame, additive_metrics: pd.DataFrame) -> pd.DataFrame:
    """Pair each additive allocation×seed with the same-seed full-real baseline."""
    base = horizon_error_accumulation(
        full_real_metrics,
        [c for c in ['model', 'variant', 'condition', 'method', 'allocation', 'seed'] if c in full_real_metrics.columns],
    )
    aug = horizon_error_accumulation(
        additive_metrics,
        [c for c in ['model', 'variant', 'condition', 'method', 'allocation', 'seed'] if c in additive_metrics.columns],
    )
    rows = []
    for _, ar in aug.iterrows():
        seed = int(ar['seed']) if 'seed' in ar else None
        model = ar.get('model', None)
        variant = ar.get('variant', None)
        cand = base.copy()
        if seed is not None and 'seed' in cand:
            cand = cand[pd.to_numeric(cand.seed).eq(seed)]
        if model is not None and 'model' in cand:
            cand = cand[cand.model.astype(str).eq(str(model))]
        if variant is not None and 'variant' in cand:
            cand = cand[cand.variant.astype(str).eq(str(variant))]
        if cand.empty:
            continue
        br = cand.iloc[0]
        sb = float(br['rmse_per_horizon_step_slope'])
        sa = float(ar['rmse_per_horizon_step_slope'])
        gb = float(br['normalized_h1_h12_growth'])
        ga = float(ar['normalized_h1_h12_growth'])
        rec = {
            'model': model,
            'variant': variant,
            'allocation': int(ar.get('allocation', 0)),
            'seed': seed,
            'baseline_slope': sb,
            'additive_slope': sa,
            'relative_slope_change': float((sa - sb) / sb) if sb != 0 else np.nan,
            'baseline_normalized_growth': gb,
            'additive_normalized_growth': ga,
            'normalized_growth_change': float(ga - gb),
        }
        rows.append(rec)
    return pd.DataFrame(rows)


def audit_existing_model_selection(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize what was actually tuned/selected before Stage 12 without rewriting history."""
    root = Path(root)
    tdr_dir = root / 'reports/v31/tdr_stability'
    cap_dir = root / 'reports/v31/capacity'
    selected = json.loads((tdr_dir / 'selected_tdr.json').read_text(encoding='utf-8'))
    tdr_candidates = pd.read_csv(tdr_dir / 'candidate_summary.csv')
    cap = pd.read_csv(cap_dir / 'capacity_results.csv')
    capacities = sorted(cap['capacity'].dropna().astype(str).unique().tolist())
    # Existing capacity ladder was a later sensitivity analysis (Stage 07), not the source of the frozen GRU choice.
    rows = [
        {
            'model_family': 'TDR',
            'frozen_choice': str(selected.get('selected')),
            'candidate_count': int(tdr_candidates['name'].nunique()),
            'selection_dataset': '2020 development',
            'test_used_for_selection': bool(not selected.get('selection_uses_2020_only', False)),
            'capacity_ladder_role': 'pre-frozen TDR stability selection',
            'note': 'TDR-B selection artifact explicitly states selection_uses_2020_only=true.',
        },
        {
            'model_family': 'GRU',
            'frozen_choice': 'Medium (hidden=32, layers=2, dropout=0.15)',
            'candidate_count': int(len(capacities)),
            'selection_dataset': 'frozen before Stage 07 capacity ladder; provenance does not document an equal-grid search',
            'test_used_for_selection': False,
            'capacity_ladder_role': 'post_hoc_sensitivity_not_model_selection',
            'note': 'Stage 07 later tested Tiny/Small/Medium/Medium+/Large/XL; it must not be reinterpreted as prospective selection.',
        },
    ]
    audit = pd.DataFrame(rows)
    cap_summary = (
        cap.groupby(['capacity', 'synthetic_fraction'], as_index=False)
        .agg(
            mean_rmse2021=('rmse2021', 'mean'),
            sd_rmse2021=('rmse2021', 'std'),
            n=('rmse2021', 'size'),
            params=('params', 'first'),
        )
    )
    return audit, cap_summary


def summarize_slope_change(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    groups = [c for c in ['model', 'variant'] if c in paired.columns]
    return (
        paired.groupby(groups, dropna=False, as_index=False)
        .agg(
            n_pairs=('relative_slope_change', 'size'),
            mean_relative_slope_change=('relative_slope_change', 'mean'),
            sd_relative_slope_change=('relative_slope_change', 'std'),
            mean_normalized_growth_change=('normalized_growth_change', 'mean'),
        )
    )
