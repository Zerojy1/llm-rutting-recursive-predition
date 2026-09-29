from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from v40_protocol import V40Protocol

KEYS = ['STR_name', 'Observation_Index']


def audit_one_step_reproduction(new_pred: pd.DataFrame, old_pred: pd.DataFrame, protocol: V40Protocol | None = None) -> dict:
    p = protocol or V40Protocol()
    n = new_pred.copy(); o = old_pred.copy()
    for f in (n, o):
        f['STR_name'] = f['STR_name'].astype(str)
        f['Observation_Index'] = pd.to_numeric(f['Observation_Index'], errors='coerce').astype('Int64')
    nk = set(map(tuple, n[KEYS].dropna().to_numpy()))
    ok = set(map(tuple, o[KEYS].dropna().to_numpy()))
    keys_match = nk == ok and len(n) == len(o)
    out = {
        'keys_match': bool(keys_match),
        'new_rows': int(len(n)),
        'old_rows': int(len(o)),
        'rmse_new': None,
        'rmse_old': None,
        'absolute_rmse_difference_mm': None,
        'prediction_correlation': None,
        'mean_absolute_prediction_difference_mm': None,
        'all_predictions_finite': False,
        'status': 'FAIL',
    }
    if not keys_match:
        out['missing_in_new'] = int(len(ok - nk)); out['extra_in_new'] = int(len(nk - ok))
        return out
    m = o[KEYS + ['y_true', 'y_pred']].merge(
        n[KEYS + ['y_true', 'y_pred']], on=KEYS, how='inner', suffixes=('_old', '_new'), validate='one_to_one'
    )
    vals = m[['y_true_old', 'y_true_new', 'y_pred_old', 'y_pred_new']].to_numpy(float)
    finite = bool(np.isfinite(vals).all())
    if not finite:
        return {**out, 'all_predictions_finite': False}
    y_old = m['y_true_old'].to_numpy(float); y_new = m['y_true_new'].to_numpy(float)
    target_max_abs_diff = float(np.max(np.abs(y_old-y_new))) if len(y_old) else 0.0
    # Evaluation arrays are float32 while archived V3.2 CSV targets are decimal/float64.
    # A 1e-6 mm tolerance accepts only representation round-trip noise, not target changes.
    if not np.allclose(y_old, y_new, atol=1e-6, rtol=0):
        out['target_values_match'] = False
        out['max_target_absolute_difference_mm'] = target_max_abs_diff
        return out
    out['max_target_absolute_difference_mm'] = target_max_abs_diff
    po = m['y_pred_old'].to_numpy(float); pn = m['y_pred_new'].to_numpy(float)
    ro = float(np.sqrt(mean_squared_error(y_old, po))); rn = float(np.sqrt(mean_squared_error(y_new, pn)))
    if len(po) > 1 and np.std(po) > 0 and np.std(pn) > 0:
        corr = float(np.corrcoef(po, pn)[0, 1])
    else:
        corr = 1.0 if np.allclose(po, pn, atol=1e-12) else np.nan
    diff = float(np.mean(np.abs(pn - po)))
    rmse_diff = float(abs(rn - ro))
    passed = finite and rmse_diff <= p.reproduction_rmse_tolerance_mm and np.isfinite(corr) and corr >= p.reproduction_prediction_corr_min
    out.update({
        'rmse_new': rn,
        'rmse_old': ro,
        'absolute_rmse_difference_mm': rmse_diff,
        'prediction_correlation': corr,
        'mean_absolute_prediction_difference_mm': diff,
        'all_predictions_finite': finite,
        'target_values_match': True,
        'status': 'PASS' if passed else 'FAIL',
    })
    return out


def run_openloop_leakage_audit(eval_arrays, model='gru', horizon=3) -> dict:
    """Behaviorally verify the rollout plumbing with deterministic probe predictors.

    The probe is intentionally simple: it depends on the legal origin state and on the
    scenario-conditioned exogenous feature path, but never on future target rutting.
    """
    from dataclasses import replace
    from v40_openloop import EvalArrays, iter_forecast_origins, forecast_gru_origin, forecast_tdr_origin

    origin = next(iter_forecast_origins(eval_arrays, int(horizon)), None)
    if origin is None:
        raise ValueError('no eligible origin for leakage audit')

    def gru_probe(_fit, X, prev):
        return np.asarray(prev, float).reshape(-1) + 0.01 * np.asarray(X, float)[:, -1, 0]

    def tdr_probe(_fit, X, prev, peak):
        return np.asarray(prev, float).reshape(-1) + 0.01 * np.asarray(X, float)[:, -1, 0]

    def preds(o):
        rows = forecast_gru_origin({}, o, predictor=gru_probe) if model == 'gru' else forecast_tdr_origin({}, o, predictor=tdr_probe)
        return np.array([r['y_pred'] for r in rows], dtype=float)

    base = preds(origin)

    y2 = np.asarray(eval_arrays.y, dtype=float).copy()
    y2[list(origin.future_positions)] += 1000.0
    target_perturbed = EvalArrays(np.asarray(eval_arrays.X).copy(), y2, np.asarray(eval_arrays.prev).copy(),
                                  np.asarray(eval_arrays.peak).copy(), eval_arrays.meta.copy())
    o2 = replace(origin, arrays=target_perturbed)
    future_target_invariance = bool(np.allclose(base, preds(o2), atol=1e-12, rtol=0))

    anchor_peak = max(float(origin.peak_t), float(origin.y_t) + 1.0)
    oa = replace(origin, y_t=float(origin.y_t) + 1.0, peak_t=anchor_peak)
    anchor_sensitivity = bool(not np.allclose(base, preds(oa), atol=1e-12, rtol=0))

    X3 = np.asarray(eval_arrays.X).copy()
    for j in origin.future_positions:
        X3[j, -1, 0] += 10.0
    exo_perturbed = EvalArrays(X3, np.asarray(eval_arrays.y).copy(), np.asarray(eval_arrays.prev).copy(),
                               np.asarray(eval_arrays.peak).copy(), eval_arrays.meta.copy())
    oe = replace(origin, arrays=exo_perturbed)
    exogenous_path_verified = bool(not np.allclose(base, preds(oe), atol=1e-12, rtol=0))

    status = 'PASS' if future_target_invariance and anchor_sensitivity and exogenous_path_verified else 'FAIL'
    return {
        'future_target_invariance': future_target_invariance,
        'anchor_sensitivity': anchor_sensitivity,
        'exogenous_path_verified': exogenous_path_verified,
        'status': status,
    }
