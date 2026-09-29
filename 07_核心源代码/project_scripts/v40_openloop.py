from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from v31_experiment_engine import fixed_real_evaluation_arrays
from v31_gru import predict_gru
from v31_tdr import predict_tdr


@dataclass
class EvalArrays:
    X: np.ndarray
    y: np.ndarray
    prev: np.ndarray
    peak: np.ndarray
    meta: pd.DataFrame

    def __post_init__(self):
        n = len(self.y)
        if not (len(self.X) == len(self.prev) == len(self.peak) == len(self.meta) == n):
            raise ValueError('evaluation arrays length mismatch')


@dataclass(frozen=True)
class ForecastOrigin:
    arrays: EvalArrays
    structure: str
    origin_pos: int
    origin_observation_index: int
    y_t: float
    peak_t: float
    future_positions: tuple[int, ...]
    requested_horizon: int


def build_eval_arrays(master: pd.DataFrame, seq_len: int = 6) -> EvalArrays:
    X, y, prev, peak, meta = fixed_real_evaluation_arrays(master, seq_len=seq_len)['eval2021']
    meta = meta.reset_index(drop=True).copy()
    return EvalArrays(np.asarray(X, np.float32), np.asarray(y, np.float32), np.asarray(prev, np.float32),
                      np.asarray(peak, np.float32), meta)


def iter_forecast_origins(eval_arrays: EvalArrays, horizon: int):
    if int(horizon) <= 0:
        raise ValueError('horizon must be positive')
    meta = eval_arrays.meta.reset_index(drop=True)
    for structure, g in meta.assign(_pos=np.arange(len(meta))).groupby('STR_name', sort=False):
        positions = g.sort_values('Observation_Order', kind='mergesort')['_pos'].astype(int).to_list()
        for k in range(0, len(positions) - int(horizon)):
            pos = positions[k]
            future = tuple(positions[k+1:k+1+int(horizon)])
            y_t = float(eval_arrays.y[pos])
            prior_peak = float(np.asarray(eval_arrays.peak[pos]).reshape(-1)[0])
            peak_t = max(y_t, prior_peak) if np.isfinite(prior_peak) else y_t
            yield ForecastOrigin(
                arrays=eval_arrays,
                structure=str(structure),
                origin_pos=pos,
                origin_observation_index=int(meta.iloc[pos]['Observation_Order']),
                y_t=y_t,
                peak_t=peak_t,
                future_positions=future,
                requested_horizon=int(horizon),
            )


def _row(origin: ForecastOrigin, j: int, h: int, pred: float) -> dict:
    meta = origin.arrays.meta.iloc[j]
    return {
        'STR_name': origin.structure,
        'origin_observation_index': int(origin.origin_observation_index),
        'target_observation_index': int(meta['Observation_Order']),
        'requested_horizon': int(origin.requested_horizon),
        'step_h': int(h),
        'y_origin': float(origin.y_t),
        'y_true': float(origin.arrays.y[j]),
        'y_pred': float(pred),
        'persistence_pred': float(origin.y_t),
    }


def forecast_gru_origin(fit, origin: ForecastOrigin, predictor=predict_gru) -> list[dict]:
    state = float(origin.y_t); rows = []
    for h, j in enumerate(origin.future_positions, 1):
        pred = float(predictor(fit, origin.arrays.X[j:j+1], np.array([[state]], dtype=np.float32))[0])
        rows.append(_row(origin, j, h, pred))
        state = pred
    return rows


def forecast_tdr_origin(fit, origin: ForecastOrigin, predictor=predict_tdr) -> list[dict]:
    state = float(origin.y_t); peak = float(origin.peak_t); rows = []
    for h, j in enumerate(origin.future_positions, 1):
        pred = float(predictor(
            fit, origin.arrays.X[j:j+1], np.array([[state]], dtype=np.float32),
            np.array([[peak]], dtype=np.float32)
        )[0])
        rows.append(_row(origin, j, h, pred))
        state = pred; peak = max(peak, pred)
    return rows


def forecast_all_origins(
    fit, eval_arrays: EvalArrays, horizon: int, model: str, run_key=None, predictor=None
) -> pd.DataFrame:
    """Forecast every eligible origin, batching model calls by open-loop step.

    All origins are complete for ``horizon`` by construction.  At each recursive
    step the model therefore receives one batch containing every origin, while
    the rutting-dependent state is carried forward only from predictions.  This
    preserves the exact open-loop semantics of the per-origin helpers but avoids
    one neural-network call per origin per step.
    """
    origins = list(iter_forecast_origins(eval_arrays, horizon))
    if not origins:
        return pd.DataFrame()

    if model == 'gru':
        pred_fn = predictor or predict_gru
    elif model == 'tdr':
        pred_fn = predictor or predict_tdr
    else:
        raise ValueError(f'unsupported model: {model}')

    states = np.asarray([o.y_t for o in origins], dtype=np.float32)
    peaks = np.asarray([o.peak_t for o in origins], dtype=np.float32)
    rows_by_origin: list[list[dict]] = [[] for _ in origins]

    for h in range(1, int(horizon) + 1):
        positions = np.asarray([o.future_positions[h - 1] for o in origins], dtype=int)
        X_batch = eval_arrays.X[positions]
        prev_batch = states.reshape(-1, 1)
        if model == 'gru':
            preds = pred_fn(fit, X_batch, prev_batch)
        else:
            preds = pred_fn(fit, X_batch, prev_batch, peaks.reshape(-1, 1))
        preds = np.asarray(preds, dtype=float).reshape(-1)
        if len(preds) != len(origins):
            raise ValueError('predictor returned a batch with unexpected length')

        for i, (origin, j, pred) in enumerate(zip(origins, positions, preds)):
            rows_by_origin[i].append(_row(origin, int(j), h, float(pred)))

        states = preds.astype(np.float32, copy=False)
        if model == 'tdr':
            peaks = np.maximum(peaks, states)

    rows = [row for origin_rows in rows_by_origin for row in origin_rows]
    out = pd.DataFrame(rows)
    if len(out) and run_key is not None:
        out.insert(0, 'seed', int(run_key.seed))
        out.insert(0, 'allocation', int(run_key.allocation))
        out.insert(0, 'synthetic_fraction', float(run_key.q))
        out.insert(0, 'method', str(run_key.method))
        out.insert(0, 'model', str(run_key.model))
    return out
