from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from v31_experiment_engine import fixed_real_evaluation_arrays
from v31_gru import predict_gru
from v31_tdr import predict_tdr


@dataclass(frozen=True)
class FixedOrigin:
    structure: str
    origin_observation_index: int
    y_t: float
    peak_t: float
    X_future: np.ndarray
    y_future: np.ndarray
    meta_future: pd.DataFrame
    future_positions: tuple[int, ...]


def build_end2020_origins(master: pd.DataFrame, seq_len: int = 6) -> list[FixedOrigin]:
    fixed = fixed_real_evaluation_arrays(master, seq_len=int(seq_len))
    _, y2020, _, peak2020, meta2020 = fixed["dev2020"]
    X2021, y2021, _, _, meta2021 = fixed["eval2021"]
    meta2020 = meta2020.reset_index(drop=True)
    meta2021 = meta2021.reset_index(drop=True)
    origins = []
    structures = sorted(
        set(meta2020["STR_name"].astype(str)) & set(meta2021["STR_name"].astype(str))
    )
    for structure in structures:
        anchor_positions = np.flatnonzero(meta2020["STR_name"].astype(str).eq(structure))
        future_positions = np.flatnonzero(meta2021["STR_name"].astype(str).eq(structure))
        if len(anchor_positions) == 0 or len(future_positions) == 0:
            continue
        anchor_order = pd.to_numeric(
            meta2020.iloc[anchor_positions]["Observation_Order"], errors="coerce"
        ).to_numpy()
        anchor_position = int(anchor_positions[int(np.nanargmax(anchor_order))])
        future_order = pd.to_numeric(
            meta2021.iloc[future_positions]["Observation_Order"], errors="coerce"
        ).to_numpy()
        future_positions = future_positions[np.argsort(future_order, kind="mergesort")]
        y_t = float(np.asarray(y2020)[anchor_position])
        prior_peak = float(np.asarray(peak2020)[anchor_position].reshape(-1)[0])
        peak_t = max(y_t, prior_peak) if np.isfinite(prior_peak) else y_t
        origins.append(
            FixedOrigin(
                structure=structure,
                origin_observation_index=int(meta2020.iloc[anchor_position]["Observation_Order"]),
                y_t=y_t,
                peak_t=peak_t,
                X_future=np.asarray(X2021)[future_positions],
                y_future=np.asarray(y2021)[future_positions],
                meta_future=meta2021.iloc[future_positions].reset_index(drop=True),
                future_positions=tuple(int(position) for position in future_positions),
            )
        )
    return origins


def forecast_end2020_fixed_origin(
    fit,
    master: pd.DataFrame,
    model: str,
    run_key=None,
    predictor=None,
    seq_len: int = 6,
) -> pd.DataFrame:
    if model not in {"gru", "tdr"}:
        raise ValueError(f"unsupported model: {model}")
    prediction_function = predictor or (predict_gru if model == "gru" else predict_tdr)
    rows = []
    for origin in build_end2020_origins(master, seq_len=int(seq_len)):
        state = float(origin.y_t)
        peak = float(origin.peak_t)
        requested_horizon = int(len(origin.future_positions))
        for step in range(1, requested_horizon + 1):
            X = origin.X_future[step - 1 : step]
            previous = np.asarray([[state]], dtype=np.float32)
            if model == "gru":
                prediction = float(
                    np.asarray(prediction_function(fit, X, previous)).reshape(-1)[0]
                )
            else:
                prediction = float(
                    np.asarray(
                        prediction_function(
                            fit, X, previous, np.asarray([[peak]], dtype=np.float32)
                        )
                    ).reshape(-1)[0]
                )
            target_meta = origin.meta_future.iloc[step - 1]
            rows.append(
                {
                    "STR_name": origin.structure,
                    "origin_observation_index": origin.origin_observation_index,
                    "target_observation_index": int(target_meta["Observation_Order"]),
                    "requested_horizon": requested_horizon,
                    "step_h": step,
                    "y_origin": origin.y_t,
                    "y_true": float(origin.y_future[step - 1]),
                    "y_pred": prediction,
                    "persistence_pred": origin.y_t,
                }
            )
            state = prediction
            if model == "tdr":
                peak = max(peak, prediction)
    out = pd.DataFrame(rows)
    if len(out) and run_key is not None:
        out.insert(0, "seed", int(run_key.seed))
        out.insert(0, "allocation", int(run_key.allocation))
        out.insert(0, "synthetic_fraction", float(run_key.synthetic_fraction))
        out.insert(0, "real_fraction", float(run_key.real_fraction))
        out.insert(0, "condition", str(run_key.condition))
        out.insert(0, "model", str(run_key.model))
    return out
