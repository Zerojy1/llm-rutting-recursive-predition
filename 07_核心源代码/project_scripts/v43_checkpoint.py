from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import torch

from v40_checkpoint import scaler_to_state, scaler_from_state
from v43_protocol import V43RunKey, V43Protocol


def _cpu_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def save_v43_checkpoint(path, run_key: V43RunKey, fit_result: dict, model_config, metadata: dict) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    cfg = asdict(model_config) if is_dataclass(model_config) else dict(model_config)
    payload = {
        "format_version": 1,
        "stage": "v43_stage13_structural_ood",
        "model_type": "tdr",
        "variant": "full",
        "run_key": asdict(run_key),
        "model_state_dict": _cpu_state_dict(fit_result["model"]),
        "model_config": cfg,
        "x_scaler": scaler_to_state(fit_result["x_scaler"]),
        "y_scaler": scaler_to_state(fit_result["y_scaler"]),
        "history": list(fit_result.get("history", [])),
        "best_val_rmse": fit_result.get("best_val_rmse"),
        "steps_completed": fit_result.get("steps_completed"),
        "parameter_count": fit_result.get("parameter_count"),
        "metadata": dict(metadata),
    }
    torch.save(payload, path)
    return path


def _load_payload(path):
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


def load_v43_checkpoint(path, device="auto") -> dict:
    payload = _load_payload(path)
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    from v31_tdr import TDRStableConfig, tdr
    stable = TDRStableConfig(**payload["model_config"])
    p = V43Protocol()
    tc = tdr.Config(
        input="", seq_len=p.seq_len, epochs=120, batch_size=64,
        lr=stable.lr, weight_decay=stable.weight_decay, d_model=stable.d_model,
        dropout=stable.dropout, run_xgb=False, run_dl_baselines=False,
        run_ablation=False, enable_residual_calibration=False,
        delta_loss_weight=.25, dual_state_loss_weight=.05, reversible_aux_weight=.03,
    )
    model, _, _, _ = tdr.build_neural_model("TDRPRSN", 23, tc, "full", dev)
    model.load_state_dict(payload["model_state_dict"])
    return {
        "model": model,
        "x_scaler": scaler_from_state(payload["x_scaler"]),
        "y_scaler": scaler_from_state(payload["y_scaler"]),
        "config": stable,
        "history": payload.get("history", []),
        "best_val_rmse": payload.get("best_val_rmse"),
        "steps_completed": payload.get("steps_completed"),
        "parameter_count": payload.get("parameter_count"),
        "metadata": payload.get("metadata", {}),
        "checkpoint_payload": payload,
    }
