from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import torch

from v40_checkpoint import scaler_to_state, scaler_from_state
from v42_protocol import V42RunKey


def _cpu_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def save_v42_checkpoint(path, run_key: V42RunKey, fit_result: dict, model_config, metadata: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = asdict(model_config) if is_dataclass(model_config) else dict(model_config)
    payload = {
        'format_version': 1,
        'stage': 'v42_stage12_posthoc',
        'model_type': str(run_key.model),
        'variant': str(run_key.variant),
        'run_key': asdict(run_key),
        'model_state_dict': _cpu_state_dict(fit_result['model']),
        'model_config': cfg,
        'x_scaler': scaler_to_state(fit_result['x_scaler']),
        'history': list(fit_result.get('history', [])),
        'best_val_rmse': fit_result.get('best_val_rmse'),
        'steps_completed': fit_result.get('steps_completed'),
        'parameter_count': fit_result.get('parameter_count'),
        'metadata': dict(metadata),
    }
    if run_key.model == 'tdr':
        payload['y_scaler'] = scaler_to_state(fit_result['y_scaler'])
    torch.save(payload, path)
    return path


def _load_payload(path: str | Path) -> dict:
    try:
        return torch.load(Path(path), map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location='cpu')


def _device(device='auto'):
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def load_v42_checkpoint(path, device='auto') -> dict:
    payload = _load_payload(path)
    dev = _device(device)
    model_type = str(payload['model_type'])
    variant = str(payload['variant'])
    cfg = payload['model_config']
    if model_type == 'gru':
        from v31_gru import GRUConfig, RuttingGRU
        gcfg = GRUConfig(**cfg)
        model = RuttingGRU(gcfg).to(dev)
        model.load_state_dict(payload['model_state_dict'])
        result = {
            'model': model,
            'x_scaler': scaler_from_state(payload['x_scaler']),
            'model_config': gcfg,
        }
    elif model_type == 'tdr':
        from v31_tdr import TDRStableConfig, tdr
        stable = TDRStableConfig(**cfg)
        tc = tdr.Config(
            input='', seq_len=6, epochs=120, batch_size=64,
            lr=stable.lr, weight_decay=stable.weight_decay,
            d_model=stable.d_model, dropout=stable.dropout,
            run_xgb=False, run_dl_baselines=False, run_ablation=False,
            enable_residual_calibration=False,
            delta_loss_weight=.25, dual_state_loss_weight=.05,
            reversible_aux_weight=.03,
        )
        model, _, _, _ = tdr.build_neural_model('TDRPRSN', 23, tc, variant, dev)
        model.load_state_dict(payload['model_state_dict'])
        result = {
            'model': model,
            'x_scaler': scaler_from_state(payload['x_scaler']),
            'y_scaler': scaler_from_state(payload['y_scaler']),
            'config': stable,
        }
    else:
        raise ValueError(f'unsupported Stage 12 checkpoint model: {model_type}')
    result.update({
        'variant': variant,
        'history': payload.get('history', []),
        'best_val_rmse': payload.get('best_val_rmse'),
        'steps_completed': payload.get('steps_completed'),
        'parameter_count': payload.get('parameter_count'),
        'metadata': payload.get('metadata', {}),
        'checkpoint_payload': payload,
    })
    return result
