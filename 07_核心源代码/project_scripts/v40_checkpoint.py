from __future__ import annotations
from dataclasses import asdict, is_dataclass
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def scaler_to_state(scaler: StandardScaler) -> dict:
    return {
        'mean_': np.asarray(scaler.mean_, dtype=float).tolist(),
        'scale_': np.asarray(scaler.scale_, dtype=float).tolist(),
        'var_': np.asarray(scaler.var_, dtype=float).tolist(),
        'n_features_in_': int(scaler.n_features_in_),
        'n_samples_seen_': int(np.asarray(scaler.n_samples_seen_).max()),
    }


def scaler_from_state(state: dict) -> StandardScaler:
    s = StandardScaler()
    s.mean_ = np.asarray(state['mean_'], dtype=float)
    s.scale_ = np.asarray(state['scale_'], dtype=float)
    s.var_ = np.asarray(state['var_'], dtype=float)
    s.n_features_in_ = int(state['n_features_in_'])
    s.n_samples_seen_ = np.int64(state['n_samples_seen_'])
    return s


def _cpu_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def save_fit_checkpoint(path, model_type, fit_result, model_config, metadata) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    cfg = asdict(model_config) if is_dataclass(model_config) else dict(model_config)
    payload = {
        'format_version': 1,
        'model_type': str(model_type),
        'model_state_dict': _cpu_state_dict(fit_result['model']),
        'model_config': cfg,
        'x_scaler': scaler_to_state(fit_result['x_scaler']),
        'metadata': dict(metadata),
    }
    if model_type == 'tdr':
        payload['y_scaler'] = scaler_to_state(fit_result['y_scaler'])
    torch.save(payload, path)
    return path


def _resolve_device(device='auto'):
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def _load_payload(path):
    try:
        return torch.load(Path(path), map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location='cpu')


def load_fit_checkpoint(path, device='auto') -> dict:
    payload = _load_payload(path)
    dev = _resolve_device(device)
    model_type = payload['model_type']
    cfg = payload['model_config']
    if model_type == 'gru':
        from v31_gru import GRUConfig, RuttingGRU
        gcfg = GRUConfig(**cfg)
        model = RuttingGRU(gcfg).to(dev)
        model.load_state_dict(payload['model_state_dict'])
        return {
            'model': model,
            'x_scaler': scaler_from_state(payload['x_scaler']),
            'metadata': payload.get('metadata', {}),
            'model_config': gcfg,
            'checkpoint_payload': payload,
        }
    if model_type == 'tdr':
        from v31_tdr import TDRStableConfig, tdr
        stable = TDRStableConfig(**cfg['stable'])
        tc = tdr.Config(
            input='', seq_len=int(cfg['seq_len']), epochs=120,
            batch_size=int(cfg.get('batch_size', 64)), lr=stable.lr,
            weight_decay=stable.weight_decay, d_model=stable.d_model,
            dropout=stable.dropout, run_xgb=False, run_dl_baselines=False,
            run_ablation=False, enable_residual_calibration=False,
            delta_loss_weight=.25, dual_state_loss_weight=.05,
            reversible_aux_weight=.03,
        )
        model, _, _, _ = tdr.build_neural_model('TDRPRSN', int(cfg['input_dim']), tc, 'full', dev)
        model.load_state_dict(payload['model_state_dict'])
        return {
            'model': model,
            'x_scaler': scaler_from_state(payload['x_scaler']),
            'y_scaler': scaler_from_state(payload['y_scaler']),
            'config': stable,
            'metadata': payload.get('metadata', {}),
            'model_config': cfg,
            'checkpoint_payload': payload,
        }
    raise ValueError(f'unsupported model_type: {model_type}')
