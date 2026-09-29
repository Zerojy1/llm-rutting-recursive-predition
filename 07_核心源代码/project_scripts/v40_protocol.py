from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
from v31_protocol import V31Protocol, protocol_signature as v31_protocol_signature


@dataclass(frozen=True)
class V40Protocol:
    protocol_name: str = 'Qwen-TDR-V4.0-Stage10'
    primary_horizon: int = 6
    horizons: tuple[int, ...] = (1, 3, 6, 12)
    seq_len: int = 6
    noninferiority_margin: float = 0.10
    sensitivity_margin: float = 0.05
    max_optimizer_steps: int = 2400
    warmup_steps: int = 100
    validation_every_steps: int = 50
    patience_checks: int = 6
    allocation_seeds: tuple[int, ...] = (202608141, 202608142, 202608143, 202608144, 202608145)
    neural_seeds: tuple[int, ...] = (42, 43, 44)
    reproduction_rmse_tolerance_mm: float = 0.003
    reproduction_prediction_corr_min: float = 0.995
    chronological_evaluation_year: int = 2021
    scenario_conditioned_exogenous: bool = True
    validation_reference: str = 'Zhu et al. 2025 Double-T evidence-chain; chronological isolation retained'


@dataclass(frozen=True)
class RunKey:
    model: str
    method: str
    q: float
    allocation: int
    seed: int

    def __post_init__(self):
        if self.model not in {'gru', 'tdr'}:
            raise ValueError(f'unsupported model: {self.model}')
        if self.method not in {'full_real', 'qwen', 'historical_only', 'real_bootstrap', 'timegan', 'timeweaver'}:
            raise ValueError(f'unsupported method: {self.method}')
        if not 0.0 <= float(self.q) <= 1.0:
            raise ValueError('q must be in [0,1]')
        if int(self.allocation) < 0:
            raise ValueError('allocation must be non-negative')

    @property
    def stem(self) -> str:
        return f'{self.model}_{self.method}_Q{int(round(float(self.q)*100)):03d}_A{int(self.allocation)}_S{int(self.seed)}'


def validate_protocol(p: V40Protocol | None = None) -> None:
    p = p or V40Protocol()
    if p.primary_horizon != 6 or tuple(p.horizons) != (1, 3, 6, 12):
        raise ValueError('V4.0 horizons changed')
    if p.max_optimizer_steps != 2400 or p.warmup_steps != 100 or p.validation_every_steps != 50 or p.patience_checks != 6:
        raise ValueError('frozen optimizer budget changed')
    if tuple(p.neural_seeds) != (42, 43, 44):
        raise ValueError('frozen neural seeds changed')
    if tuple(p.allocation_seeds) != (202608141, 202608142, 202608143, 202608144, 202608145):
        raise ValueError('frozen allocation seeds changed')
    if p.chronological_evaluation_year != 2021:
        raise ValueError('chronological evaluation year changed')


def protocol_signature(p: V40Protocol | None = None) -> str:
    p = p or V40Protocol()
    validate_protocol(p)
    payload = {'v40': asdict(p), 'parent_v31_signature': v31_protocol_signature(V31Protocol())}
    blob = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()


def _expanded(method: str, q_values: tuple[float, ...], allocations: tuple[int, ...], p: V40Protocol) -> list[RunKey]:
    out = []
    for model in ('gru', 'tdr'):
        for q in q_values:
            for allocation in allocations:
                for seed in p.neural_seeds:
                    out.append(RunKey(model, method, q, allocation, seed))
    return out


def phase_a_run_plan(p: V40Protocol | None = None) -> list[RunKey]:
    p = p or V40Protocol(); validate_protocol(p)
    plan = _expanded('full_real', (0.0,), (0,), p)
    plan += _expanded('qwen', (0.25, 0.50, 0.75, 1.00), (1, 2, 3, 4, 5), p)
    return plan


def phase_b_run_plan(p: V40Protocol | None = None) -> list[RunKey]:
    p = p or V40Protocol(); validate_protocol(p)
    plan = _expanded('historical_only', (0.0,), (0,), p)
    plan += _expanded('real_bootstrap', (0.25, 0.50, 0.75), (1, 2, 3, 4, 5), p)
    plan += _expanded('timegan', (0.25, 0.50, 0.75, 1.00), (1, 2, 3, 4, 5), p)
    plan += _expanded('timeweaver', (0.25, 0.50, 0.75, 1.00), (1, 2, 3, 4, 5), p)
    return plan


def run_stem(run_key: RunKey) -> str:
    return run_key.stem


def checkpoint_dir(root: str | Path, run_key: RunKey) -> Path:
    root = Path(root)
    method = run_key.method if run_key.q == 0 else f'{run_key.method}_Q{int(round(run_key.q*100)):03d}'
    return root / 'checkpoints' / 'v40' / run_key.model / method / f'A{run_key.allocation}' / f'S{run_key.seed}'
