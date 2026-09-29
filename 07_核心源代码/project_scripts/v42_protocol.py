from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json

from v41_protocol import V41Protocol, protocol_signature_v41


@dataclass(frozen=True)
class V42Protocol:
    protocol_name: str = "Qwen-TDR-V4.2-Stage12-PostHocValidation"
    post_hoc: bool = True
    primary_horizon: int = 6
    horizons: tuple[int, ...] = (1, 3, 6, 12)
    seq_len: int = 6
    max_optimizer_steps: int = 2400
    warmup_steps: int = 100
    validation_every_steps: int = 50
    patience_checks: int = 6
    allocation_seeds: tuple[int, ...] = (
        202608141,
        202608142,
        202608143,
        202608144,
        202608145,
    )
    neural_seeds: tuple[int, ...] = (42, 43, 44)
    development_year: int = 2020
    chronological_evaluation_year: int = 2021
    synthetic_fraction: float = 0.50
    ablation_variants: tuple[str, ...] = (
        "A3_no_tcn",
        "A4_no_thermal_gate",
        "A5_no_reversible",
        "A7_no_obs_adapter",
    )
    curve_probe_allocation: int = 3
    curve_probe_seed: int = 42


@dataclass(frozen=True)
class V42RunKey:
    experiment: str
    model: str
    variant: str
    condition: str
    allocation: int
    seed: int

    def __post_init__(self) -> None:
        if self.experiment not in {"gru_large_sensitivity", "tdr_ablation", "curve_probe"}:
            raise ValueError(f"unsupported Stage 12 experiment: {self.experiment}")
        if self.model not in {"gru", "tdr"}:
            raise ValueError(f"unsupported Stage 12 model: {self.model}")
        if self.condition not in {"full_real", "qwen_additive"}:
            raise ValueError(f"unsupported Stage 12 condition: {self.condition}")
        if int(self.seed) not in {42, 43, 44}:
            raise ValueError("seed must be one of 42,43,44")
        if self.condition == "full_real" and int(self.allocation) != 0:
            raise ValueError("full_real Stage 12 runs are allocation-independent and use allocation=0")
        if self.condition == "qwen_additive" and int(self.allocation) not in {1, 2, 3, 4, 5}:
            raise ValueError("qwen_additive allocation must be 1..5")
        if self.experiment == "gru_large_sensitivity":
            if self.model != "gru" or self.variant != "Large":
                raise ValueError("gru_large_sensitivity requires model=gru, variant=Large")
        if self.experiment == "tdr_ablation":
            if self.model != "tdr" or self.variant not in set(V42Protocol().ablation_variants):
                raise ValueError("tdr_ablation requires one frozen Stage 12 ablation variant")
        if self.experiment == "curve_probe":
            expected = "frozen_gru" if self.model == "gru" else "full"
            if self.variant != expected:
                raise ValueError(f"curve_probe {self.model} requires variant={expected}")

    @property
    def stem(self) -> str:
        qtag = 50 if self.condition == "qwen_additive" else 0
        return (
            f"{self.experiment}_{self.model}_{self.variant}_{self.condition}_"
            f"Q{qtag:03d}_A{int(self.allocation)}_S{int(self.seed)}"
        )


def validate_protocol_v42(protocol: V42Protocol | None = None) -> None:
    p = protocol or V42Protocol()
    parent = V41Protocol()
    frozen = (
        "primary_horizon",
        "horizons",
        "seq_len",
        "max_optimizer_steps",
        "warmup_steps",
        "validation_every_steps",
        "patience_checks",
        "allocation_seeds",
        "neural_seeds",
        "development_year",
        "chronological_evaluation_year",
    )
    for name in frozen:
        if getattr(p, name) != getattr(parent, name):
            raise ValueError(f"Stage 12 changed frozen Stage 11 field: {name}")
    if p.post_hoc is not True:
        raise ValueError("Stage 12 must remain explicitly post-hoc")
    if float(p.synthetic_fraction) != 0.5:
        raise ValueError("Stage 12 additive sensitivity is frozen at Q50")


def _condition_runs(experiment: str, model: str, variant: str) -> list[V42RunKey]:
    p = V42Protocol()
    rows = [V42RunKey(experiment, model, variant, "full_real", 0, seed) for seed in p.neural_seeds]
    rows += [
        V42RunKey(experiment, model, variant, "qwen_additive", allocation, seed)
        for allocation in range(1, 6)
        for seed in p.neural_seeds
    ]
    return rows


def gru_large_sensitivity_plan() -> list[V42RunKey]:
    validate_protocol_v42()
    return _condition_runs("gru_large_sensitivity", "gru", "Large")


def tdr_ablation_plan() -> list[V42RunKey]:
    validate_protocol_v42()
    out: list[V42RunKey] = []
    for variant in V42Protocol().ablation_variants:
        out.extend(_condition_runs("tdr_ablation", "tdr", variant))
    return out


def curve_probe_plan() -> list[V42RunKey]:
    p = V42Protocol()
    validate_protocol_v42(p)
    out: list[V42RunKey] = []
    for model, variant in (("gru", "frozen_gru"), ("tdr", "full")):
        out.append(V42RunKey("curve_probe", model, variant, "full_real", 0, p.curve_probe_seed))
        out.append(
            V42RunKey(
                "curve_probe",
                model,
                variant,
                "qwen_additive",
                p.curve_probe_allocation,
                p.curve_probe_seed,
            )
        )
    return out


def formal_run_allowed_v42(max_steps: int) -> bool:
    return int(max_steps) == int(V42Protocol().max_optimizer_steps)


def checkpoint_dir_v42(root: str | Path, run_key: V42RunKey) -> Path:
    root = Path(root)
    return (
        root
        / "checkpoints"
        / "v42"
        / run_key.experiment
        / run_key.model
        / run_key.variant
        / run_key.condition
        / f"A{run_key.allocation}"
        / f"S{run_key.seed}"
    )


def protocol_signature_v42(protocol: V42Protocol | None = None) -> str:
    p = protocol or V42Protocol()
    validate_protocol_v42(p)
    payload = {"v42": asdict(p), "parent_v41_signature": protocol_signature_v41()}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
