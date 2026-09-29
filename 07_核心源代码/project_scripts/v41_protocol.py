from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json

from v40_protocol import V40Protocol, protocol_signature as v40_protocol_signature


@dataclass(frozen=True)
class V41Protocol:
    protocol_name: str = "Qwen-TDR-V4.1-Stage11"
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
    thresholds_mm: tuple[float, ...] = (8.0, 10.0)


@dataclass(frozen=True)
class V41RunKey:
    model: str
    condition: str
    real_fraction: float
    synthetic_fraction: float
    allocation: int
    seed: int

    def __post_init__(self) -> None:
        if self.model not in {"gru", "tdr"}:
            raise ValueError(f"unsupported model: {self.model}")
        allowed = {
            "real_reduced": (0.5, 0.0),
            "qwen_additive": (1.0, 0.5),
        }
        if self.condition not in allowed:
            raise ValueError(f"unsupported condition: {self.condition}")
        expected = allowed[self.condition]
        actual = (float(self.real_fraction), float(self.synthetic_fraction))
        if actual != expected:
            raise ValueError(
                f"{self.condition} requires real/synthetic fractions {expected}, got {actual}"
            )
        if int(self.allocation) not in {1, 2, 3, 4, 5}:
            raise ValueError("allocation must be one of 1,2,3,4,5")
        if int(self.seed) not in {42, 43, 44}:
            raise ValueError("seed must be one of 42,43,44")

    @property
    def stem(self) -> str:
        rtag = int(round(float(self.real_fraction) * 100))
        qtag = int(round(float(self.synthetic_fraction) * 100))
        return (
            f"{self.model}_{self.condition}_R{rtag:03d}_Q{qtag:03d}_"
            f"A{int(self.allocation)}_S{int(self.seed)}"
        )


def validate_protocol_v41(protocol: V41Protocol | None = None) -> None:
    p = protocol or V41Protocol()
    parent = V40Protocol()
    frozen_fields = (
        "primary_horizon",
        "horizons",
        "seq_len",
        "max_optimizer_steps",
        "warmup_steps",
        "validation_every_steps",
        "patience_checks",
        "allocation_seeds",
        "neural_seeds",
        "chronological_evaluation_year",
    )
    for name in frozen_fields:
        if getattr(p, name) != getattr(parent, name):
            raise ValueError(f"Stage 11 changed frozen Stage 10 field: {name}")
    if p.development_year != 2020:
        raise ValueError("development year must remain 2020")
    if tuple(p.thresholds_mm) != (8.0, 10.0):
        raise ValueError("maintenance thresholds must remain 8 and 10 mm")


def stage11_run_plan(protocol: V41Protocol | None = None) -> list[V41RunKey]:
    p = protocol or V41Protocol()
    validate_protocol_v41(p)
    plan: list[V41RunKey] = []
    conditions = (
        ("real_reduced", 0.5, 0.0),
        ("qwen_additive", 1.0, 0.5),
    )
    for model in ("gru", "tdr"):
        for condition, real_fraction, synthetic_fraction in conditions:
            for allocation in range(1, 6):
                for seed in p.neural_seeds:
                    plan.append(
                        V41RunKey(
                            model,
                            condition,
                            real_fraction,
                            synthetic_fraction,
                            allocation,
                            seed,
                        )
                    )
    return plan


def checkpoint_dir_v41(root: str | Path, run_key: V41RunKey) -> Path:
    root = Path(root)
    rtag = int(round(float(run_key.real_fraction) * 100))
    qtag = int(round(float(run_key.synthetic_fraction) * 100))
    condition = f"{run_key.condition}_R{rtag:03d}_Q{qtag:03d}"
    return (
        root
        / "checkpoints"
        / "v41"
        / run_key.model
        / condition
        / f"A{run_key.allocation}"
        / f"S{run_key.seed}"
    )


def protocol_signature_v41(protocol: V41Protocol | None = None) -> str:
    p = protocol or V41Protocol()
    validate_protocol_v41(p)
    payload = {
        "v41": asdict(p),
        "parent_v40_signature": v40_protocol_signature(),
    }
    blob = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
