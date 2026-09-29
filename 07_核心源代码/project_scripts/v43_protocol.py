from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json

from v42_protocol import protocol_signature_v42


STRUCTURE_GROUPS: dict[str, tuple[str, ...]] = {
    "I_SemiRigid_12cmAC": ("STR1", "STR2", "STR3"),
    "II_RigidComposite_12cmAC": ("STR4", "STR5"),
    "III_SemiRigid_16_18cmAC": ("STR6", "STR7", "STR8", "STR9"),
    "IV_Inverted_24_28cmAC": ("STR10", "STR12"),
    "V_ThickAC_I_24_28cm": ("STR11", "STR13", "STR14"),
    "VI_ThickAC_II_36cm": ("STR15", "STR16", "STR17"),
    "VII_FullDepth_48_52cm": ("STR18", "STR19"),
}


@dataclass(frozen=True)
class V43Protocol:
    protocol_name: str = "Qwen-TDR-V4.3-Stage13-StructuralOOD"
    post_hoc: bool = True
    primary_horizon: int = 6
    horizons: tuple[int, ...] = (6, 12)
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
    synthetic_fraction: float = 0.50
    qwen_train_year_max: int = 2017
    qwen_dev_year: int = 2018
    qwen_refit_year_max: int = 2018
    downstream_train_year_max: int = 2019
    downstream_dev_year: int = 2020
    chronological_evaluation_year: int = 2021
    generator_history_steps: int = 4
    generator_future_steps: int = 6


@dataclass(frozen=True)
class V43Fold:
    fold_id: int
    group_name: str
    held_out_structures: tuple[str, ...]
    train_structures: tuple[str, ...]

    @property
    def stem(self) -> str:
        return f"fold{int(self.fold_id):02d}_{self.group_name}"


@dataclass(frozen=True)
class V43RunKey:
    fold_id: int
    condition: str
    allocation: int
    seed: int

    def __post_init__(self) -> None:
        if int(self.fold_id) not in range(1, 8):
            raise ValueError("fold_id must be 1..7")
        if self.condition not in {"full_real", "qwen_additive"}:
            raise ValueError("condition must be full_real or qwen_additive")
        if int(self.seed) not in set(V43Protocol().neural_seeds):
            raise ValueError("seed must be one of 42,43,44")
        if self.condition == "full_real" and int(self.allocation) != 0:
            raise ValueError("full_real is allocation-independent and must use allocation=0")
        if self.condition == "qwen_additive" and int(self.allocation) not in {1, 2, 3, 4, 5}:
            raise ValueError("qwen_additive allocation must be 1..5")

    @property
    def q(self) -> float:
        return 0.5 if self.condition == "qwen_additive" else 0.0

    @property
    def stem(self) -> str:
        qtag = 50 if self.condition == "qwen_additive" else 0
        return (
            f"fold{int(self.fold_id):02d}_tdr_{self.condition}_"
            f"Q{qtag:03d}_A{int(self.allocation)}_S{int(self.seed)}"
        )


def validate_protocol_v43(protocol: V43Protocol | None = None) -> None:
    p = protocol or V43Protocol()
    members = [s for group in STRUCTURE_GROUPS.values() for s in group]
    if len(STRUCTURE_GROUPS) != 7 or len(members) != 19 or len(set(members)) != 19:
        raise ValueError("Stage 13 structure groups must partition STR1-STR19 exactly once")
    if set(members) != {f"STR{i}" for i in range(1, 20)}:
        raise ValueError("Stage 13 structure groups do not match STR1-STR19")
    if p.post_hoc is not True:
        raise ValueError("Stage 13 must remain explicitly post-hoc")
    if p.primary_horizon != 6 or p.horizons != (6, 12):
        raise ValueError("Stage 13 H=6/H=12 evaluation changed")
    if p.max_optimizer_steps != 2400:
        raise ValueError("Stage 13 changed frozen optimizer budget")
    if float(p.synthetic_fraction) != 0.50:
        raise ValueError("Stage 13 additive comparison is frozen at Q50")
    if p.downstream_train_year_max != 2019 or p.downstream_dev_year != 2020 or p.chronological_evaluation_year != 2021:
        raise ValueError("Stage 13 chronology changed")


def folds_v43() -> list[V43Fold]:
    validate_protocol_v43()
    all_structures = tuple(f"STR{i}" for i in range(1, 20))
    out: list[V43Fold] = []
    for fold_id, (group_name, held_out) in enumerate(STRUCTURE_GROUPS.items(), 1):
        held = tuple(held_out)
        train = tuple(s for s in all_structures if s not in set(held))
        out.append(V43Fold(fold_id, group_name, held, train))
    return out


def get_fold_v43(fold_id: int) -> V43Fold:
    fold_id = int(fold_id)
    for fold in folds_v43():
        if fold.fold_id == fold_id:
            return fold
    raise ValueError(f"unknown Stage 13 fold: {fold_id}")


def downstream_run_plan_v43(fold_ids: tuple[int, ...] | list[int] | None = None) -> list[V43RunKey]:
    p = V43Protocol()
    ids = tuple(int(x) for x in (fold_ids or range(1, 8)))
    out: list[V43RunKey] = []
    for fold_id in ids:
        get_fold_v43(fold_id)
        out.extend(V43RunKey(fold_id, "full_real", 0, seed) for seed in p.neural_seeds)
        out.extend(
            V43RunKey(fold_id, "qwen_additive", allocation, seed)
            for allocation in range(1, 6)
            for seed in p.neural_seeds
        )
    return out


def formal_run_allowed_v43(max_steps: int) -> bool:
    return int(max_steps) == int(V43Protocol().max_optimizer_steps)


def fold_root_v43(root: str | Path, fold_id: int) -> Path:
    fold = get_fold_v43(fold_id)
    return Path(root) / "data" / "v43_losgo" / fold.stem


def checkpoint_dir_v43(root: str | Path, run_key: V43RunKey) -> Path:
    return (
        Path(root)
        / "checkpoints"
        / "v43"
        / f"fold{int(run_key.fold_id):02d}"
        / run_key.condition
        / f"A{int(run_key.allocation)}"
        / f"S{int(run_key.seed)}"
    )


def protocol_signature_v43(protocol: V43Protocol | None = None) -> str:
    p = protocol or V43Protocol()
    validate_protocol_v43(p)
    payload = {
        "v43": asdict(p),
        "structure_groups": STRUCTURE_GROUPS,
        "parent_v42_signature": protocol_signature_v42(),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
