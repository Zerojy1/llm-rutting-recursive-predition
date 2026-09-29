from __future__ import annotations

from pathlib import Path
import gc
import json
import os
import numpy as np
import pandas as pd
import torch

from v31_experiment_engine import make_masked_update
from v31_qwen_train import train_phase
from v31_quality import empirical_increment_bounds, prospective_rutting_bounds
from v43_data import split_master_for_fold, strip_identity_from_condition, audit_fold_isolation
from v43_protocol import V43Fold, V43Protocol, get_fold_v43, protocol_signature_v43


def fold_qwen_paths(root: str | Path, fold_id: int) -> dict[str, Path]:
    root = Path(root)
    fold = get_fold_v43(fold_id)
    token = fold.stem
    return {
        "data_dir": root / "data" / "v43_losgo" / token / "qwen",
        "updates_dir": root / "data" / "v43_losgo" / token / "updates",
        "model_dir": root / "models" / "v43_losgo" / token / "qwen",
        "report_dir": root / "reports" / "v43" / "qwen" / token,
    }


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def train_fold_qwen(
    root: str | Path,
    fold: V43Fold,
    base_model_dir: str | Path | None = None,
    train_phase_fn=train_phase,
) -> dict:
    p = V43Protocol()
    paths = fold_qwen_paths(root, fold.fold_id)
    data = paths["data_dir"]
    required = {
        "train": data / "qwen_train_2016_2017.jsonl",
        "dev": data / "qwen_dev_2018.jsonl",
        "refit": data / "qwen_refit_2016_2018.jsonl",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Stage 13 fold Qwen datasets missing: " + ", ".join(missing))
    model_dir = Path(base_model_dir or os.environ.get("QWEN_BASE_MODEL_DIR", r"K:\ollama\Qwen3.5-9B-hf"))
    if train_phase_fn is train_phase and not model_dir.exists():
        raise FileNotFoundError(f"Qwen base model not found: {model_dir}")

    paths["model_dir"].mkdir(parents=True, exist_ok=True)
    existing_manifest = paths["report_dir"] / "training_manifest.json"
    existing_adapter = paths["model_dir"] / "refit" / "refit_adapter"
    if existing_manifest.exists() and existing_adapter.exists():
        return json.loads(existing_manifest.read_text(encoding="utf-8"))
    selection = train_phase_fn(
        model_dir,
        required["train"],
        required["dev"],
        paths["model_dir"] / "selection",
    )
    best_epoch = int(selection["best_epoch"])
    if best_epoch <= 0:
        raise RuntimeError("Stage 13 Qwen selection returned invalid best_epoch")

    # Explicitly release selection model resources before refit; train_phase loads the base model anew.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    refit = train_phase_fn(
        model_dir,
        required["refit"],
        None,
        paths["model_dir"] / "refit",
        fixed_epochs=best_epoch,
    )
    manifest = {
        "stage": "V4.3 Stage 13",
        "post_hoc": True,
        "fold_id": fold.fold_id,
        "group_name": fold.group_name,
        "held_out_structures": list(fold.held_out_structures),
        "train_structures": list(fold.train_structures),
        "selection": selection,
        "refit": refit,
        "base_model_dir": str(model_dir),
        "protocol_signature": protocol_signature_v43(),
        "heldout_seen_by_qwen": False,
        "qwen_target_years": {"selection_train_max": 2017, "selection_dev": 2018, "refit_max": 2018},
    }
    _json_dump(paths["report_dir"] / "training_manifest.json", manifest)
    return manifest


def _fold_qc_generator(root: Path, fold: V43Fold, base_model_dir: str | Path | None = None):
    from v31_qwen_generate import QwenTrajectoryGenerator
    from v31_qwen_qc import QualityControlledQwen

    paths = fold_qwen_paths(root, fold.fold_id)
    adapter = paths["model_dir"] / "refit" / "refit_adapter"
    if not adapter.exists():
        raise FileNotFoundError(f"Stage 13 fold refit adapter not found: {adapter}")
    model_dir = Path(base_model_dir or os.environ.get("QWEN_BASE_MODEL_DIR", r"K:\ollama\Qwen3.5-9B-hf"))
    if not model_dir.exists():
        raise FileNotFoundError(f"Qwen base model not found: {model_dir}")

    windows = pd.read_csv(paths["data_dir"] / "refit_windows.csv")
    p = V43Protocol()
    increments = []
    for i in range(1, p.generator_future_steps + 1):
        increments.extend(pd.to_numeric(windows[f"target_increment_f{i}"], errors="coerce").dropna().tolist())
    train_master = pd.read_csv(Path(root) / "data/processed/rutting_master_v31.csv")
    train_master, _ = split_master_for_fold(train_master, fold)
    source = train_master[
        pd.to_numeric(train_master["Year"], errors="coerce").le(p.qwen_refit_year_max)
        & pd.to_numeric(train_master["Rutting_Observed"], errors="coerce").fillna(0).astype(int).eq(1)
    ]
    ib = empirical_increment_bounds(increments, .005, .995, .05)
    rb = prospective_rutting_bounds(source.Rutting_mm.to_numpy(float), ib, horizon=p.generator_future_steps, pad=.25)
    train_real = np.asarray(
        [[r[f"target_increment_f{i}"] for i in range(1, p.generator_future_steps + 1)] for _, r in windows.iterrows()],
        float,
    )
    q = QwenTrajectoryGenerator(model_dir, adapter)
    return QualityControlledQwen(
        q, ib, rb, k=5,
        local_center=np.mean(train_real, axis=0),
        local_scale=np.std(train_real, axis=0) + 1e-6,
    )


def generate_fold_q50_updates(
    root: str | Path,
    master: pd.DataFrame,
    fold: V43Fold,
    generator_fn=None,
    base_model_dir: str | Path | None = None,
    allocation_seeds: tuple[int, ...] | list[int] | None = None,
    output_prefix: str = "qwen",
) -> dict[int, Path]:
    root = Path(root)
    p = V43Protocol()
    paths = fold_qwen_paths(root, fold.fold_id)
    paths["updates_dir"].mkdir(parents=True, exist_ok=True)
    train_master, _ = split_master_for_fold(master, fold)
    base_generator = generator_fn or _fold_qc_generator(root, fold, base_model_dir=base_model_dir)

    # Neither a raw Qwen generator nor QC receives road identity. Grouping identity is used only outside the model call.
    def anonymous_generator(condition: dict):
        return base_generator(strip_identity_from_condition(condition))

    seeds = tuple(int(x) for x in (allocation_seeds or p.allocation_seeds))
    outputs: dict[int, Path] = {}
    for allocation, allocation_seed in enumerate(seeds, 1):
        upath = paths["updates_dir"] / f"{output_prefix}_Q050_A{allocation}.csv"
        spath = paths["updates_dir"] / f"{output_prefix}_Q050_A{allocation}_states.csv"
        if upath.exists() and spath.exists():
            update = pd.read_csv(upath)
            audit = audit_fold_isolation(fold, train_master=train_master, synthetic_update=update)
            if not audit["pass"]:
                raise RuntimeError(f"existing Stage 13 synthetic update failed isolation audit: {audit}")
            outputs[allocation] = upath
            print(f"[Stage13 synth] Fold {fold.fold_id} allocation A{allocation}/A{len(seeds)} already complete -> skip", flush=True)
            continue
        print(f"[Stage13 synth] Fold {fold.fold_id} allocation A{allocation}/A{len(seeds)} START", flush=True)
        update, _, states = make_masked_update(
            train_master,
            p.synthetic_fraction,
            int(allocation_seed),
            anonymous_generator,
            year=p.downstream_train_year_max,
        )
        audit = audit_fold_isolation(fold, train_master=train_master, synthetic_update=update)
        if not audit["pass"]:
            raise RuntimeError(f"Stage 13 synthetic update failed isolation audit: {audit}")
        update.to_csv(upath, index=False, encoding="utf-8-sig")
        states.to_csv(spath, index=False, encoding="utf-8-sig")
        outputs[allocation] = upath
        print(f"[Stage13 synth] Fold {fold.fold_id} allocation A{allocation}/A{len(seeds)} DONE rows={len(update)}", flush=True)

    _json_dump(
        paths["report_dir"] / "generation_manifest.json",
        {
            "stage": "V4.3 Stage 13",
            "fold_id": fold.fold_id,
            "group_name": fold.group_name,
            "held_out_structures": list(fold.held_out_structures),
            "train_structures": list(fold.train_structures),
            "synthetic_fraction": p.synthetic_fraction,
            "allocation_seeds": list(seeds),
            "outputs": {str(k): str(v) for k, v in outputs.items()},
            "identity_keys_sent_to_generator": [],
            "protocol_signature": protocol_signature_v43(),
        },
    )
    return outputs
