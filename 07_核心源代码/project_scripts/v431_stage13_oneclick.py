from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import gc
import json
import threading
import time

import pandas as pd

from v43_protocol import (
    V43Protocol,
    checkpoint_dir_v43,
    downstream_run_plan_v43,
    get_fold_v43,
)
from v43_qwen import fold_qwen_paths, generate_fold_q50_updates, train_fold_qwen
from v43_stage13 import (
    build_stage13_reports,
    prepare_fold_v43,
    run_stage13_training,
)


ONECLICK_BOOTSTRAP_N = 10000
ONECLICK_HEARTBEAT_SECONDS = 60
_REQUIRED_RUN_FILES = (
    "checkpoint.pt",
    "run_manifest.json",
    "pred_H006.csv",
    "pred_H012.csv",
    "pred_fixed_origin_2021.csv",
    "horizon_metrics.csv",
    "fixed_origin_metrics.csv",
    "isolation_audit.json",
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _run_files_complete(directory: Path) -> bool:
    return all((directory / name).exists() for name in _REQUIRED_RUN_FILES)


def fold_completion_state(root: str | Path, fold_id: int) -> dict:
    root = Path(root)
    fold = get_fold_v43(fold_id)
    prep_path = root / "reports" / "v43" / "audit" / f"fold{fold_id:02d}_prep_audit.json"
    prep = bool(prep_path.exists() and _load_json(prep_path).get("pass") is True)

    paths = fold_qwen_paths(root, fold_id)
    qwen_manifest = paths["report_dir"] / "training_manifest.json"
    qwen_adapter = paths["model_dir"] / "refit" / "refit_adapter"
    qwen = qwen_manifest.exists() and qwen_adapter.exists()

    synth_complete = []
    for allocation in range(1, 6):
        update = paths["updates_dir"] / f"qwen_Q050_A{allocation}.csv"
        states = paths["updates_dir"] / f"qwen_Q050_A{allocation}_states.csv"
        if update.exists() and states.exists():
            synth_complete.append(allocation)

    formal_runs = 0
    smoke_runs = 0
    incomplete_runs = 0
    expected = downstream_run_plan_v43((fold_id,))
    for key in expected:
        directory = checkpoint_dir_v43(root, key)
        manifest_path = directory / "run_manifest.json"
        if not manifest_path.exists():
            incomplete_runs += 1
            continue
        manifest = _load_json(manifest_path)
        status = str(manifest.get("status", ""))
        if status == "SMOKE_ONLY":
            smoke_runs += 1
            continue
        if status == "FORMAL" and int(manifest.get("max_steps", -1)) == V43Protocol().max_optimizer_steps and _run_files_complete(directory):
            formal_runs += 1
        else:
            incomplete_runs += 1

    train_complete = formal_runs == len(expected) and smoke_runs == 0 and incomplete_runs == 0
    return {
        "fold_id": int(fold_id),
        "group_name": fold.group_name,
        "prep": prep,
        "qwen": bool(qwen),
        "synth_allocations": synth_complete,
        "synth_count": len(synth_complete),
        "formal_runs": formal_runs,
        "smoke_runs": smoke_runs,
        "incomplete_runs": incomplete_runs,
        "expected_runs": len(expected),
        "train_complete": train_complete,
        "complete": bool(prep and qwen and len(synth_complete) == 5 and train_complete),
    }


def report_is_complete(root: str | Path, bootstrap_n: int = ONECLICK_BOOTSTRAP_N) -> bool:
    path = Path(root) / "reports" / "v43" / "stage13_summary.json"
    if not path.exists():
        return False
    summary = _load_json(path)
    return summary.get("status") == "COMPLETE" and int(summary.get("n_boot", -1)) == int(bootstrap_n)


def _cleanup_accelerator() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@contextmanager
def _heartbeat(label: str, seconds: int):
    seconds = int(seconds)
    if seconds <= 0:
        yield
        return
    stop = threading.Event()
    started = time.time()

    def worker():
        while not stop.wait(seconds):
            elapsed = (time.time() - started) / 60.0
            print(f"[Stage13 one-click] {label} still running... elapsed={elapsed:.1f} min", flush=True)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)


def _save_oneclick_status(root: Path, current_fold: int | None, phase: str, status: str) -> None:
    fold_states = [fold_completion_state(root, i) for i in range(1, 8)]
    _write_json(
        root / "reports" / "v43" / "audit" / "stage13_oneclick_status.json",
        {
            "stage": "V4.3.1 Stage 13 one-click scheduler",
            "scientific_protocol": "V4.3 Stage 13 unchanged",
            "status": status,
            "phase": phase,
            "current_fold": current_fold,
            "formal_optimizer_steps_forced": V43Protocol().max_optimizer_steps,
            "bootstrap_n": ONECLICK_BOOTSTRAP_N,
            "folds": fold_states,
            "updated_at_unix": time.time(),
        },
    )


def _raise_if_smoke_exists(root: Path) -> None:
    bad = []
    for fold_id in range(1, 8):
        state = fold_completion_state(root, fold_id)
        if state["smoke_runs"]:
            bad.append((fold_id, state["smoke_runs"]))
    if bad:
        details = ", ".join(f"fold{fold_id:02d}={n}" for fold_id, n in bad)
        raise RuntimeError(
            "SMOKE_ONLY Stage 13 run directories exist and cannot be reused for formal 2400-step training. "
            f"Remove only those smoke run directories before using the one-click formal runner: {details}"
        )


def run_oneclick_stage13(
    root: str | Path,
    *,
    prepare_fn=prepare_fold_v43,
    qwen_fn=train_fold_qwen,
    synth_fn=generate_fold_q50_updates,
    train_fn=run_stage13_training,
    report_fn=build_stage13_reports,
    bootstrap_n: int = ONECLICK_BOOTSTRAP_N,
    heartbeat_seconds: int = ONECLICK_HEARTBEAT_SECONDS,
) -> dict:
    root = Path(root)
    master_path = root / "data" / "processed" / "rutting_master_v31.csv"
    if not master_path.exists():
        raise FileNotFoundError(master_path)
    master = pd.read_csv(master_path)

    # The one-click entry is formal-only. Stale V43_MODE/V43_FOLD/V43_MAX_STEPS
    # environment variables are intentionally ignored.
    _raise_if_smoke_exists(root)
    _save_oneclick_status(root, None, "startup", "RUNNING")

    print("=" * 78, flush=True)
    print("V4.3.1 Stage 13 LOSGO ONE-CLICK FORMAL RUNNER", flush=True)
    print("Scientific protocol unchanged: 7 LOSGO folds; per-fold Qwen refit; Q50; TDR; H=6/H=12/fixed-origin", flush=True)
    print(f"Formal optimizer budget is forced to {V43Protocol().max_optimizer_steps} steps; final bootstrap={int(bootstrap_n)}", flush=True)
    print("Existing completed artifacts are resumed/skipped automatically.", flush=True)
    print("=" * 78, flush=True)

    for fold_id in range(1, 8):
        fold = get_fold_v43(fold_id)
        state = fold_completion_state(root, fold_id)
        print(
            f"\n[Stage13] Overall fold {fold_id}/7 | {fold.group_name} | "
            f"prep={state['prep']} qwen={state['qwen']} synth={state['synth_count']}/5 "
            f"formal={state['formal_runs']}/18",
            flush=True,
        )
        if state["complete"]:
            print(f"[Fold {fold_id}] COMPLETE -> skipped", flush=True)
            continue

        if not state["prep"]:
            _save_oneclick_status(root, fold_id, "prep", "RUNNING")
            print(f"[Fold {fold_id}] prep START", flush=True)
            prepare_fn(root, master, fold_id)
            state = fold_completion_state(root, fold_id)
            if not state["prep"]:
                raise RuntimeError(f"Fold {fold_id} prep did not produce a passing isolation audit")
            print(f"[Fold {fold_id}] prep PASS", flush=True)

        if not state["qwen"]:
            _save_oneclick_status(root, fold_id, "qwen", "RUNNING")
            print(f"[Fold {fold_id}] qwen selection+refit START", flush=True)
            with _heartbeat(f"Fold {fold_id} Qwen selection/refit", heartbeat_seconds):
                qwen_fn(root, fold)
            _cleanup_accelerator()
            state = fold_completion_state(root, fold_id)
            if not state["qwen"]:
                raise RuntimeError(f"Fold {fold_id} Qwen refit did not complete")
            print(f"[Fold {fold_id}] qwen DONE", flush=True)

        if state["synth_count"] < 5:
            _save_oneclick_status(root, fold_id, "synth", "RUNNING")
            print(f"[Fold {fold_id}] synth START ({state['synth_count']}/5 already complete)", flush=True)
            with _heartbeat(f"Fold {fold_id} Q50 synthetic generation", heartbeat_seconds):
                synth_fn(root, master, fold)
            _cleanup_accelerator()
            state = fold_completion_state(root, fold_id)
            if state["synth_count"] != 5:
                raise RuntimeError(f"Fold {fold_id} generated only {state['synth_count']}/5 Q50 allocations")
            print(f"[Fold {fold_id}] synth 5/5 DONE", flush=True)

        if not state["train_complete"]:
            _save_oneclick_status(root, fold_id, "train", "RUNNING")
            print(f"[Fold {fold_id}] TDR formal training START ({state['formal_runs']}/18 already complete)", flush=True)
            train_fn(
                root,
                (fold_id,),
                max_steps=V43Protocol().max_optimizer_steps,
                device="auto",
                max_runs=None,
            )
            _cleanup_accelerator()
            state = fold_completion_state(root, fold_id)
            if not state["train_complete"]:
                raise RuntimeError(
                    f"Fold {fold_id} training incomplete after runner returned: "
                    f"formal={state['formal_runs']}/18 incomplete={state['incomplete_runs']} smoke={state['smoke_runs']}"
                )
            print(f"[Fold {fold_id}] train 18/18 DONE", flush=True)

        state = fold_completion_state(root, fold_id)
        if not state["complete"]:
            raise RuntimeError(f"Fold {fold_id} did not pass final completion gate: {state}")
        _save_oneclick_status(root, fold_id, "fold_complete", "RUNNING")
        print(f"[Fold {fold_id}] audit PASS | FOLD COMPLETE", flush=True)

    if report_is_complete(root, bootstrap_n):
        print(f"\n[Stage13 report] COMPLETE n_boot={int(bootstrap_n)} -> skipped", flush=True)
    else:
        _save_oneclick_status(root, None, "report", "RUNNING")
        print(f"\n[Stage13 report] START final aggregation + bootstrap n={int(bootstrap_n)}", flush=True)
        print("A heartbeat will print while a bootstrap block is busy, so the console will not look frozen.", flush=True)
        with _heartbeat("final report/bootstrap", heartbeat_seconds):
            report_fn(root, n_boot=int(bootstrap_n))

    summary_path = root / "reports" / "v43" / "stage13_summary.json"
    if not summary_path.exists():
        raise RuntimeError("Stage 13 report returned without stage13_summary.json")
    summary = _load_json(summary_path)
    if summary.get("status") != "COMPLETE":
        _save_oneclick_status(root, None, "final_gate", "FAILED")
        raise RuntimeError(f"Stage 13 final summary is not COMPLETE: {summary}")
    if int(summary.get("formal_runs", -1)) != 126 or int(summary.get("smoke_runs", -1)) != 0:
        _save_oneclick_status(root, None, "final_gate", "FAILED")
        raise RuntimeError(
            "Stage 13 final formal-run gate failed: "
            f"formal_runs={summary.get('formal_runs')} smoke_runs={summary.get('smoke_runs')}"
        )
    if summary.get("formal_isolation_audits_all_pass") is not True:
        _save_oneclick_status(root, None, "final_gate", "FAILED")
        raise RuntimeError("Stage 13 final isolation audit gate failed")

    _save_oneclick_status(root, None, "complete", "COMPLETE")
    print("\n" + "=" * 78, flush=True)
    print("STAGE 13 COMPLETE: 7/7 folds, 126/126 formal runs, isolation PASS", flush=True)
    print(f"Reports: {root / 'reports' / 'v43'}", flush=True)
    print("=" * 78, flush=True)
    return summary


def main(root: str | Path | None = None) -> None:
    root = Path(root or Path(__file__).resolve().parents[1])
    run_oneclick_stage13(root)


if __name__ == "__main__":
    main()
