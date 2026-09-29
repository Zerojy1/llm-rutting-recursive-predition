from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from v31_statistics import hierarchical_cluster_relative_bootstrap
from v40_metrics import paired_noninferiority_by_horizon
from v43_checkpoint import save_v43_checkpoint, load_v43_checkpoint
from v43_data import (
    audit_fold_isolation,
    audit_identity_free_features,
    build_fold_qwen_datasets,
    split_master_for_fold,
)
from v43_eval import forecast_ood_fixed_origin, forecast_ood_horizon, summarize_ood_predictions
from v43_fit import build_v43_dev_arrays, build_v43_training_arrays, fit_v43_run
from v43_protocol import (
    STRUCTURE_GROUPS,
    V43Protocol,
    V43RunKey,
    checkpoint_dir_v43,
    downstream_run_plan_v43,
    formal_run_allowed_v43,
    get_fold_v43,
    protocol_signature_v43,
)
from v43_qwen import fold_qwen_paths, generate_fold_q50_updates, train_fold_qwen


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def parse_fold_ids(raw: str | None) -> tuple[int, ...]:
    text = str(raw or "").strip()
    if not text or text.lower() in {"all", "*"}:
        return tuple(range(1, 8))
    ids = tuple(dict.fromkeys(int(x.strip()) for x in text.split(",") if x.strip()))
    if not ids or any(x not in range(1, 8) for x in ids):
        raise ValueError("V43_FOLD must be 1..7, comma-separated, or empty/all")
    return ids


def stage13_mode_plan(mode: str, fold_ids: tuple[int, ...]) -> list[V43RunKey]:
    mode = str(mode).strip().lower()
    if mode == "train":
        return downstream_run_plan_v43(fold_ids)
    if mode in {"prep", "qwen", "synth", "report", "all"}:
        return []
    raise ValueError(f"unsupported V43_MODE={mode}")


def validate_existing_run_manifest(manifest_path: str | Path, formal_requested: bool) -> dict:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if formal_requested and manifest.get("status") != "FORMAL":
        raise RuntimeError(
            "A Stage 13 smoke checkpoint/run directory cannot be reused as a formal result. "
            f"Delete only this smoke run directory: {path.parent}"
        )
    return manifest


def prepare_fold_v43(root: str | Path, master: pd.DataFrame, fold_id: int) -> dict:
    root = Path(root); fold = get_fold_v43(fold_id); p = V43Protocol()
    paths = fold_qwen_paths(root, fold_id)
    train_master, _ = split_master_for_fold(master, fold)
    frames = build_fold_qwen_datasets(
        master, fold, paths["data_dir"],
        history_steps=p.generator_history_steps,
        future_steps=p.generator_future_steps,
    )
    audit = audit_fold_isolation(fold, train_master=train_master, qwen_frames=frames)
    if not audit["pass"]:
        raise RuntimeError(f"Stage 13 fold preparation isolation audit failed: {audit}")
    out = root / "reports" / "v43" / "audit" / f"fold{fold_id:02d}_prep_audit.json"
    _json_dump(out, audit)
    return audit


def _qwen_update_for_run(root: Path, key: V43RunKey) -> pd.DataFrame | None:
    if key.condition != "qwen_additive":
        return None
    path = fold_qwen_paths(root, key.fold_id)["updates_dir"] / f"qwen_Q050_A{key.allocation}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _history_frame(fit: dict) -> pd.DataFrame:
    h = pd.DataFrame(fit.get("history", []))
    if h.empty:
        return pd.DataFrame(columns=["step","train_rmse","val_rmse","oi","lr"])
    return h


def run_one_v43(
    root: str | Path,
    run_key: V43RunKey,
    max_steps: int | None = None,
    device: str = "auto",
) -> Path:
    root = Path(root); p = V43Protocol(); fold = get_fold_v43(run_key.fold_id)
    steps = int(p.max_optimizer_steps if max_steps is None else max_steps)
    formal = formal_run_allowed_v43(steps)
    directory = checkpoint_dir_v43(root, run_key)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.pt"
    manifest_path = directory / "run_manifest.json"
    master = pd.read_csv(root / "data/processed/rutting_master_v31.csv")

    if checkpoint.exists():
        if not manifest_path.exists():
            raise RuntimeError(f"Stage 13 checkpoint exists without run manifest: {checkpoint}")
        validate_existing_run_manifest(manifest_path, formal_requested=formal)
        fit = load_v43_checkpoint(checkpoint, device=device)
    else:
        result = fit_v43_run(root, master, run_key, max_steps=steps, device=device)
        fit = result["fit"]
        metadata = {
            "formal": formal,
            "status": "FORMAL" if formal else "SMOKE_ONLY",
            "post_hoc": True,
            "structural_ood": True,
            "max_steps": steps,
            "fold_id": fold.fold_id,
            "group_name": fold.group_name,
            "held_out_structures": list(fold.held_out_structures),
            "train_structures": list(fold.train_structures),
            "dev_metrics": result["dev_metrics"],
            "training_counts": result["training_counts"],
            "isolation_audit": result["isolation_audit"],
            "protocol_signature": protocol_signature_v43(),
        }
        save_v43_checkpoint(checkpoint, run_key, fit, result["model_config"], metadata)
        _json_dump(directory / "dev_metrics.json", result["dev_metrics"])
        _json_dump(directory / "training_counts.json", result["training_counts"])

    _history_frame(fit).to_csv(directory / "history.csv", index=False)

    horizon_frames=[]
    for horizon in p.horizons:
        pred_path = directory / f"pred_H{int(horizon):03d}.csv"
        if pred_path.exists():
            pred = pd.read_csv(pred_path)
        else:
            pred = forecast_ood_horizon(fit, master, fold, int(horizon), run_key)
            pred.to_csv(pred_path, index=False)
        horizon_frames.append(pred)

    fixed_path = directory / "pred_fixed_origin_2021.csv"
    if fixed_path.exists():
        fixed = pd.read_csv(fixed_path)
    else:
        fixed = forecast_ood_fixed_origin(fit, master, fold, run_key)
        fixed.to_csv(fixed_path, index=False)

    horizon_all = pd.concat(horizon_frames, ignore_index=True) if horizon_frames else pd.DataFrame()
    summaries = summarize_ood_predictions(horizon_all, fixed)
    summaries["horizon_metrics"].to_csv(directory / "horizon_metrics.csv", index=False)
    summaries["step_metrics"].to_csv(directory / "step_metrics.csv", index=False)
    summaries["fixed_origin_metrics"].to_csv(directory / "fixed_origin_metrics.csv", index=False)

    qwen_update = _qwen_update_for_run(root, run_key)
    train_meta = build_v43_training_arrays(master, fold, run_key.condition, qwen_update=qwen_update)["meta"]
    dev_meta = build_v43_dev_arrays(master, fold)["meta"]
    test_pred = pd.concat([horizon_all, fixed], ignore_index=True) if len(fixed) else horizon_all
    isolation = audit_fold_isolation(
        fold,
        train_master=split_master_for_fold(master, fold)[0],
        synthetic_update=qwen_update,
        downstream_train_meta=train_meta,
        downstream_dev_meta=dev_meta,
        test_predictions=test_pred,
    )
    if not isolation["pass"]:
        raise RuntimeError(f"Stage 13 post-run isolation audit failed: {isolation}")
    _json_dump(directory / "isolation_audit.json", isolation)

    manifest = {
        "stage": "V4.3 Stage 13",
        "post_hoc": True,
        "structural_ood": True,
        "status": "FORMAL" if formal else "SMOKE_ONLY",
        "max_steps": steps,
        "run_key": asdict(run_key),
        "fold": asdict(fold),
        "protocol_signature": protocol_signature_v43(),
        "horizons": list(p.horizons),
        "primary_horizon": p.primary_horizon,
        "steps_completed": fit.get("steps_completed"),
        "parameter_count": fit.get("parameter_count"),
        "checkpoint": str(checkpoint),
        "isolation_audit_pass": True,
    }
    _json_dump(manifest_path, manifest)
    return directory


def run_stage13_training(
    root: str | Path,
    fold_ids: tuple[int, ...],
    max_steps: int | None = None,
    device: str = "auto",
    max_runs: int | None = None,
) -> list[Path]:
    plan = downstream_run_plan_v43(fold_ids)
    if max_runs is not None:
        plan = plan[: int(max_runs)]
    started = time.time(); outputs=[]
    for i, key in enumerate(plan, 1):
        out = run_one_v43(root, key, max_steps=max_steps, device=device)
        outputs.append(out)
        print(f"[Stage13 train] {i:03d}/{len(plan):03d} {key.stem} done elapsed={(time.time()-started)/60:.1f} min", flush=True)
    return outputs


def _collect_formal_artifacts(root: Path) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    horizon_metrics=[]; fixed_metrics=[]; horizon_predictions=[]; fixed_predictions=[]
    base = root / "checkpoints" / "v43"
    if not base.exists():
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    for manifest_path in sorted(base.rglob("run_manifest.json")):
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "FORMAL":
            continue
        d=manifest_path.parent
        for path, bucket in ((d/"horizon_metrics.csv",horizon_metrics),(d/"fixed_origin_metrics.csv",fixed_metrics)):
            if path.exists(): bucket.append(pd.read_csv(path))
        for path in sorted(d.glob("pred_H*.csv")):
            horizon_predictions.append(pd.read_csv(path))
        fp=d/"pred_fixed_origin_2021.csv"
        if fp.exists(): fixed_predictions.append(pd.read_csv(fp))
    return (
        pd.concat(horizon_metrics,ignore_index=True) if horizon_metrics else pd.DataFrame(),
        pd.concat(fixed_metrics,ignore_index=True) if fixed_metrics else pd.DataFrame(),
        pd.concat(horizon_predictions,ignore_index=True) if horizon_predictions else pd.DataFrame(),
        pd.concat(fixed_predictions,ignore_index=True) if fixed_predictions else pd.DataFrame(),
    )


def fold_condition_means(
    metrics: pd.DataFrame,
    value_cols: list[str] | tuple[str, ...],
    horizon_col: str | None = None,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    group_cols = ["fold_id", "group_name", "condition"]
    if horizon_col:
        group_cols.append(horizon_col)
    cols = [c for c in value_cols if c in metrics.columns]
    if not cols:
        return pd.DataFrame(columns=group_cols)
    return metrics.groupby(group_cols, as_index=False)[cols].mean()


def macro_fold_comparison(
    metrics: pd.DataFrame,
    value_col: str,
    horizon_col: str | None = None,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    group_cols=["fold_id","group_name","condition"]
    if horizon_col:
        group_cols.append(horizon_col)
    fold_condition=(metrics.groupby(group_cols,as_index=False)[value_col].mean())
    index_cols=["fold_id","group_name"]+([horizon_col] if horizon_col else [])
    pivot=fold_condition.pivot_table(index=index_cols,columns="condition",values=value_col,aggfunc="mean").reset_index()
    required={"full_real","qwen_additive"}
    if not required.issubset(pivot.columns):
        return pd.DataFrame()
    pivot["relative_change"]=(pivot["qwen_additive"]-pivot["full_real"])/pivot["full_real"]
    summary_cols=[horizon_col] if horizon_col else []
    groups=pivot.groupby(summary_cols,dropna=False,sort=True) if summary_cols else [((),pivot)]
    rows=[]
    for keys,g in groups:
        if summary_cols and not isinstance(keys,tuple): keys=(keys,)
        rec=dict(zip(summary_cols,keys if summary_cols else ()))
        rec.update({
            "n_folds":int(len(g)),
            "folds_improved":int(pd.to_numeric(g["relative_change"]).lt(0).sum()),
            "mean_relative_change":float(pd.to_numeric(g["relative_change"]).mean()),
            "median_relative_change":float(pd.to_numeric(g["relative_change"]).median()),
            "mean_full_real":float(pd.to_numeric(g["full_real"]).mean()),
            "mean_qwen_additive":float(pd.to_numeric(g["qwen_additive"]).mean()),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def _micro_horizon_summary(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty: return pd.DataFrame()
    rows=[]
    for (condition,h),g in pred.groupby(["condition","requested_horizon"],sort=True):
        y=g.y_true.to_numpy(float); p=g.y_pred.to_numpy(float); per=g.persistence_pred.to_numpy(float)
        rmse=float(np.sqrt(mean_squared_error(y,p))); prmse=float(np.sqrt(mean_squared_error(y,per)))
        rows.append({"condition":condition,"requested_horizon":int(h),"rmse":rmse,"mae":float(mean_absolute_error(y,p)),
                     "r2":float(r2_score(y,p)) if len(y)>=2 and np.std(y)>0 else np.nan,
                     "persistence_rmse":prmse,"persistence_skill":float(1-rmse/prmse) if prmse>0 else np.nan,
                     "n_rows":int(len(g)),"n_folds":int(g.fold_id.nunique()),"n_structures":int(g.STR_name.nunique())})
    return pd.DataFrame(rows)


def _micro_fixed_summary(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty: return pd.DataFrame()
    rows=[]
    for condition,g in pred.groupby("condition",sort=True):
        y=g.y_true.to_numpy(float); p=g.y_pred.to_numpy(float); per=g.persistence_pred.to_numpy(float)
        rmse=float(np.sqrt(mean_squared_error(y,p))); prmse=float(np.sqrt(mean_squared_error(y,per)))
        rows.append({"condition":condition,"rmse":rmse,"mae":float(mean_absolute_error(y,p)),
                     "r2":float(r2_score(y,p)) if len(y)>=2 and np.std(y)>0 else np.nan,
                     "persistence_rmse":prmse,"persistence_skill":float(1-rmse/prmse) if prmse>0 else np.nan,
                     "n_rows":int(len(g)),"n_folds":int(g.fold_id.nunique()),"n_structures":int(g.STR_name.nunique())})
    return pd.DataFrame(rows)


def _paired_fixed_bootstrap(full_real: pd.DataFrame, additive: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows=[]
    for fold_id in sorted(pd.to_numeric(additive.fold_id,errors="coerce").dropna().astype(int).unique()):
        real_fold=full_real[pd.to_numeric(full_real.fold_id).eq(fold_id)]
        add_fold=additive[pd.to_numeric(additive.fold_id).eq(fold_id)]
        frames=[]
        for (allocation,seed),gm in add_fold.groupby(["allocation","seed"],sort=True):
            gr=real_fold[pd.to_numeric(real_fold.seed).eq(int(seed))]
            if gr.empty: continue
            keys=["STR_name","origin_observation_index","target_observation_index","step_h"]
            j=gr[keys+["y_true","y_pred"]].merge(gm[keys+["y_true","y_pred"]],on=keys,how="inner",suffixes=("_real","_mix"),validate="one_to_one")
            if j.empty: continue
            if not np.allclose(j.y_true_real,j.y_true_mix,atol=1e-9,rtol=0):
                raise RuntimeError("Stage 13 fixed-origin paired target mismatch")
            frames.append(pd.DataFrame({"STR_name":j.STR_name.astype(str),"y_true":j.y_true_real.astype(float),"real_pred":j.y_pred_real.astype(float),"mix_pred":j.y_pred_mix.astype(float)}))
        if frames:
            b=hierarchical_cluster_relative_bootstrap(frames,n_boot=int(n_boot),seed=20260818+fold_id)
            rows.append({"fold_id":fold_id,"n_paired_runs":int(b["n_runs"]),"n_clusters":int(b["n_clusters"]),
                         "mean_relative_degradation":float(b["mean_relative_degradation"]),"ci_low":float(b["ci_low"]),"ci_high":float(b["ci_high"]),
                         "superior_to_full_real":bool(float(b["ci_high"])<0)})
    return pd.DataFrame(rows)


def build_stage13_reports(root: str | Path, n_boot: int = 10000) -> Path:
    root=Path(root); report=root/"reports"/"v43"; report.mkdir(parents=True,exist_ok=True)
    metrics_dir=report/"metrics"; stats_dir=report/"statistics"; audit_dir=report/"audit"
    for d in (metrics_dir,stats_dir,audit_dir): d.mkdir(parents=True,exist_ok=True)

    horizon_metrics,fixed_metrics,horizon_pred,fixed_pred=_collect_formal_artifacts(root)
    if len(horizon_metrics): horizon_metrics.to_csv(metrics_dir/"structural_ood_horizon_run_metrics.csv",index=False)
    if len(fixed_metrics): fixed_metrics.to_csv(metrics_dir/"structural_ood_fixed_origin_run_metrics.csv",index=False)
    if len(horizon_pred):
        fold_h=fold_condition_means(
            horizon_metrics,
            ["trajectory_rmse","trajectory_mae","trajectory_r2","persistence_rmse","persistence_skill","n_rows","n_origins"],
            horizon_col="requested_horizon",
        )
        fold_h.to_csv(stats_dir/"fold_horizon_condition_means.csv",index=False)
        macro_h=macro_fold_comparison(horizon_metrics,"trajectory_rmse","requested_horizon")
        macro_h.to_csv(stats_dir/"macro_horizon_comparison.csv",index=False)
        _micro_horizon_summary(horizon_pred).to_csv(stats_dir/"micro_horizon_summary.csv",index=False)
        pair=[]
        for fold_id in sorted(pd.to_numeric(horizon_pred.fold_id,errors="coerce").dropna().astype(int).unique()):
            g=horizon_pred[pd.to_numeric(horizon_pred.fold_id).eq(fold_id)]
            real=g[g.condition.astype(str).eq("full_real")]
            add=g[g.condition.astype(str).eq("qwen_additive")]
            if real.empty or add.empty: continue
            print(f"[Stage13 report] rolling bootstrap fold={fold_id} n_boot={int(n_boot)}",flush=True)
            comp=paired_noninferiority_by_horizon(real,add,margin=0.0,n_boot=int(n_boot),seed=20260818+fold_id*1000)
            if len(comp):
                comp.insert(0,"fold_id",fold_id); comp.insert(1,"group_name",get_fold_v43(fold_id).group_name)
                comp["superior_to_full_real"]=pd.to_numeric(comp.upper_ci_relative_degradation,errors="coerce").lt(0)
                pair.append(comp)
        if pair: pd.concat(pair,ignore_index=True).to_csv(stats_dir/"paired_bootstrap_horizon.csv",index=False)
    if len(fixed_pred):
        fold_f=fold_condition_means(
            fixed_metrics,
            ["rmse","mae","r2","persistence_rmse","persistence_skill","n_rows","n_structures"],
            horizon_col=None,
        )
        fold_f.to_csv(stats_dir/"fold_fixed_origin_condition_means.csv",index=False)
        macro_f=macro_fold_comparison(fixed_metrics,"rmse",None)
        macro_f.to_csv(stats_dir/"macro_fixed_origin_comparison.csv",index=False)
        _micro_fixed_summary(fixed_pred).to_csv(stats_dir/"micro_fixed_origin_summary.csv",index=False)
        print(f"[Stage13 report] fixed-origin bootstrap n_boot={int(n_boot)}",flush=True)
        fixed_boot=_paired_fixed_bootstrap(fixed_pred[fixed_pred.condition.eq("full_real")],fixed_pred[fixed_pred.condition.eq("qwen_additive")],int(n_boot))
        if len(fixed_boot): fixed_boot.to_csv(stats_dir/"paired_bootstrap_fixed_origin.csv",index=False)

    group_rows=[]
    for fold_id in range(1,8):
        fold=get_fold_v43(fold_id)
        for s in fold.held_out_structures:
            group_rows.append({"fold_id":fold_id,"group_name":fold.group_name,"held_out_structure":s})
    pd.DataFrame(group_rows).to_csv(report/"structure_group_map.csv",index=False)

    manifests=[]
    base=root/"checkpoints"/"v43"
    if base.exists():
        for path in base.rglob("run_manifest.json"):
            m=json.loads(path.read_text(encoding="utf-8")); rk=m.get("run_key",{})
            manifests.append({**rk,"status":m.get("status"),"max_steps":m.get("max_steps"),"isolation_audit_pass":m.get("isolation_audit_pass")})
    mdf=pd.DataFrame(manifests)
    if len(mdf): mdf.to_csv(audit_dir/"stage13_run_manifest.csv",index=False)

    prep_pass=[]; qwen_folds=[]; synth_counts={}
    for fold_id in range(1,8):
        prep=audit_dir/f"fold{fold_id:02d}_prep_audit.json"
        prep_pass.append(prep.exists() and json.loads(prep.read_text(encoding="utf-8")).get("pass") is True)
        paths=fold_qwen_paths(root,fold_id)
        qwen_manifest=paths["report_dir"]/"training_manifest.json"
        adapter=paths["model_dir"]/"refit"/"refit_adapter"
        qwen_folds.append(qwen_manifest.exists() and adapter.exists())
        synth_counts[fold_id]=sum((paths["updates_dir"]/f"qwen_Q050_A{i}.csv").exists() for i in range(1,6))
    expected_runs=len(downstream_run_plan_v43())
    formal_runs=int((mdf.status=="FORMAL").sum()) if len(mdf) else 0
    smoke_runs=int((mdf.status=="SMOKE_ONLY").sum()) if len(mdf) else 0
    isolation_all=bool(len(mdf) and mdf.loc[mdf.status.eq("FORMAL"),"isolation_audit_pass"].fillna(False).all()) if formal_runs else False
    complete=(all(prep_pass) and all(qwen_folds) and all(v==5 for v in synth_counts.values()) and formal_runs==expected_runs and isolation_all)
    summary={
        "stage":"V4.3 Stage 13","post_hoc":True,"structural_ood":True,"status":"COMPLETE" if complete else "PARTIAL",
        "protocol_signature":protocol_signature_v43(),"primary_horizon":6,"secondary_horizon":12,
        "expected_formal_runs":expected_runs,"formal_runs":formal_runs,"smoke_runs":smoke_runs,"n_boot":int(n_boot),
        "prep_folds_passed":int(sum(prep_pass)),"qwen_folds_complete":int(sum(qwen_folds)),"synthetic_updates_per_fold":synth_counts,
        "formal_isolation_audits_all_pass":isolation_all,"identity_feature_audit":audit_identity_free_features(),
        "interpretation":"Leave-one-structure-group-out within RIOHTrack; not external validation on arbitrary pavement systems.",
    }
    _json_dump(report/"stage13_summary.json",summary)
    return report


def _run_prep(root: Path, master: pd.DataFrame, fold_ids: tuple[int,...]):
    for i,fold_id in enumerate(fold_ids,1):
        print(f"[Stage13 prep] {i}/{len(fold_ids)} fold={fold_id}",flush=True)
        prepare_fold_v43(root,master,fold_id)


def _run_qwen(root: Path, fold_ids: tuple[int,...]):
    for i,fold_id in enumerate(fold_ids,1):
        fold=get_fold_v43(fold_id)
        print(f"[Stage13 qwen] {i}/{len(fold_ids)} {fold.stem}",flush=True)
        train_fold_qwen(root,fold)


def _run_synth(root: Path, master: pd.DataFrame, fold_ids: tuple[int,...]):
    for i,fold_id in enumerate(fold_ids,1):
        fold=get_fold_v43(fold_id)
        print(f"[Stage13 synth] {i}/{len(fold_ids)} {fold.stem}",flush=True)
        generate_fold_q50_updates(root,master,fold)


def main(root: str | Path | None = None) -> None:
    root=Path(root or Path(__file__).resolve().parents[1])
    mode=os.environ.get("V43_MODE","prep").strip().lower()
    fold_ids=parse_fold_ids(os.environ.get("V43_FOLD",""))
    master=pd.read_csv(root/"data/processed/rutting_master_v31.csv")
    steps=int(os.environ.get("V43_MAX_STEPS",str(V43Protocol().max_optimizer_steps)))
    max_runs_raw=os.environ.get("V43_MAX_RUNS","").strip(); max_runs=int(max_runs_raw) if max_runs_raw else None
    n_boot=int(os.environ.get("V43_BOOTSTRAP_N","10000"))

    if mode=="prep": _run_prep(root,master,fold_ids)
    elif mode=="qwen": _run_qwen(root,fold_ids)
    elif mode=="synth": _run_synth(root,master,fold_ids)
    elif mode=="train": run_stage13_training(root,fold_ids,max_steps=steps,device="auto",max_runs=max_runs)
    elif mode=="report": build_stage13_reports(root,n_boot=n_boot)
    elif mode=="all":
        _run_prep(root,master,fold_ids); _run_qwen(root,fold_ids); _run_synth(root,master,fold_ids)
        run_stage13_training(root,fold_ids,max_steps=steps,device="auto",max_runs=max_runs)
        build_stage13_reports(root,n_boot=n_boot)
    else: raise ValueError(f"unsupported V43_MODE={mode}")


if __name__=="__main__":
    main()
