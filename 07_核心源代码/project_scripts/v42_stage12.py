from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import os
import time
import pandas as pd

from v40_metrics import summarize_prediction_rows, paired_noninferiority_by_horizon
from v40_openloop import build_eval_arrays, forecast_all_origins
from v42_analysis import (
    audit_existing_model_selection,
    horizon_error_accumulation,
    paired_horizon_slope_change,
    summarize_slope_change,
)
from v42_checkpoint import save_v42_checkpoint, load_v42_checkpoint
from v42_fit import fit_v42_run
from v42_protocol import (
    V42Protocol,
    V42RunKey,
    checkpoint_dir_v42,
    curve_probe_plan,
    formal_run_allowed_v42,
    gru_large_sensitivity_plan,
    protocol_signature_v42,
    tdr_ablation_plan,
)


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def _annotate_predictions(frame: pd.DataFrame, key: V42RunKey) -> pd.DataFrame:
    out = frame.copy()
    q = 0.5 if key.condition == 'qwen_additive' else 0.0
    values = [
        ('seed', int(key.seed)),
        ('allocation', int(key.allocation)),
        ('synthetic_fraction', float(q)),
        ('method', f'v42|{key.experiment}|{key.variant}|{key.condition}'),
        ('model', str(key.model)),
        ('condition', str(key.condition)),
        ('variant', str(key.variant)),
        ('experiment', str(key.experiment)),
    ]
    for column, value in values:
        if column in out.columns:
            out[column] = value
        else:
            out.insert(0, column, value)
    return out


def _history_frame(fit: dict) -> pd.DataFrame:
    hist = pd.DataFrame(fit.get('history', []))
    if hist.empty:
        return pd.DataFrame(columns=['step','train_rmse','val_rmse','oi','lr'])
    return hist


def run_one_v42(
    root: str | Path,
    run_key: V42RunKey,
    max_steps: int | None = None,
    device: str = 'auto',
) -> Path:
    root = Path(root)
    p = V42Protocol()
    steps = int(p.max_optimizer_steps if max_steps is None else max_steps)
    formal = formal_run_allowed_v42(steps)
    directory = checkpoint_dir_v42(root, run_key)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / 'checkpoint.pt'
    manifest_path = directory / 'run_manifest.json'
    master = pd.read_csv(root / 'data/processed/rutting_master_v31.csv')

    if checkpoint.exists():
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text(encoding='utf-8'))
            if formal and old.get('status') != 'FORMAL':
                raise RuntimeError(
                    'A Stage 12 smoke checkpoint cannot be reused as a formal result. '
                    f'Delete only this smoke directory: {directory}'
                )
        fit = load_v42_checkpoint(checkpoint, device=device)
    else:
        result = fit_v42_run(root, master, run_key, max_steps=steps, device=device)
        fit = result['fit']
        metadata = {
            'formal': formal,
            'status': 'FORMAL' if formal else 'SMOKE_ONLY',
            'post_hoc': True,
            'max_steps': steps,
            'dev_metrics': result['dev_metrics'],
            'training_counts': result['training_counts'],
            'protocol_signature': protocol_signature_v42(),
        }
        save_v42_checkpoint(checkpoint, run_key, fit, result['model_config'], metadata)
        _json_dump(directory / 'training_counts.json', result['training_counts'])
        _json_dump(directory / 'dev_metrics.json', result['dev_metrics'])

    _history_frame(fit).to_csv(directory / 'history.csv', index=False)

    eval_arrays = build_eval_arrays(master, seq_len=p.seq_len)
    for horizon in p.horizons:
        pred_path = directory / f'pred_H{horizon:03d}.csv'
        if pred_path.exists():
            continue
        pred = forecast_all_origins(fit, eval_arrays, int(horizon), run_key.model)
        pred = _annotate_predictions(pred, run_key)
        pred.to_csv(pred_path, index=False)

    all_predictions = pd.concat(
        [pd.read_csv(directory / f'pred_H{h:03d}.csv') for h in p.horizons],
        ignore_index=True,
    )
    metrics = summarize_prediction_rows(all_predictions)['horizon_metrics']
    for col, value in (
        ('experiment', run_key.experiment),
        ('variant', run_key.variant),
        ('condition', run_key.condition),
    ):
        if col not in metrics.columns:
            metrics.insert(0, col, value)
        else:
            metrics[col] = value
    metrics.to_csv(directory / 'horizon_metrics.csv', index=False)

    manifest = {
        'stage': 'V4.2 Stage 12',
        'post_hoc': True,
        'status': 'FORMAL' if formal else 'SMOKE_ONLY',
        'max_steps': steps,
        'run_key': asdict(run_key),
        'protocol_signature': protocol_signature_v42(),
        'horizons': list(p.horizons),
        'primary_horizon': p.primary_horizon,
        'steps_completed': fit.get('steps_completed'),
        'parameter_count': fit.get('parameter_count'),
        'checkpoint': str(checkpoint),
    }
    _json_dump(manifest_path, manifest)
    return directory


def _plan_for_mode(mode: str) -> list[V42RunKey]:
    mode = str(mode).strip().lower()
    if mode in {'gru', 'gru_large', 'fairness'}:
        return gru_large_sensitivity_plan()
    if mode in {'tdr', 'ablation', 'tdr_ablation'}:
        return tdr_ablation_plan()
    if mode in {'curve', 'curves', 'curve_probe'}:
        return curve_probe_plan()
    if mode in {'existing', 'report', 'reports', 'report_only'}:
        return []
    if mode in {'all', ''}:
        return gru_large_sensitivity_plan() + tdr_ablation_plan() + curve_probe_plan()
    raise ValueError(f'unsupported V42_MODE={mode}')


def run_stage12(
    root: str | Path,
    mode: str = 'all',
    max_steps: int | None = None,
    device: str = 'auto',
    max_runs: int | None = None,
) -> list[Path]:
    plan = _plan_for_mode(mode)
    if max_runs is not None:
        plan = plan[: int(max_runs)]
    started = time.time()
    done: list[Path] = []
    for i, key in enumerate(plan, 1):
        out = run_one_v42(root, key, max_steps=max_steps, device=device)
        done.append(out)
        elapsed = time.time() - started
        print(f'[Stage12] {i:03d}/{len(plan):03d} {key.stem} done elapsed={elapsed/60:.1f} min', flush=True)
    return done


def _collect_run_artifacts(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics=[]; predictions=[]
    base=root/'checkpoints/v42'
    if not base.exists():
        return pd.DataFrame(), pd.DataFrame()
    for manifest_path in sorted(base.rglob('run_manifest.json')):
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('status')!='FORMAL':
            continue
        d=manifest_path.parent
        mp=d/'horizon_metrics.csv'
        if mp.exists():
            metrics.append(pd.read_csv(mp))
        for p in sorted(d.glob('pred_H*.csv')):
            predictions.append(pd.read_csv(p))
    return (
        pd.concat(metrics,ignore_index=True) if metrics else pd.DataFrame(),
        pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame(),
    )


def _existing_horizon_validation(root: Path, out_dir: Path) -> None:
    v40p=root/'reports/v40/metrics/multi_horizon_run_metrics.csv'
    v41p=root/'reports/v41/metrics/multi_horizon_run_metrics.csv'
    if not (v40p.exists() and v41p.exists()):
        return
    v40=pd.read_csv(v40p); v41=pd.read_csv(v41p)
    base=v40[(v40['method'].astype(str)=='full_real') & pd.to_numeric(v40['synthetic_fraction']).eq(0)].copy()
    add=v41[v41['condition'].astype(str).eq('qwen_additive')].copy()
    base['condition']='full_real'; base['variant']='frozen'
    add['variant']='frozen'
    pairs=[]
    for model in ('gru','tdr'):
        p=paired_horizon_slope_change(base[base.model.eq(model)],add[add.model.eq(model)])
        if len(p): pairs.append(p)
    paired=pd.concat(pairs,ignore_index=True) if pairs else pd.DataFrame()
    if len(paired):
        paired.to_csv(out_dir/'existing_v40_v41_paired_horizon_slope_change.csv',index=False)
        summarize_slope_change(paired).to_csv(out_dir/'existing_v40_v41_horizon_slope_summary.csv',index=False)


def _training_curve_summary(root: Path, out_dir: Path) -> None:
    rows=[]
    base=root/'checkpoints/v42/curve_probe'
    if not base.exists(): return
    for hist_path in sorted(base.rglob('history.csv')):
        manifest_path=hist_path.parent/'run_manifest.json'
        if not manifest_path.exists(): continue
        m=json.loads(manifest_path.read_text(encoding='utf-8'))
        if m.get('status')!='FORMAL': continue
        h=pd.read_csv(hist_path)
        if h.empty: continue
        key=m['run_key']; best=h.loc[pd.to_numeric(h.val_rmse).idxmin()]
        rows.append({
            'model':key['model'],'condition':key['condition'],'allocation':key['allocation'],'seed':key['seed'],
            'best_step':int(best['step']),'best_train_rmse':float(best['train_rmse']),
            'best_val_rmse':float(best['val_rmse']),'best_generalization_gap':float(best['val_rmse']-best['train_rmse']),
            'best_overfitting_index':float(best.get('oi',float('nan'))),
        })
    if rows: pd.DataFrame(rows).to_csv(out_dir/'training_curve_probe_summary.csv',index=False)


def build_stage12_reports(root: str | Path, n_boot: int = 10000) -> Path:
    root=Path(root); report=root/'reports/v42'; report.mkdir(parents=True,exist_ok=True)
    fairness=report/'fairness'; fairness.mkdir(exist_ok=True)
    mechanism=report/'mechanism'; mechanism.mkdir(exist_ok=True)
    audit,capacity=audit_existing_model_selection(root)
    audit.to_csv(fairness/'model_selection_provenance.csv',index=False)
    capacity.to_csv(fairness/'existing_gru_capacity_ladder_summary.csv',index=False)
    _existing_horizon_validation(root,mechanism)
    _training_curve_summary(root,mechanism)

    metrics,predictions=_collect_run_artifacts(root)
    if len(metrics):
        metrics.to_csv(report/'multi_horizon_run_metrics.csv',index=False)
        slope=horizon_error_accumulation(metrics)
        slope.to_csv(mechanism/'v42_horizon_error_accumulation.csv',index=False)
    superiority=[]
    if len(predictions):
        for (experiment,model,variant),g in predictions.groupby(['experiment','model','variant'],sort=True):
            real=g[g.condition.astype(str).eq('full_real')]
            aug=g[g.condition.astype(str).eq('qwen_additive')]
            if real.empty or aug.empty: continue
            print(f'[Stage12 report] bootstrap {experiment} {model} {variant} n_boot={int(n_boot)}',flush=True)
            comp=paired_noninferiority_by_horizon(real,aug,margin=0.0,n_boot=int(n_boot),seed=20260818)
            if len(comp):
                comp.insert(0,'variant',variant); comp.insert(0,'model',model); comp.insert(0,'experiment',experiment)
                comp['superior_to_full_real']=pd.to_numeric(comp['upper_ci_relative_degradation'],errors='coerce').lt(0)
                superiority.append(comp)
        if superiority:
            pd.concat(superiority,ignore_index=True).to_csv(mechanism/'augmentation_superiority_by_variant.csv',index=False)

    manifests=[]
    base=root/'checkpoints/v42'
    if base.exists():
        for p in base.rglob('run_manifest.json'):
            m=json.loads(p.read_text(encoding='utf-8')); rk=m.get('run_key',{})
            manifests.append({**rk,'status':m.get('status'),'max_steps':m.get('max_steps'),'post_hoc':m.get('post_hoc'),'steps_completed':m.get('steps_completed')})
    manifest_df=pd.DataFrame(manifests)
    if len(manifest_df): manifest_df.to_csv(report/'stage12_run_manifest.csv',index=False)
    summary={
        'stage':'V4.2 Stage 12','post_hoc':True,'protocol_signature':protocol_signature_v42(),
        'formal_runs':int((manifest_df.status=='FORMAL').sum()) if len(manifest_df) else 0,
        'smoke_runs':int((manifest_df.status=='SMOKE_ONLY').sum()) if len(manifest_df) else 0,
        'n_boot':int(n_boot),
    }
    _json_dump(report/'stage12_summary.json',summary)
    return report



def _should_build_reports() -> bool:
    raw=os.environ.get('V42_SKIP_REPORTS','0').strip().lower()
    return raw not in {'1','true','yes','y','on'}

def main(root: str | Path | None = None) -> None:
    root=Path(root or Path(__file__).resolve().parents[1])
    mode=os.environ.get('V42_MODE','all')
    steps=int(os.environ.get('V42_MAX_STEPS',str(V42Protocol().max_optimizer_steps)))
    max_runs_raw=os.environ.get('V42_MAX_RUNS','').strip()
    max_runs=int(max_runs_raw) if max_runs_raw else None
    run_stage12(root,mode=mode,max_steps=steps,device='auto',max_runs=max_runs)
    if _should_build_reports():
        n_boot=int(os.environ.get('V42_BOOTSTRAP_N','10000'))
        build_stage12_reports(root,n_boot=n_boot)


if __name__=='__main__':
    main()
