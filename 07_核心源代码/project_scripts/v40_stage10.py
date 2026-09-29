from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np
import pandas as pd
from v31_experiment_engine import fixed_real_evaluation_arrays
from v31_gru import predict_gru
from v31_tdr import predict_tdr
from v40_protocol import V40Protocol, RunKey, phase_a_run_plan, phase_b_run_plan, checkpoint_dir, protocol_signature
from v40_checkpoint import save_fit_checkpoint, load_fit_checkpoint
from v40_refit import fit_frozen_run, load_frozen_model_configs, find_v31_reference_prediction
from v40_openloop import build_eval_arrays, forecast_all_origins
from v40_audit import audit_one_step_reproduction, run_openloop_leakage_audit
from v40_metrics import summarize_prediction_rows, compute_error_growth, horizon_degradation_table


def _json_dump(path: Path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=True,default=str),encoding='utf-8')


def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def formal_run_allowed(max_steps: int) -> bool:
    return int(max_steps) == V40Protocol().max_optimizer_steps


def _audit_accepted(path: Path, require_formal: bool = False) -> bool:
    if not path.exists(): return False
    try: a=json.loads(path.read_text(encoding='utf-8'))
    except Exception: return False
    allowed={'PASS'} if require_formal else {'PASS','SMOKE_ONLY'}
    return a.get('status') in allowed


def _manifest_is_formal(path: Path) -> bool:
    if not path.exists(): return False
    try: m=json.loads(path.read_text(encoding='utf-8'))
    except Exception: return False
    return bool(m.get('formal')) and int(m.get('max_steps',-1)) == V40Protocol().max_optimizer_steps


def is_run_complete(root, run_key: RunKey, require_formal: bool = False) -> bool:
    d=checkpoint_dir(root,run_key)
    req=[d/'checkpoint.pt',d/'run_manifest.json',d/'reproduction_audit.json',d/'metrics.json']
    req += [d/f'pred_H{h:03d}.csv' for h in V40Protocol().horizons]
    if not all(x.exists() for x in req): return False
    if require_formal and not _manifest_is_formal(d/'run_manifest.json'): return False
    return _audit_accepted(d/'reproduction_audit.json',require_formal=require_formal)


def _one_step_predictions(master, fit, model, seq_len=6):
    Xe,ye,pe,pke,me=fixed_real_evaluation_arrays(master,seq_len=seq_len)['eval2021']
    if model=='gru': ph=predict_gru(fit,Xe,pe)
    elif model=='tdr': ph=predict_tdr(fit,Xe,pe,pke)
    else: raise ValueError(model)
    return pd.DataFrame({'STR_name':me.STR_name.astype(str),'Observation_Index':pd.to_numeric(me.Observation_Order).astype(int),
                         'y_true':ye,'y_pred':ph,'prev':pe.reshape(-1)})


def _save_new_checkpoint(root, run_key, result, max_steps):
    d=checkpoint_dir(root,run_key); cfgs=load_frozen_model_configs(root)
    meta={
        'run_stem':run_key.stem,'model':run_key.model,'method':run_key.method,'synthetic_fraction':run_key.q,
        'allocation':run_key.allocation,'seed':run_key.seed,'max_steps':int(max_steps),
        'formal':formal_run_allowed(max_steps),'v40_protocol_signature':protocol_signature(),
        'selected_dev_rmse':result.get('dev_metrics',{}).get('rmse'),
        'steps_completed':result.get('fit',{}).get('steps_completed'),
    }
    if run_key.model=='gru': model_cfg=asdict(cfgs['gru'])
    else: model_cfg={'input_dim':23,'seq_len':V40Protocol().seq_len,'batch_size':64,'stable':asdict(cfgs['tdr'])}
    save_fit_checkpoint(d/'checkpoint.pt',run_key.model,result['fit'],model_cfg,meta)
    _json_dump(d/'run_manifest.json',meta)
    return result['fit'],meta


def _ensure_reproduction_audit(root, run_key, master, fit, max_steps):
    d=checkpoint_dir(root,run_key); ap=d/'reproduction_audit.json'
    if _audit_accepted(ap): return json.loads(ap.read_text(encoding='utf-8'))
    formal=formal_run_allowed(max_steps)
    try:
        new=_one_step_predictions(master,fit,run_key.model,V40Protocol().seq_len)
        old=pd.read_csv(find_v31_reference_prediction(root,run_key))
        audit=audit_one_step_reproduction(new,old,V40Protocol())
    except Exception as e:
        audit={'status':'FAIL','error':repr(e)}
    if not formal:
        audit={'status':'SMOKE_ONLY','formal_reproduction_required':False,'observed_audit_status':audit.get('status'),
               'note':'Non-2400-step runs are smoke tests only and cannot support scientific conclusions.',**{k:v for k,v in audit.items() if k!='status'}}
    _json_dump(ap,audit)
    if formal and audit.get('status')!='PASS':
        raise RuntimeError(f'reproduction audit failed for {run_key.stem}: {audit}')
    return audit


def _write_run_metrics(d: Path):
    frames=[]
    for h in V40Protocol().horizons:
        p=d/f'pred_H{h:03d}.csv'
        if p.exists(): frames.append(pd.read_csv(p))
    if not frames: raise RuntimeError('no multi-horizon predictions to summarize')
    allp=pd.concat(frames,ignore_index=True)
    s=summarize_prediction_rows(allp)
    eg=compute_error_growth(s['step_metrics'])
    hd=horizon_degradation_table(s['horizon_metrics'])
    s['horizon_metrics'].to_csv(d/'horizon_metrics.csv',index=False)
    s['step_metrics'].to_csv(d/'step_metrics.csv',index=False)
    eg.to_csv(d/'error_growth.csv',index=False)
    hd.to_csv(d/'horizon_degradation.csv',index=False)
    payload={
        'horizon_metrics':s['horizon_metrics'].to_dict('records'),
        'step_metrics':s['step_metrics'].to_dict('records'),
        'error_growth':eg.to_dict('records'),
        'horizon_degradation':hd.to_dict('records'),
        'primary_horizon':V40Protocol().primary_horizon,
    }
    _json_dump(d/'metrics.json',payload)
    return payload


def run_one_v40_condition(root, run_key: RunKey, master: pd.DataFrame | None = None, max_steps: int | None = None):
    root=Path(root); p=V40Protocol(); steps=int(p.max_optimizer_steps if max_steps is None else max_steps)
    d=checkpoint_dir(root,run_key); d.mkdir(parents=True,exist_ok=True)
    if master is None: master=pd.read_csv(root/'data/processed/rutting_master_v31.csv')
    cp=d/'checkpoint.pt'
    if cp.exists():
        manifest_path=d/'run_manifest.json'
        if formal_run_allowed(steps) and not _manifest_is_formal(manifest_path):
            raise RuntimeError(
                f'smoke checkpoint cannot be reused for a formal 2400-step run: {d}. '
                'Remove the V40 smoke run directory or run formal Stage 10 from a clean project copy.'
            )
        fit=load_fit_checkpoint(cp,'auto')
        if not manifest_path.exists():
            _json_dump(manifest_path,{'run_stem':run_key.stem,'formal':False,'max_steps':steps,'recovered_manifest':True})
    else:
        result=fit_frozen_run(root,master,run_key,max_steps=steps)
        fit,_=_save_new_checkpoint(root,run_key,result,steps)
    _ensure_reproduction_audit(root,run_key,master,fit,steps)
    missing=[h for h in p.horizons if not (d/f'pred_H{h:03d}.csv').exists()]
    if missing:
        arrays=build_eval_arrays(master,p.seq_len)
        for h in missing:
            pred=forecast_all_origins(fit,arrays,h,run_key.model,run_key=run_key)
            pred.to_csv(d/f'pred_H{h:03d}.csv',index=False)
    metrics=_write_run_metrics(d)
    return {'run_key':run_key,'directory':d,'metrics':metrics,'complete':is_run_complete(root,run_key,require_formal=formal_run_allowed(steps))}


def _filter_plan(plan):
    raw=os.environ.get('V40_MODELS','').strip().lower()
    if not raw: return plan
    allowed={x.strip() for x in raw.split(',') if x.strip()}
    return [r for r in plan if r.model in allowed]


def _run_plan(root, plan, label):
    root=Path(root); plan=_filter_plan(plan); total=len(plan); start=time.time(); results=[]
    env_steps=os.environ.get('V40_MAX_STEPS'); max_steps=int(env_steps) if env_steps else V40Protocol().max_optimizer_steps
    master=pd.read_csv(root/'data/processed/rutting_master_v31.csv')
    require_formal=formal_run_allowed(max_steps)
    for i,r in enumerate(plan,1):
        if is_run_complete(root,r,require_formal=require_formal):
            state='complete'
        else:
            run_one_v40_condition(root,r,master=master,max_steps=max_steps); state='done'
        elapsed=time.time()-start; rate=elapsed/max(i,1); eta=rate*(total-i)
        print(f'[Stage10][{label}] {i:03d}/{total:03d} {r.stem} state={state} elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m')
        results.append(r.stem)
    write_v40_reports(root)
    return results


def stage10_phase_a(root):
    return _run_plan(root,phase_a_run_plan(),'A')


def stage10_phase_b(root):
    return _run_plan(root,phase_b_run_plan(),'B')


def stage10(root, phase='A'):
    phase=str(phase).upper()
    if phase=='A': return stage10_phase_a(root)
    if phase=='B': return stage10_phase_b(root)
    if phase=='ALL': return {'A':stage10_phase_a(root),'B':stage10_phase_b(root)}
    raise ValueError('V40_PHASE must be A, B, or ALL')


def _read_csvs(paths):
    frames=[]
    for path in paths:
        try:
            d=pd.read_csv(path)
            if len(d): frames.append(d)
        except Exception:
            continue
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def _write_placeholder_csv(path, columns):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists(): pd.DataFrame(columns=columns).to_csv(path,index=False)


def _aggregate_run_artifacts(root: Path):
    dirs=sorted({p.parent for p in (root/'checkpoints/v40').glob('**/run_manifest.json')}) if (root/'checkpoints/v40').exists() else []
    audits=[]; hm=[]; sm=[]; eg=[]; hd=[]; preds=[]
    for d in dirs:
        try: manifest=json.loads((d/'run_manifest.json').read_text(encoding='utf-8'))
        except Exception: manifest={}
        if (d/'reproduction_audit.json').exists():
            try: a=json.loads((d/'reproduction_audit.json').read_text(encoding='utf-8'))
            except Exception: a={'status':'UNREADABLE'}
            audits.append({**{k:manifest.get(k) for k in ['run_stem','model','method','synthetic_fraction','allocation','seed','formal','max_steps']},**a})
        for name,target in [('horizon_metrics.csv',hm),('step_metrics.csv',sm),('error_growth.csv',eg),('horizon_degradation.csv',hd)]:
            p=d/name
            if p.exists():
                try: target.append(pd.read_csv(p))
                except Exception: pass
        for h in V40Protocol().horizons:
            p=d/f'pred_H{h:03d}.csv'
            if p.exists():
                try: preds.append(pd.read_csv(p))
                except Exception: pass
    cat=lambda xs: pd.concat(xs,ignore_index=True) if xs else pd.DataFrame()
    return dirs,pd.DataFrame(audits),cat(hm),cat(sm),cat(eg),cat(hd),cat(preds)


def _write_noninferiority_reports(predictions, outdir):
    from v40_metrics import paired_noninferiority_by_horizon, contiguous_mssr_by_horizon
    cols=['model','method','requested_horizon','synthetic_fraction','margin','n_paired_runs','n_clusters','mean_relative_degradation','lower_ci_relative_degradation','upper_ci_relative_degradation','noninferior']
    if predictions.empty or not {'model','method','seed','requested_horizon','y_true','y_pred'}.issubset(predictions.columns):
        noninf=pd.DataFrame(columns=cols)
    else:
        pieces=[]; n_boot=int(os.environ.get('V40_BOOTSTRAP_N','10000'))
        for model in sorted(predictions.model.astype(str).unique()):
            mr=predictions[predictions.model.astype(str).eq(model)]
            real=mr[mr.method.astype(str).eq('full_real')]
            if real.empty: continue
            for method in ['qwen','real_bootstrap','timegan','timeweaver']:
                mix=mr[mr.method.astype(str).eq(method)]
                if mix.empty: continue
                z=paired_noninferiority_by_horizon(real,mix,margin=.10,n_boot=n_boot,seed=20260817)
                if len(z):
                    z.insert(0,'method',method); z.insert(0,'model',model); pieces.append(z)
        noninf=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=cols)
    path=outdir/'substitution/horizon_noninferiority.csv'; path.parent.mkdir(parents=True,exist_ok=True); noninf.to_csv(path,index=False)
    if len(noninf): mssr=contiguous_mssr_by_horizon(noninf,margins=(.10,.05))
    else: mssr=pd.DataFrame(columns=['model','method','requested_horizon','margin','mssr','is_primary_horizon'])
    mssr.to_csv(outdir/'substitution/contiguous_mssr_by_horizon.csv',index=False)
    return noninf,mssr


def _write_fidelity_reports(root: Path, outdir: Path, master):
    from v40_fidelity import assess_synthetic_transition_fidelity
    summary=[]; conditional=[]; plaus=[]; representative=None; representative_meta=None
    updates=root/'data/v31_updates'
    if master is not None and updates.exists():
        for path in sorted(updates.glob('qwen_Q*_A*.csv')):
            if path.name.endswith('_states.csv'): continue
            try:
                stem=path.stem; q=int(stem.split('_Q')[1].split('_A')[0])/100; alloc=int(stem.rsplit('_A',1)[1])
                res=assess_synthetic_transition_fidelity(master,pd.read_csv(path))
                summary.append({'method':'qwen','synthetic_fraction':q,'allocation':alloc,**res['summary'],**res['plausibility']})
                c=res['conditional'].copy(); c.insert(0,'allocation',alloc); c.insert(0,'synthetic_fraction',q); conditional.append(c)
                plaus.append({'synthetic_fraction':q,'allocation':alloc,**res['plausibility']})
                if representative is None or path.name=='qwen_Q050_A1.csv':
                    representative=res; representative_meta={'synthetic_fraction':q,'allocation':alloc,'file':path.name}
            except Exception as e:
                summary.append({'method':'qwen','file':path.name,'error':repr(e)})
    fdir=outdir/'fidelity'; fdir.mkdir(parents=True,exist_ok=True)
    sdf=pd.DataFrame(summary); sdf.to_csv(fdir/'synthetic_transition_fidelity.csv',index=False)
    (pd.concat(conditional,ignore_index=True) if conditional else pd.DataFrame()).to_csv(fdir/'synthetic_transition_conditional.csv',index=False)
    pd.DataFrame(plaus).to_csv(fdir/'synthetic_transition_plausibility.csv',index=False)
    _json_dump(fdir/'synthetic_transition_summary.json',{
        'n_evaluated':len(summary),'reference_year_max':2018,'representative':representative_meta,'rows':summary
    })
    figdir=outdir/'figures'; figdir.mkdir(parents=True,exist_ok=True)
    if representative is not None:
        import matplotlib.pyplot as plt
        emb=representative['embedding'].copy()
        if representative_meta:
            emb.insert(0,'allocation',int(representative_meta['allocation']))
            emb.insert(0,'synthetic_fraction',float(representative_meta['synthetic_fraction']))
        emb.to_csv(fdir/'synthetic_transition_embedding.csv',index=False)

        label=(f"Qwen Q{int(round(float(representative_meta['synthetic_fraction'])*100))} A{int(representative_meta['allocation'])}"
               if representative_meta else 'Qwen synthetic')
        fig=plt.figure(figsize=(6.8,4.4)); ax=fig.add_subplot(111)
        ax.hist(representative['real_transitions'].Delta_Rutting_mm.to_numpy(float),bins=30,density=True,alpha=.55,label='Real 2016-2018')
        ax.hist(representative['synthetic_transitions'].Delta_Rutting_mm.to_numpy(float),bins=30,density=True,alpha=.55,label=label)
        ax.set_xlabel('Rutting increment (mm)'); ax.set_ylabel('Density'); ax.legend(); fig.tight_layout()
        fig.savefig(figdir/'synthetic_delta_rutting_density.png',dpi=180); plt.close(fig)

        fig=plt.figure(figsize=(6.8,4.8)); ax=fig.add_subplot(111)
        for source,g in representative['embedding'].groupby('source',sort=False):
            ax.scatter(g.pc1,g.pc2,s=18,alpha=.65,label=str(source).title())
        ax.set_xlabel('PCA component 1'); ax.set_ylabel('PCA component 2'); ax.legend(); fig.tight_layout()
        fig.savefig(figdir/'synthetic_transition_embedding.png',dpi=180); plt.close(fig)
    else:
        _write_placeholder_csv(fdir/'synthetic_transition_embedding.csv',['synthetic_fraction','allocation','source','pc1','pc2'])
    return sdf


def _write_benchmark_reports(root: Path, outdir: Path, master):
    bdir=outdir/'benchmarks'; bdir.mkdir(parents=True,exist_ok=True)
    pred_path=bdir/'benchmark_predictions.csv'; metrics_path=bdir/'benchmark_metrics.csv'
    if master is None or os.environ.get('V40_SKIP_BENCHMARK','0')=='1':
        _write_placeholder_csv(metrics_path,['model','method','requested_horizon','trajectory_rmse','trajectory_mae','trajectory_r2','persistence_rmse','persistence_skill','n_rows','n_origins'])
        return pd.DataFrame()
    try:
        if pred_path.exists(): bp=pd.read_csv(pred_path)
        else:
            from v40_benchmark import run_static_full_real_benchmark
            _,bp=run_static_full_real_benchmark(master,horizons=V40Protocol().horizons,seq_len=V40Protocol().seq_len,backend='auto')
            bp.to_csv(pred_path,index=False)
        bm=summarize_prediction_rows(bp)['horizon_metrics']; bm.to_csv(metrics_path,index=False); return bm
    except Exception as e:
        _write_placeholder_csv(metrics_path,['model','method','requested_horizon','trajectory_rmse','error'])
        _json_dump(bdir/'benchmark_error.json',{'error':repr(e)}); return pd.DataFrame()


def _plot_v40_figures(outdir: Path, hm: pd.DataFrame, eg: pd.DataFrame, mssr: pd.DataFrame):
    import matplotlib.pyplot as plt
    fdir=outdir/'figures'; fdir.mkdir(parents=True,exist_ok=True)
    if len(hm) and {'model','method','synthetic_fraction','requested_horizon','trajectory_rmse'}.issubset(hm.columns):
        for model in ['gru','tdr']:
            d=hm[hm.model.astype(str).eq(model)].copy()
            if d.empty: continue
            fig=plt.figure(figsize=(7.2,4.6)); ax=fig.add_subplot(111)
            for (method,q),g in d.groupby(['method','synthetic_fraction'],dropna=False,sort=True):
                if method not in {'full_real','qwen'}: continue
                z=g.groupby('requested_horizon',as_index=False).trajectory_rmse.mean().sort_values('requested_horizon')
                label='Full-real' if method=='full_real' else f'Qwen Q{int(round(float(q)*100))}'
                ax.plot(z.requested_horizon,z.trajectory_rmse,marker='o',label=label)
            ax.set_xlabel('Forecast horizon H'); ax.set_ylabel('Trajectory RMSE (mm)'); ax.set_xticks(list(V40Protocol().horizons)); ax.legend(); fig.tight_layout()
            fig.savefig(fdir/f'{model}_rmse_by_horizon.png',dpi=180); plt.close(fig)
        fig=plt.figure(figsize=(7.2,4.6)); ax=fig.add_subplot(111)
        z=hm.groupby('requested_horizon',as_index=False).persistence_skill.mean().sort_values('requested_horizon')
        ax.plot(z.requested_horizon,z.persistence_skill,marker='o'); ax.axhline(0,linewidth=1)
        ax.set_xlabel('Forecast horizon H'); ax.set_ylabel('Mean persistence skill'); ax.set_xticks(list(V40Protocol().horizons)); fig.tight_layout()
        fig.savefig(fdir/'persistence_skill_by_horizon.png',dpi=180); plt.close(fig)
    if len(eg) and {'step_h','error_growth'}.issubset(eg.columns):
        fig=plt.figure(figsize=(7.2,4.6)); ax=fig.add_subplot(111)
        z=eg.groupby('step_h',as_index=False).error_growth.mean().sort_values('step_h')
        ax.plot(z.step_h,z.error_growth,marker='o'); ax.axhline(1,linewidth=1)
        ax.set_xlabel('Open-loop step h'); ax.set_ylabel('RMSE growth vs step 1'); fig.tight_layout()
        fig.savefig(fdir/'error_growth_by_horizon.png',dpi=180); plt.close(fig)
    if len(mssr) and {'method','margin','requested_horizon','mssr'}.issubset(mssr.columns):
        d=mssr[(mssr.method.astype(str).eq('qwen')) & np.isclose(pd.to_numeric(mssr.margin),.10)]
        if len(d):
            fig=plt.figure(figsize=(7.2,4.6)); ax=fig.add_subplot(111)
            for model,g in d.groupby('model',sort=True):
                z=g.sort_values('requested_horizon'); ax.plot(z.requested_horizon,z.mssr,marker='o',label=str(model).upper())
            ax.set_xlabel('Forecast horizon H'); ax.set_ylabel('Contiguous MSSR (10% margin)'); ax.set_ylim(-.02,1.02); ax.set_xticks(list(V40Protocol().horizons)); ax.legend(); fig.tight_layout()
            fig.savefig(fdir/'qwen_mssr_by_horizon.png',dpi=180); plt.close(fig)


def write_v40_reports(root):
    root=Path(root); out=root/'reports/v40'; out.mkdir(parents=True,exist_ok=True)
    dirs,audits,hm,sm,eg,hd,preds=_aggregate_run_artifacts(root)
    rdir=out/'reproduction'; rdir.mkdir(parents=True,exist_ok=True)
    if len(audits): audits.to_csv(rdir/'reproduction_audit_summary.csv',index=False)
    else: _write_placeholder_csv(rdir/'reproduction_audit_summary.csv',['run_stem','model','method','synthetic_fraction','allocation','seed','formal','max_steps','status'])
    mdir=out/'metrics'; mdir.mkdir(parents=True,exist_ok=True)
    files=[('multi_horizon_run_metrics.csv',hm,['model','method','synthetic_fraction','allocation','seed','requested_horizon','trajectory_rmse','trajectory_mae','trajectory_r2','persistence_rmse','persistence_skill']),
           ('horizon_step_metrics.csv',sm,['model','method','synthetic_fraction','allocation','seed','requested_horizon','step_h','rmse','mae','r2','persistence_rmse','persistence_skill']),
           ('error_growth.csv',eg,['model','method','synthetic_fraction','allocation','seed','requested_horizon','step_h','rmse','error_growth']),
           ('horizon_degradation.csv',hd,['model','method','synthetic_fraction','allocation','seed','requested_horizon','trajectory_rmse','horizon_rmse_growth_vs_h1'])]
    for name,df,cols in files:
        if len(df): df.to_csv(mdir/name,index=False)
        else: _write_placeholder_csv(mdir/name,cols)
    if len(sm):
        pc=[c for c in ['model','method','synthetic_fraction','allocation','seed','requested_horizon','step_h','persistence_skill'] if c in sm.columns]
        sm[pc].to_csv(mdir/'persistence_skill.csv',index=False)
    else: _write_placeholder_csv(mdir/'persistence_skill.csv',['model','method','synthetic_fraction','allocation','seed','requested_horizon','step_h','persistence_skill'])
    noninf,mssr=_write_noninferiority_reports(preds,out)
    master_path=root/'data/processed/rutting_master_v31.csv'; master=pd.read_csv(master_path) if master_path.exists() else None
    fidelity=_write_fidelity_reports(root,out,master)
    benchmark=_write_benchmark_reports(root,out,master)
    adir=out/'audit'; adir.mkdir(parents=True,exist_ok=True)
    if master is not None:
        try:
            arrays=build_eval_arrays(master,V40Protocol().seq_len)
            lg=run_openloop_leakage_audit(arrays,'gru',horizon=3); lt=run_openloop_leakage_audit(arrays,'tdr',horizon=3)
            leakage={'gru':lg,'tdr':lt,'status':'PASS' if lg['status']=='PASS' and lt['status']=='PASS' else 'FAIL'}
        except Exception as e: leakage={'status':'FAIL','error':repr(e)}
    else: leakage={'status':'NOT_RUN','reason':'master data unavailable'}
    _json_dump(adir/'leakage_test_results.json',leakage)
    update_hashes={}
    upd=root/'data/v31_updates'
    if upd.exists():
        for pth in sorted(upd.glob('*.csv')): update_hashes[pth.name]=_sha256(pth)
    formal_runs=0; smoke_runs=0; completed_runs=0; failed_runs=0; incomplete_runs=0
    for d in dirs:
        try: run_manifest=json.loads((d/'run_manifest.json').read_text(encoding='utf-8'))
        except Exception: run_manifest={}
        is_formal=bool(run_manifest.get('formal')); formal_runs+=int(is_formal); smoke_runs+=int(not is_formal)
        try: audit_status=json.loads((d/'reproduction_audit.json').read_text(encoding='utf-8')).get('status')
        except Exception: audit_status=None
        required=[d/'checkpoint.pt',d/'metrics.json']+[d/f'pred_H{h:03d}.csv' for h in V40Protocol().horizons]
        accepted=(audit_status=='PASS') if is_formal else (audit_status in {'PASS','SMOKE_ONLY'})
        complete=all(x.exists() for x in required) and accepted
        if complete: completed_runs+=1
        elif audit_status=='FAIL': failed_runs+=1
        else: incomplete_runs+=1
    protocol=V40Protocol()
    manifest={
        'stage':'V4.0 Stage 10','v40_protocol_signature':protocol_signature(),
        'v31_protocol_signature':__import__('v31_protocol').protocol_signature(),
        'primary_horizon':protocol.primary_horizon,'horizons':list(protocol.horizons),
        'max_optimizer_steps':protocol.max_optimizer_steps,'warmup_steps':protocol.warmup_steps,
        'validation_every_steps':protocol.validation_every_steps,'patience_checks':protocol.patience_checks,
        'model_configs':{
            'gru':{'input_dim':23,'hidden_dim':32,'num_layers':2,'dropout':0.15},
            'tdr':{'selected':'TDR-B','input_dim':23,'d_model':48,'dropout':0.15,'weight_decay':0.0005,'lr':0.0008,'selection_uses_2020_only':True},
        },
        'run_directories_seen':len(dirs),'completed_run_directories':completed_runs,
        'failed_run_directories':failed_runs,'incomplete_run_directories':incomplete_runs,
        'formal_run_directories':formal_runs,'smoke_run_directories':smoke_runs,
        'reproduction_pass_count':int((audits.status.astype(str).eq('PASS')).sum()) if len(audits) and 'status' in audits else 0,
        'leakage_audit_status':leakage.get('status'),'master_sha256':_sha256(master_path) if master_path.exists() else None,
        'update_cache_sha256':update_hashes,'fidelity_rows':int(len(fidelity)),'benchmark_rows':int(len(benchmark)),
        'validation_design':'H6 primary; H1/H3/H12 secondary; horizon degradation + synthetic-transition fidelity; chronological isolation before augmentation',
    }
    _json_dump(adir/'stage10_manifest.json',manifest)
    _plot_v40_figures(out,hm,eg,mssr)
    return out
