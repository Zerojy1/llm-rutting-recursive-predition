from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

from v31_protocol import V31Protocol,write_protocol_json,validate_protocol
from v31_master_data import read_raw_xlsx,build_master_panel_from_frames
from v31_qwen_data import build_chronological_qwen_datasets
from v31_windows import build_generator_windows
from v31_quality import empirical_increment_bounds,prospective_rutting_bounds,memorization_audit
from v31_generation_eval import evaluate_generator_on_windows,save_generator_evaluation,condition_from_window
from v31_reporting import plot_substitution_curve,plot_capacity_map,plot_monitoring_frontier


def root_from_here(): return Path(__file__).resolve().parents[1]
def _json(path,obj): Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def _read_master(root): return pd.read_csv(Path(root)/'data/processed/rutting_master_v31.csv')

def stage01_master(root=None):
    root=Path(root or root_from_here()); p=V31Protocol(); validate_protocol(p); write_protocol_json(root/'config/v31_frozen_protocol.json',p)
    raw_path=root/'data/raw/足尺环道车辙数据.xlsx'; raw=read_raw_xlsx(raw_path,header=1); master=build_master_panel_from_frames(raw)
    out=root/'data/processed'; out.mkdir(parents=True,exist_ok=True); master.to_csv(out/'rutting_master_v31.csv',index=False,encoding='utf-8-sig')
    audit={'rows':len(master),'observed_rutting':int(master.Rutting_Observed.sum()),'missing_rutting':int((1-master.Rutting_Observed).sum()),
           'year_min':int(master.Year.min()),'year_max':int(master.Year.max()),'structures':int(master.STR_name.nunique()),
           'rutting_mm_min':float(master.Rutting_mm.min()),'rutting_mm_max':float(master.Rutting_mm.max()),'protocol':'V3.1'}
    _json(root/'reports/v31/01_master_data_audit.json',audit); print(json.dumps(audit,ensure_ascii=False,indent=2)); return master

def stage02_qwen_data(root=None):
    root=Path(root or root_from_here()); master=_read_master(root); frames=build_chronological_qwen_datasets(master,root/'data/v31_qwen',4,6)
    summary={k:{'n':len(v),'target_min':None if len(v)==0 else int(v.target_year_min.min()),'target_max':None if len(v)==0 else int(v.target_year_max.max())} for k,v in frames.items()}
    _json(root/'reports/v31/02_qwen_data_summary.json',summary); print(json.dumps(summary,ensure_ascii=False,indent=2)); return frames

def stage03_qwen_train(root=None):
    from v31_qwen_train import run_v31_qwen_training
    return run_v31_qwen_training(Path(root or root_from_here()))

def stage04_train_generator_baselines(root=None):
    from v31_baseline_models import TimeGANBaseline,ConditionalDDPMBaseline,save_baseline
    root=Path(root or root_from_here()); w=pd.read_csv(root/'data/v31_qwen/refit_windows.csv')
    te=int(os.getenv('V31_TIMEGAN_EMBED_STEPS','1000')); ts=int(os.getenv('V31_TIMEGAN_SUP_STEPS','1000')); tj=int(os.getenv('V31_TIMEGAN_JOINT_STEPS','2000'))
    dd=int(os.getenv('V31_DDPM_TRAIN_STEPS','3000'))
    tg,hg=TimeGANBaseline.fit(w,6,steps_embed=te,steps_supervisor=ts,steps_joint=tj,batch_size=64,hidden_dim=24,device='auto',seed=42)
    save_baseline(tg,root/'models/v31_generators/timegan.pt',hg)
    tw,hw=ConditionalDDPMBaseline.fit(w,6,steps=dd,batch_size=64,hidden=64,diffusion_steps=100,device='auto',seed=42)
    save_baseline(tw,root/'models/v31_generators/timeweaver_inspired.pt',hw)
    _json(root/'reports/v31/04_baseline_training.json',{'timegan_steps':[te,ts,tj],'timeweaver_inspired_steps':dd,'timeweaver_exact_official_reproduction':False})
    return tg,tw

def _increment_bounds_from_refit(root):
    w=pd.read_csv(Path(root)/'data/v31_qwen/refit_windows.csv'); inc=[]
    for i in range(1,7): inc.extend(pd.to_numeric(w[f'target_increment_f{i}'],errors='coerce').dropna().tolist())
    master=_read_master(root); source=master[(master.Year<=2018)&master.Rutting_Observed.eq(1)]; ib=empirical_increment_bounds(inc,.005,.995,.05); rb=prospective_rutting_bounds(source.Rutting_mm.to_numpy(float),ib,horizon=V31Protocol().future_steps,pad=.25)
    return ib,rb,np.asarray([[r[f'target_increment_f{i}'] for i in range(1,7)] for _,r in w.iterrows()],float)

def stage04_generator_quality(root=None):
    from v31_baseline_models import load_timegan_baseline,load_ddpm_baseline
    from v31_qwen_generate import QwenTrajectoryGenerator
    from v31_qwen_qc import QualityControlledQwen
    root=Path(root or root_from_here()); hold=pd.read_csv(root/'data/v31_qwen/holdout_windows.csv'); ib,rb,train_real=_increment_bounds_from_refit(root); out=root/'reports/v31/generator_quality'; maxw=int(os.getenv('V31_QUALITY_MAX_WINDOWS','0')) or None
    generators={}
    tg_path=root/'models/v31_generators/timegan.pt'; tw_path=root/'models/v31_generators/timeweaver_inspired.pt'
    if tg_path.exists(): generators['TimeGAN_retrieval']=load_timegan_baseline(tg_path).generator_fn
    if tw_path.exists(): generators['TimeWeaver_inspired']=load_ddpm_baseline(tw_path).generator_fn
    model_dir=Path(os.environ.get('QWEN_BASE_MODEL_DIR',r'K:\ollama\Qwen3.5-9B-hf')); adapter=root/'models/v31_qwen/refit/refit_adapter'
    if model_dir.exists():
        try: generators['Base_Qwen']=QwenTrajectoryGenerator(model_dir).__call__
        except Exception as e: print('[WARN] Base Qwen unavailable:',e)
        if adapter.exists():
            try:
                ft=QwenTrajectoryGenerator(model_dir,adapter); generators['FT_Qwen']=ft.__call__; generators['FT_Qwen_QC']=QualityControlledQwen(ft,ib,rb,k=5,local_center=np.mean(train_real,axis=0),local_scale=np.std(train_real,axis=0)+1e-6).__call__
            except Exception as e: print('[WARN] FT Qwen unavailable:',e)
    summaries={}
    from v31_controllability import conditional_controllability_audit
    cont_n=int(os.getenv('V31_CONTROLLABILITY_WINDOWS','30')); cont_rep=int(os.getenv('V31_CONTROLLABILITY_REPLICATES','2'))
    cont_rows=hold.iloc[:min(cont_n,len(hold))] if cont_n>0 else hold.iloc[0:0]
    conditions=[condition_from_window(r,6,4) for _,r in cont_rows.iterrows()]
    alt_structures=[]
    seen=set()
    for c in conditions:
        key=tuple(sorted(c['structure_cm'].items()))
        if key not in seen: seen.add(key); alt_structures.append(c['structure_cm'])
    master=_read_master(root); ref=master[master.Year.le(2018)]
    limits={'delta_load_10k':(float(ref.Delta_Load_10k.min()),float(ref.Delta_Load_10k.max())),
            'cum_load_10k':(float(ref.Cum_Load_10k.min()),float(ref.Cum_Load_10k.max())),
            'temperature_c':(float(ref.Avg_Temp_C.min()),float(ref.Avg_Temp_C.max())),
            'rutting_mm':rb}
    for name,gen in generators.items():
        print('[generator audit]',name); rows,s,R,S=evaluate_generator_on_windows(hold,gen,6,ib,rb,maxw); save_generator_evaluation(out,name,rows,s,R,S,training_real=train_real)
        if conditions:
            cr,cs=conditional_controllability_audit(conditions,gen,variant_kinds=('load','temperature','state','structure'),replicates=cont_rep,limits=limits,alternate_structures=alt_structures)
            cr.to_csv(out/f'{name}_conditional_controllability.csv',index=False,encoding='utf-8-sig'); _json(out/f'{name}_conditional_controllability.json',cs); s['conditional_controllability']=cs
        summaries[name]=s
    _json(out/'all_generator_summaries.json',summaries); return summaries

def tdr_stability_masks(meta):
    years=pd.to_numeric(meta['Year'],errors='coerce').astype('Int64')
    # The 2019 pool is the object of the later real/synthetic substitution experiment,
    # so it is not used as supervised training data when selecting TDR capacity/regularization.
    train=years.le(2018).fillna(False).to_numpy(bool)
    val=years.eq(2020).fillna(False).to_numpy(bool)
    return train,val


def stage05_tdr_stability(root=None):
    from v31_downstream_data import build_real_downstream_frame,build_sequence_arrays_simple
    from v31_tdr import STABILITY_CANDIDATES,fit_tdr_step_budget
    from v31_training import choose_stable_candidate
    root=Path(root or root_from_here()); m=_read_master(root); df=build_real_downstream_frame(m); X,y,p,pk,meta=build_sequence_arrays_simple(df,6); train,val=tdr_stability_masks(meta); rows=[]; histories=root/'reports/v31/tdr_stability/histories'; histories.mkdir(parents=True,exist_ok=True)
    max_steps=int(os.getenv('V31_MAX_STEPS','2400'))
    for name,cfg in STABILITY_CANDIDATES.items():
        vals=[]; gaps=[]
        for seed in (42,43,44):
            print('[TDR stability]',name,'seed',seed); r=fit_tdr_step_budget(X[train],y[train],p[train],pk[train],X[val],y[val],p[val],pk[val],cfg,max_steps=max_steps,warmup_steps=min(100,max_steps//5),val_every=max(1,min(50,max_steps//4)),patience_checks=6,seed=seed)
            h=pd.DataFrame(r['history']); h.to_csv(histories/f'{name}_seed{seed}.csv',index=False); best=h.loc[h.val_rmse.idxmin()]; vals.append(float(r['best_val_rmse'])); gaps.append(float(best.val_rmse-best.train_rmse))
        arr=np.asarray(vals); rows.append({'name':name,'mean_val_rmse':float(arr.mean()),'se_val_rmse':float(arr.std(ddof=1)/np.sqrt(len(arr))),'seed_sd':float(arr.std(ddof=1)),'gap':float(np.mean(gaps)),'params':int(r['parameter_count'])})
    table=pd.DataFrame(rows); selected=choose_stable_candidate(table); out=root/'reports/v31/tdr_stability'; table.to_csv(out/'candidate_summary.csv',index=False,encoding='utf-8-sig'); _json(out/'selected_tdr.json',{'selected':selected,'config':STABILITY_CANDIDATES[selected].__dict__,'selection_uses_2020_only':True}); print('selected:',selected); return selected

def _load_main_generators(root):
    from v31_baseline_models import load_timegan_baseline,load_ddpm_baseline
    from v31_qwen_generate import QwenTrajectoryGenerator
    from v31_qwen_qc import QualityControlledQwen
    root=Path(root); gens={}; tp=root/'models/v31_generators/timegan.pt'; wp=root/'models/v31_generators/timeweaver_inspired.pt'
    if tp.exists(): gens['timegan']=load_timegan_baseline(tp).generator_fn
    if wp.exists(): gens['timeweaver']=load_ddpm_baseline(wp).generator_fn
    model_dir=Path(os.environ.get('QWEN_BASE_MODEL_DIR',r'K:\ollama\Qwen3.5-9B-hf')); ad=root/'models/v31_qwen/refit/refit_adapter'
    if model_dir.exists() and ad.exists():
        ib,rb,train_real=_increment_bounds_from_refit(root); q=QwenTrajectoryGenerator(model_dir,ad); gens['qwen']=QualityControlledQwen(q,ib,rb,k=5,local_center=np.mean(train_real,axis=0),local_scale=np.std(train_real,axis=0)+1e-6).__call__
    return gens

def _real_update(master):
    return master[(master.Year==2019)&master.Rutting_Observed.eq(1)].copy().assign(source_type='real')


def substitution_run_plan(methods, models, protocol=None):
    """Return the frozen Stage-06 execution plan without duplicating shared controls.

    Historical-only is independent of allocation, full-real is trained once per model/seed
    and later aliased across allocations for paired reporting, while real-bootstrap is
    shared across generator methods because it contains no generated samples.
    """
    p=protocol or V31Protocol(); plan=[]
    methods=[str(x).strip() for x in methods if str(x).strip()]
    models=[str(x).strip() for x in models if str(x).strip()]
    # Historical-only control: once per model and neural seed.
    for model in models:
        for seed in p.neural_seeds:
            plan.append({'method':'historical_only','q':0.0,'allocation':0,'allocation_seed':None,'seed':seed,'model':model})
    # Full-real reference: one training run per model/seed; aliases are written for A1..A5.
    for model in models:
        for seed in p.neural_seeds:
            plan.append({'method':'full_real','q':0.0,'allocation':0,'allocation_seed':None,'seed':seed,'model':model})
    # Real-bootstrap is a shared no-new-information control and must not be multiplied by generator.
    bootstrap_q=[q for q in p.replacement_fractions if 0 < q < 1]
    for ai,aseed in enumerate(p.allocation_seeds,1):
        for q in bootstrap_q:
            for model in models:
                for seed in p.neural_seeds:
                    plan.append({'method':'real_bootstrap','q':float(q),'allocation':ai,'allocation_seed':aseed,'seed':seed,'model':model})
        for method in methods:
            for q in [q for q in p.replacement_fractions if q>0]:
                for model in models:
                    for seed in p.neural_seeds:
                        plan.append({'method':method,'q':float(q),'allocation':ai,'allocation_seed':aseed,'seed':seed,'model':model})
    return plan

def _run_stem(method,q,alloc,seed,model):
    return f'{model}_{method}_Q{int(q*100):03d}_A{alloc}_S{seed}'


def _load_completed_run(out,method,q,alloc,seed,model):
    """Return saved Stage-06 metrics only when the full run bundle is present.

    This makes Stage 06 restart-safe after a later generator/update failure without
    re-training already completed historical/full-real/bootstrap/mixed conditions.
    """
    out=Path(out); stem=_run_stem(method,q,alloc,seed,model)
    required=[out/f'{stem}_metrics.json', out/f'{stem}_history.csv',
              out/f'{stem}_pred2021.csv', out/f'{stem}_sparse.csv']
    if not all(x.exists() for x in required):
        return None
    try:
        return json.loads(required[0].read_text(encoding='utf-8'))
    except Exception:
        return None


def _save_run(out,method,q,alloc,seed,model,res):
    out.mkdir(parents=True,exist_ok=True); fit=res['fit']; h=pd.DataFrame(fit['history']); stem=_run_stem(method,q,alloc,seed,model)
    h.to_csv(out/f'{stem}_history.csv',index=False); metrics={'method':method,'synthetic_fraction':q,'allocation':alloc,'seed':seed,'model':model,'n_train':res['n_train'],'dev':res['dev_metrics'],'eval2021':res['eval_metrics'],'best_val_rmse':fit['best_val_rmse'],'steps_completed':fit['steps_completed'],'parameter_count':fit.get('parameter_count')}; _json(out/f'{stem}_metrics.json',metrics)
    if 'eval_predictions' in res: res['eval_predictions'].to_csv(out/f'{stem}_pred2021.csv',index=False)
    if 'sparse' in res: res['sparse'].to_csv(out/f'{stem}_sparse.csv',index=False)
    return metrics

def stage06_equal_budget(root=None):
    from v31_experiment_engine import make_masked_update,run_gru_condition,run_tdr_condition,run_gru_real_bootstrap,run_tdr_real_bootstrap,select_retained_indices
    from v31_gru import GRUConfig
    from v31_tdr import TDRStableConfig
    from v31_sparse_eval import sparse_eval_fit
    root=Path(root or root_from_here()); master=_read_master(root); p=V31Protocol(); gens=_load_main_generators(root)
    methods=[x.strip() for x in os.getenv('V31_GENERATORS','qwen,timegan,timeweaver').split(',') if x.strip() in gens]
    if not methods: raise RuntimeError('No trained generator available. Run stages 03/04 first.')
    models=[x.strip() for x in os.getenv('V31_DOWNSTREAMS','gru,tdr').split(',') if x.strip()]
    unknown=[x for x in models if x not in {'gru','tdr'}]
    if unknown: raise ValueError('Unsupported V31_DOWNSTREAMS: '+','.join(unknown))
    sel=json.loads((root/'reports/v31/tdr_stability/selected_tdr.json').read_text(encoding='utf-8')) if (root/'reports/v31/tdr_stability/selected_tdr.json').exists() else {'selected':'TDR-C','config':{'d_model':32,'dropout':.2,'weight_decay':5e-4,'lr':8e-4}}
    tcfg=TDRStableConfig(**sel['config']); gcfg=GRUConfig(input_dim=23,hidden_dim=32,num_layers=2,dropout=.15)
    out=root/'reports/v31/substitution_runs'; cache=root/'data/v31_updates'; cache.mkdir(parents=True,exist_ok=True); results=[]; max_steps=int(os.getenv('V31_MAX_STEPS','2400'))

    def run_condition(model,update,seed):
        if model=='gru': return run_gru_condition(master,update,gcfg,seed,max_steps,p.warmup_steps,p.validation_every_steps,p.patience_checks)
        return run_tdr_condition(master,update,tcfg,seed,max_steps,p.warmup_steps,p.validation_every_steps,p.patience_checks)

    def run_bootstrap(model,retained,seed,aseed):
        if model=='gru': return run_gru_real_bootstrap(master,retained,gcfg,seed,aseed,max_steps,p.warmup_steps,p.validation_every_steps,p.patience_checks)
        return run_tdr_real_bootstrap(master,retained,tcfg,seed,aseed,max_steps,p.warmup_steps,p.validation_every_steps,p.patience_checks)

    # Mandatory no-update control: trained once per model/seed. 2019 exposure remains in the sequence,
    # while all 2019 rutting targets are hidden by training_frame_with_update().
    empty_update=_real_update(master).iloc[0:0].copy()
    for model in models:
        for seed in p.neural_seeds:
            done=_load_completed_run(out,'historical_only',0.,0,seed,model)
            if done is not None:
                print('[Stage06 resume] historical-only',model,'seed',seed); results.append(done); continue
            print('[Stage06 historical-only]',model,'seed',seed)
            res=run_condition(model,empty_update,seed); res['sparse']=sparse_eval_fit(master,res['fit'],model)
            results.append(_save_run(out,'historical_only',0.,0,seed,model,res))

    # Full-real Q0 is independent of replacement allocation. Train once per model/seed and write
    # deterministic A1..A5 aliases so downstream paired analysis has matching filenames without
    # spending five times the optimizer budget on an identical condition.
    full=_real_update(master)
    for model in models:
        for seed in p.neural_seeds:
            aliases=[_load_completed_run(out,'full_real',0.,ai,seed,model) for ai,_ in enumerate(p.allocation_seeds,1)]
            if all(x is not None for x in aliases):
                print('[Stage06 resume] full-real',model,'seed',seed); results.extend(aliases); continue
            print('[Stage06 full-real]',model,'seed',seed)
            res=run_condition(model,full,seed); res['sparse']=sparse_eval_fit(master,res['fit'],model)
            for ai,_ in enumerate(p.allocation_seeds,1):
                results.append(_save_run(out,'full_real',0.,ai,seed,model,res))

    # Shared no-new-information bootstrap control. It depends on the retained-real allocation and q,
    # but NOT on which generator method will later fill the withheld states.
    for ai,aseed in enumerate(p.allocation_seeds,1):
        for q in [x for x in p.replacement_fractions if 0 < x < 1]:
            retained=select_retained_indices(master,q,aseed)
            for model in models:
                for seed in p.neural_seeds:
                    done=_load_completed_run(out,'real_bootstrap',q,ai,seed,model)
                    if done is not None:
                        print('[Stage06 resume] real-bootstrap',f'Q{int(q*100)}',f'A{ai}',model,'seed',seed); results.append(done); continue
                    print('[Stage06 real-bootstrap]',f'Q{int(q*100)}',f'A{ai}',model,'seed',seed)
                    br=run_bootstrap(model,retained,seed,aseed); br['sparse']=sparse_eval_fit(master,br['fit'],model)
                    results.append(_save_run(out,'real_bootstrap',q,ai,seed,model,br))

    # Generator-specific equal-budget replacement conditions.
    for ai,aseed in enumerate(p.allocation_seeds,1):
        for method in methods:
            for q in [x for x in p.replacement_fractions if x>0]:
                completed={(model,seed):_load_completed_run(out,method,q,ai,seed,model)
                           for model in models for seed in p.neural_seeds}
                if all(v is not None for v in completed.values()):
                    print('[Stage06 resume] mixed',method,f'Q{int(q*100)}',f'A{ai}','all downstream runs complete')
                    results.extend(completed.values()); continue
                upath=cache/f'{method}_Q{int(q*100):03d}_A{ai}.csv'
                if upath.exists(): update=pd.read_csv(upath)
                else:
                    print('[Stage06 generate update]',method,f'Q{int(q*100)}',f'A{ai}')
                    update,retained,states=make_masked_update(master,q,aseed,gens[method])
                    update.to_csv(upath,index=False,encoding='utf-8-sig')
                    states.to_csv(cache/f'{method}_Q{int(q*100):03d}_A{ai}_states.csv',index=False,encoding='utf-8-sig')
                for model in models:
                    for seed in p.neural_seeds:
                        done=completed[(model,seed)]
                        if done is not None:
                            print('[Stage06 resume] mixed',method,f'Q{int(q*100)}',f'A{ai}',model,'seed',seed); results.append(done); continue
                        print('[Stage06 mixed]',method,f'Q{int(q*100)}',f'A{ai}',model,'seed',seed)
                        res=run_condition(model,update,seed); res['sparse']=sparse_eval_fit(master,res['fit'],model)
                        results.append(_save_run(out,method,q,ai,seed,model,res))

    table=pd.json_normalize(results); table.to_csv(out/'all_metrics.csv',index=False,encoding='utf-8-sig')
    _json(out/'run_plan.json',substitution_run_plan(methods,models,p))
    return results

def stage07_capacity_and_state(root=None):
    from v31_gru import CAPACITY_LADDER,GRUConfig
    from v31_experiment_engine import run_gru_condition
    root=Path(root or root_from_here()); master=_read_master(root); metrics=[]; max_steps=int(os.getenv('V31_MAX_STEPS','2400')); updates=root/'data/v31_updates'
    # capacity boundary uses frozen Qwen updates already generated in stage 06; no generator retuning.
    for level,c in CAPACITY_LADDER.items():
        cfg=GRUConfig(23,c['hidden_dim'],c['num_layers'],.15)
        for q in (0.,.25,.5,.75):
            for ai,aseed in enumerate(V31Protocol().allocation_seeds,1):
                up=_real_update(master) if q==0 else pd.read_csv(updates/f'qwen_Q{int(q*100):03d}_A{ai}.csv')
                for seed in V31Protocol().neural_seeds:
                    r=run_gru_condition(master,up,cfg,seed,max_steps,100,50,6); metrics.append({'capacity':level,'params':r['fit']['parameter_count'],'synthetic_fraction':q,'allocation':ai,'seed':seed,'rmse2021':r['eval_metrics']['rmse'],'oi_best':min(r['fit']['history'],key=lambda x:x['val_rmse'])['oi']})
    out=root/'reports/v31/capacity'; out.mkdir(parents=True,exist_ok=True); table=pd.DataFrame(metrics); table.to_csv(out/'capacity_results.csv',index=False,encoding='utf-8-sig'); base=table[table.synthetic_fraction.eq(0)].groupby(['capacity','allocation','seed']).rmse2021.first()
    rel=[]
    for _,r in table.iterrows():
        b=base.get((r.capacity,r.allocation,r.seed),np.nan); rel.append((r.rmse2021-b)/b if np.isfinite(b) else np.nan)
    table['relative_degradation']=rel; table.to_csv(out/'capacity_results_with_relative.csv',index=False,encoding='utf-8-sig'); plot_capacity_map(table,out/'capacity_synthetic_map.png')
    # V3.2 mechanism reanalysis separates chronological progression from context support/OOD.
    from v32_reanalysis import recompute_state_mechanism
    recompute_state_mechanism(root)
    return table

def stage08_sparse_monitoring(root=None):
    root=Path(root or root_from_here()); master=_read_master(root); rows=[]
    run_dir=root/'reports/v31/substitution_runs'
    for f in run_dir.glob('*_sparse.csv'):
        g=pd.read_csv(f); stem=f.stem.replace('_sparse',''); parts=stem.split('_')
        # preserve exact run name; detailed method/model metadata remain available in metrics files
        for _,r in g.iterrows(): rows.append({'condition':stem,'gap':r['gap'],'observation_frequency':r['observation_frequency'],'rmse':r['rmse'],'n_predictions':r.get('n_predictions',np.nan)})
    # persistence benchmark using the same first-2021-observation anchor convention
    for gap in (1,2,4,6,12,None):
        freq=0.0 if gap is None else 1.0/gap; errs=[]
        for sid,g in master[(master.Year==2021)&master.Rutting_Observed.eq(1)].groupby('STR_name'):
            y=g.sort_values('Observation_Index').Rutting_mm.to_numpy(float)
            if len(y)<2: continue
            state=y[0]
            for j in range(1,len(y)):
                pred=state; errs.append((pred-y[j])**2)
                if gap is not None and j%gap==0: state=y[j]
        rows.append({'condition':'Persistence','gap':'open' if gap is None else gap,'observation_frequency':freq,'rmse':float(np.sqrt(np.mean(errs))),'n_predictions':len(errs)})
    out=root/'reports/v31/sparse_monitoring'; out.mkdir(parents=True,exist_ok=True); tab=pd.DataFrame(rows); tab.to_csv(out/'monitoring_frontier_all_runs.csv',index=False,encoding='utf-8-sig')
    if len(tab):
        # publication plot uses persistence plus mean curves for full-real and Qwen if available
        pp=[tab[tab.condition.eq('Persistence')]]
        for token,label in [('full_real','Full-real'),('_qwen_','Qwen-augmented')]:
            q=tab[tab.condition.str.contains(token,case=False,regex=False)]
            if len(q):
                agg=q.groupby(['gap','observation_frequency'],as_index=False).rmse.mean(); agg['condition']=label; pp.append(agg)
        plot_monitoring_frontier(pd.concat(pp,ignore_index=True),out/'monitoring_efficiency_frontier.png')
    return tab

def stage09_report(root=None):
    from v31_statistics import maximum_synthetic_substitution_ratio,hierarchical_cluster_relative_bootstrap
    root=Path(root or root_from_here()); src=root/'reports/v31/substitution_runs/all_metrics.csv'
    if not src.exists(): raise FileNotFoundError('Run stage 06 first')
    df=pd.read_csv(src); run_dir=root/'reports/v31/substitution_runs'; rows=[]
    # Cluster-aware primary table: pair each mixed run with its full-real run using same model/allocation/seed.
    for (model,method,q),g in df[(~df.method.eq('full_real'))&(~df.method.eq('real_bootstrap'))&(~df.method.eq('historical_only'))].groupby(['model','method','synthetic_fraction']):
        paired=[]
        for _,r in g.iterrows():
            ai=int(r.allocation); seed=int(r.seed); qtag=int(round(float(q)*100))
            realf=run_dir/f'{model}_full_real_Q000_A{ai}_S{seed}_pred2021.csv'; mixf=run_dir/f'{model}_{method}_Q{qtag:03d}_A{ai}_S{seed}_pred2021.csv'
            if not realf.exists() or not mixf.exists(): continue
            a=pd.read_csv(realf).rename(columns={'y_pred':'real_pred'}); b=pd.read_csv(mixf).rename(columns={'y_pred':'mix_pred'})
            j=a[['STR_name','Observation_Index','y_true','real_pred']].merge(b[['STR_name','Observation_Index','mix_pred']],on=['STR_name','Observation_Index'],how='inner'); paired.append(j)
        if paired:
            bs=hierarchical_cluster_relative_bootstrap(paired,n_boot=10000,seed=20260814)
            rows.append({'model':model,'method':method,'synthetic_fraction':float(q),'relative_degradation':bs['mean_relative_degradation'],'ci_low':bs['ci_low'],'upper_ci_relative_degradation':bs['ci_high'],'n_paired_runs':bs['n_runs'],'n_structures':bs['n_clusters']})
    table=pd.DataFrame(rows); out=root/'reports/v31/final'; out.mkdir(parents=True,exist_ok=True); table.to_csv(out/'substitution_cluster_aware_summary.csv',index=False,encoding='utf-8-sig')
    mssr={}
    for (model,method),g in table.groupby(['model','method']): mssr[f'{model}|{method}']=maximum_synthetic_substitution_ratio(g,.10)
    _json(out/'mssr_10pct_cluster_aware.json',mssr)
    from v32_reanalysis import recompute_contiguous_mssr
    recompute_contiguous_mssr(root)
    if len(table):
        qg=table[(table.model=='gru')&(table.method=='qwen')]
        if len(qg): plot_substitution_curve(qg,out/'qwen_gru_substitution_curve.png')
    return table
