from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from v31_state_features import context_support_frame, progression_frame
from v31_state_space import fit_state_support, in_domain_mask, support_distance, relative_support_quartiles, progression_extrapolation_flags
from v31_statistics import maximum_synthetic_substitution_ratio


def _json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def _master(root):
    return pd.read_csv(Path(root)/'data/processed/rutting_master_v31.csv')


def build_context_support_labels(master:pd.DataFrame):
    """V3.2 support definition: context support is separated from chronological progression."""
    train=master[(pd.to_numeric(master.Year,errors='coerce')<=2018)&master.Rutting_Observed.eq(1)].copy()
    evalm=master[(pd.to_numeric(master.Year,errors='coerce')==2021)&master.Rutting_Observed.eq(1)].copy()
    Xtr,_=context_support_frame(train); Xev,meta=context_support_frame(evalm)
    support=fit_state_support(Xtr,k=5,sparse_quantile=.75,domain_padding=.10)
    in_dom=in_domain_mask(support,Xev); dist=support_distance(support,Xev)
    meta=meta.copy(); meta['context_in_domain']=in_dom; meta['context_support_distance']=dist
    meta['context_support_quartile']=relative_support_quartiles(dist,in_dom)

    Ptr,_=progression_frame(train); Pev,pmeta=progression_frame(evalm)
    flags=progression_extrapolation_flags(Ptr,Pev)
    pmeta=pmeta[['STR_name','Observation_Index']].copy()
    pmeta['load_stage_extrapolation']=flags['load_stage_extrapolation']
    pmeta['rutting_state_extrapolation']=flags['rutting_state_extrapolation']
    meta=meta.merge(pmeta,on=['STR_name','Observation_Index'],how='left')
    return support,meta,train,evalm


def recompute_state_mechanism(root):
    root=Path(root); master=_master(root); out=root/'reports/v31/capacity'; out.mkdir(parents=True,exist_ok=True)
    support,labels,train,evalm=build_context_support_labels(master)
    labels.to_csv(out/'2021_state_support_labels.csv',index=False,encoding='utf-8-sig')

    # Prediction error by relative context-support quartile. No outcome label is used to define the quartiles.
    density=[]; run_dir=root/'reports/v31/substitution_runs'
    for f in run_dir.glob('*_pred2021.csv'):
        pr=pd.read_csv(f); j=pr.merge(labels,on=['STR_name','Observation_Index'],how='inner')
        if not {'y_pred','y_true'}.issubset(j.columns): continue
        j['sqerr']=(pd.to_numeric(j.y_pred,errors='coerce')-pd.to_numeric(j.y_true,errors='coerce'))**2
        for label,g in j.groupby('context_support_quartile'):
            density.append({'run':f.stem.replace('_pred2021',''),'context_support_quartile':label,'n':len(g),
                            'rmse':float(np.sqrt(g.sqerr.mean())),
                            'context_ood_fraction':float(np.mean(~g.context_in_domain.astype(bool))),
                            'load_stage_extrapolation_fraction':float(np.nanmean(g.load_stage_extrapolation.astype(float))),
                            'rutting_state_extrapolation_fraction':float(np.nanmean(g.rutting_state_extrapolation.astype(float)))})
    pd.DataFrame(density).to_csv(out/'density_stratified_prediction_error.csv',index=False,encoding='utf-8-sig')

    # Direct mechanism result: Qwen-vs-full-real error difference within each support quartile.
    gains=[]; metrics_path=run_dir/'all_metrics.csv'
    if metrics_path.exists():
        metrics=pd.read_csv(metrics_path); qrows=metrics[metrics.method.astype(str).eq('qwen')]
        for _,r in qrows.iterrows():
            model=str(r.model); q=float(r.synthetic_fraction); ai=int(r.allocation); seed=int(r.seed); qtag=int(round(q*100))
            realf=run_dir/f'{model}_full_real_Q000_A{ai}_S{seed}_pred2021.csv'
            qf=run_dir/f'{model}_qwen_Q{qtag:03d}_A{ai}_S{seed}_pred2021.csv'
            if not realf.exists() or not qf.exists(): continue
            a=pd.read_csv(realf).rename(columns={'y_pred':'real_pred'}); b=pd.read_csv(qf).rename(columns={'y_pred':'qwen_pred'})
            j=a[['STR_name','Observation_Index','y_true','real_pred']].merge(b[['STR_name','Observation_Index','qwen_pred']],on=['STR_name','Observation_Index'],how='inner')
            j=j.merge(labels[['STR_name','Observation_Index','context_support_quartile','context_support_distance','context_in_domain']],on=['STR_name','Observation_Index'],how='inner')
            for label,g in j.groupby('context_support_quartile'):
                y=pd.to_numeric(g.y_true,errors='coerce').to_numpy(float); rp=pd.to_numeric(g.real_pred,errors='coerce').to_numpy(float); qp=pd.to_numeric(g.qwen_pred,errors='coerce').to_numpy(float)
                rr=float(np.sqrt(np.mean((rp-y)**2))); qr=float(np.sqrt(np.mean((qp-y)**2)))
                gains.append({'model':model,'synthetic_fraction':q,'allocation':ai,'seed':seed,'context_support_quartile':label,'n':len(g),
                              'full_real_rmse':rr,'qwen_rmse':qr,'qwen_minus_real_rmse':qr-rr,
                              'relative_qwen_gain':(rr-qr)/rr if rr>0 else np.nan,
                              'mean_context_support_distance':float(g.context_support_distance.mean())})
    pd.DataFrame(gains).to_csv(out/'support_dependent_qwen_gain.csv',index=False,encoding='utf-8-sig')

    # Describe where generated 2019 transitions lie in context support. This is not called input-space expansion:
    # Qwen generates responses for supplied load/temperature/structure conditions rather than inventing new covariates.
    real2019=master[(pd.to_numeric(master.Year,errors='coerce')==2019)&master.Rutting_Observed.eq(1)].copy()
    X19,_=context_support_frame(real2019); d19=support_distance(support,X19); q75=float(np.quantile(d19,.75)) if len(d19) else np.nan
    Ptr,_=progression_frame(train)
    coverage=[]
    for f in (root/'data/v31_updates').glob('qwen_Q*_A*.csv'):
        up=pd.read_csv(f); src=up.get('source_type',pd.Series('',index=up.index)).astype(str); syn=up[src.eq('synthetic')].copy()
        if not len(syn): continue
        Xsyn,_=context_support_frame(syn); d=support_distance(support,Xsyn); ind=in_domain_mask(support,Xsyn)
        Psyn,_=progression_frame(syn); flags=progression_extrapolation_flags(Ptr,Psyn)
        coverage.append({'file':f.name,'n_synthetic':len(Xsyn),'mean_context_support_distance':float(np.mean(d)),
                         'p25_context_support_distance':float(np.quantile(d,.25)),'median_context_support_distance':float(np.median(d)),
                         'p75_context_support_distance':float(np.quantile(d,.75)),'p90_context_support_distance':float(np.quantile(d,.90)),
                         'context_ood_fraction':float(np.mean(~ind)),
                         'low_support_context_fraction_vs_real2019_q75':float(np.mean((d>q75)&ind)) if np.isfinite(q75) else np.nan,
                         'load_stage_extrapolation_fraction':float(np.mean(flags['load_stage_extrapolation'])),
                         'rutting_state_extrapolation_fraction':float(np.mean(flags['rutting_state_extrapolation']))})
    pd.DataFrame(coverage).to_csv(out/'qwen_synthetic_support_coverage.csv',index=False,encoding='utf-8-sig')

    counts=labels.context_support_quartile.value_counts().to_dict()
    audit={'definition':'V3.2 context support; cumulative load and previous rutting are progression coordinates, not hard OOD gates',
           'n_eval2021':int(len(labels)),'context_ood_n':int((~labels.context_in_domain.astype(bool)).sum()),
           'support_quartile_counts':{str(k):int(v) for k,v in counts.items()},
           'load_stage_extrapolation_fraction':float(np.nanmean(labels.load_stage_extrapolation.astype(float))),
           'rutting_state_extrapolation_fraction':float(np.nanmean(labels.rutting_state_extrapolation.astype(float))),
           'real2019_context_support_q75':q75,
           'interpretation':'Support quartiles rank future contexts by historical context distance without using 2021 rutting outcomes. They test support-dependent utility, not claimed expansion of load/temperature/structure inputs.'}
    _json(out/'state_support_v32_audit.json',audit)
    return audit


def recompute_contiguous_mssr(root):
    root=Path(root); final=root/'reports/v31/final'; src=final/'substitution_cluster_aware_summary.csv'
    if not src.exists(): raise FileNotFoundError('Run stage 09 once or provide substitution_cluster_aware_summary.csv')
    table=pd.read_csv(src); outputs={}
    audit_rows=[]
    for margin,label in [(.10,'10pct'),(.05,'5pct')]:
        result={}
        for (model,method),g in table.groupby(['model','method']):
            key=f'{model}|{method}'; result[key]=maximum_synthetic_substitution_ratio(g,margin)
            gg=g.sort_values('synthetic_fraction').copy(); failed=False
            for _,r in gg.iterrows():
                passed=float(r.upper_ci_relative_degradation)<margin
                contiguous=bool(passed and not failed)
                audit_rows.append({'margin':margin,'model':model,'method':method,'synthetic_fraction':float(r.synthetic_fraction),
                                   'upper_ci_relative_degradation':float(r.upper_ci_relative_degradation),'point_pass':passed,'contiguous_pass':contiguous})
                if not passed: failed=True
        _json(final/f'mssr_{label}_cluster_aware.json',result); outputs[label]=result
    pd.DataFrame(audit_rows).to_csv(final/'mssr_contiguity_audit.csv',index=False,encoding='utf-8-sig')
    _json(final/'mssr_definition_v32.json',{'definition':'contiguous MSSR: all evaluated lower replacement ratios must satisfy the margin before a higher ratio can count','margins':[0.10,0.05]})
    return outputs
