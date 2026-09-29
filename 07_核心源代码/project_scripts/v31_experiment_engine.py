from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error
from v31_downstream_data import build_real_downstream_frame,build_sequence_arrays_simple
from v31_substitution import _stratified_take
from v31_masked_generation import generate_masked_update
from v31_synthetic_rows import assemble_masked_update_frame
from v31_gru import GRUConfig,fit_gru_step_budget,predict_gru
from v31_statistics import persistence_skill


def select_retained_indices(master,synthetic_fraction,seed,year=2019):
    pool=master[(master.Year==year)&(master.Rutting_Observed==1)].copy()
    n=int(round(len(pool)*(1-float(synthetic_fraction))))
    keep=_stratified_take(pool,n,seed)
    return set(zip(keep.STR_name.astype(str),pd.to_numeric(keep.Observation_Index).astype(int)))


def _arrays_for_target_years(frame,target_years,seq_len=6):
    X,y,p,pk,meta=build_sequence_arrays_simple(frame,seq_len=seq_len)
    mask=meta.Year.astype(int).isin(list(target_years)).to_numpy()
    return X[mask],y[mask],p[mask],pk[mask],meta.loc[mask].reset_index(drop=True)


def fixed_real_evaluation_arrays(master,seq_len=6):
    real=build_real_downstream_frame(master)
    return {'dev2020':_arrays_for_target_years(real,[2020],seq_len),'eval2021':_arrays_for_target_years(real,[2021],seq_len)}


def training_frame_with_update(master,update_frame,year=2019):
    """Build a leakage-safe training panel while retaining every exposure row.

    Historical rows retain their observed/missing rutting flags. In the update year all
    original rutting targets are first hidden; only retained-real or synthetic rows in
    ``update_frame`` are overlaid as supervised states. Thus withheld observations remain
    available as load/temperature exposure history but never leak their true rutting label.
    """
    base=master[master.Year<=year].copy()
    mask=base.Year.eq(year)
    if 'Rutting_Observed' in base: base.loc[mask,'Rutting_Observed']=0
    if 'Rutting_mm' in base: base.loc[mask,'Rutting_mm']=np.nan
    keys=['STR_name','Observation_Index']
    upd=update_frame.copy()
    cols=sorted(set(base.columns).union(upd.columns))
    base=base.reindex(columns=cols); upd=upd.reindex(columns=cols)
    for c in cols:
        if c in upd.columns and upd[c].dtype == object and base[c].dtype != object:
            base[c]=base[c].astype(object)
    if len(upd):
        overlay={(str(r.STR_name),int(r.Observation_Index)):r for _,r in upd.iterrows()}
        for idx,r in base.loc[mask].iterrows():
            key=(str(r.STR_name),int(r.Observation_Index))
            rr=overlay.get(key)
            if rr is not None:
                for c in cols: base.at[idx,c]=rr[c]
    return base.sort_values(['STR_name','Observation_Index'],kind='mergesort').reset_index(drop=True)


def make_masked_update(master,synthetic_fraction,allocation_seed,generator_fn,year=2019):
    retained=select_retained_indices(master,synthetic_fraction,allocation_seed,year)
    states=generate_masked_update(master,retained,generator_fn,year=year,history_steps=4,max_horizon=6)
    return assemble_masked_update_frame(master,states,year),retained,states


def evaluate_predictions(y,pred,prev):
    rmse=float(np.sqrt(mean_squared_error(y,pred))); prmse=float(np.sqrt(mean_squared_error(y,np.asarray(prev).reshape(-1))))
    inc_true=np.asarray(y)-np.asarray(prev).reshape(-1); inc_pred=np.asarray(pred)-np.asarray(prev).reshape(-1)
    corr=float(np.corrcoef(inc_true,inc_pred)[0,1]) if len(y)>2 and np.std(inc_true)>0 and np.std(inc_pred)>0 else np.nan
    return {'rmse':rmse,'mae':float(mean_absolute_error(y,pred)),'persistence_rmse':prmse,'persistence_skill':persistence_skill(rmse,prmse),'increment_corr':corr}


def run_gru_condition(master,update_frame,gru_cfg:GRUConfig,seed=42,max_steps=2400,warmup_steps=100,val_every=50,patience_checks=6,seq_len=6):
    train_master=training_frame_with_update(master,update_frame)
    train_real=build_real_downstream_frame(train_master)
    Xtr,ytr,ptr,pktr,mtr=_arrays_for_target_years(train_real,range(2016,2020),seq_len)
    fixed=fixed_real_evaluation_arrays(master,seq_len); Xv,yv,pv,pkv,mv=fixed['dev2020']; Xe,ye,pe,pke,me=fixed['eval2021']
    fit=fit_gru_step_budget(Xtr,ytr,ptr,Xv,yv,pv,gru_cfg,max_steps=max_steps,warmup_steps=warmup_steps,val_every=val_every,patience_checks=patience_checks,seed=seed)
    pvhat=predict_gru(fit,Xv,pv); pehat=predict_gru(fit,Xe,pe)
    return {'fit':fit,'dev_metrics':evaluate_predictions(yv,pvhat,pv),'eval_metrics':evaluate_predictions(ye,pehat,pe),
            'dev_predictions':pd.DataFrame({'STR_name':mv.STR_name,'Observation_Index':mv.Observation_Order,'y_true':yv,'y_pred':pvhat,'prev':pv.reshape(-1)}),
            'eval_predictions':pd.DataFrame({'STR_name':me.STR_name,'Observation_Index':me.Observation_Order,'y_true':ye,'y_pred':pehat,'prev':pe.reshape(-1)}),
            'n_train':len(ytr)}


def _fit_gru_arrays(Xtr,ytr,ptr,Xv,yv,pv,Xe,ye,pe,gru_cfg,seed,max_steps,warmup_steps,val_every,patience_checks):
    fit=fit_gru_step_budget(Xtr,ytr,ptr,Xv,yv,pv,gru_cfg,max_steps=max_steps,warmup_steps=warmup_steps,val_every=val_every,patience_checks=patience_checks,seed=seed)
    pvhat=predict_gru(fit,Xv,pv); pehat=predict_gru(fit,Xe,pe)
    return fit,pvhat,pehat


def run_gru_real_bootstrap(master,retained_keys,gru_cfg:GRUConfig,seed=42,allocation_seed=20260814,max_steps=2400,warmup_steps=100,val_every=50,patience_checks=6,seq_len=6):
    update=master[(master.Year==2019)&(master.Rutting_Observed==1)].copy()
    mask=[(str(s),int(o)) in retained_keys for s,o in zip(update.STR_name,update.Observation_Index)]; retained=update.loc[mask].copy()
    train_master=training_frame_with_update(master,retained); trframe=build_real_downstream_frame(train_master)
    X,y,p,pk,meta=build_sequence_arrays_simple(trframe,seq_len); hmask=meta.Year.astype(int).le(2018).to_numpy(); umask=meta.Year.astype(int).eq(2019).to_numpy()
    # target update sample budget is full-real number of 2019 supervised transitions
    full=build_real_downstream_frame(master[master.Year<=2019].copy()); Xf,yf,pf,pkf,mf=build_sequence_arrays_simple(full,seq_len); n_target=int(mf.Year.astype(int).eq(2019).sum())
    idxu=np.flatnonzero(umask); rng=np.random.default_rng(allocation_seed+seed)
    if len(idxu)==0: raise ValueError('real-bootstrap needs retained 2019 samples')
    chosen=rng.choice(idxu,size=n_target,replace=True)
    Xtr=np.concatenate([X[hmask],X[chosen]]); ytr=np.concatenate([y[hmask],y[chosen]]); ptr=np.concatenate([p[hmask],p[chosen]])
    fixed=fixed_real_evaluation_arrays(master,seq_len); Xv,yv,pv,pkv,mv=fixed['dev2020']; Xe,ye,pe,pke,me=fixed['eval2021']
    fit,pvhat,pehat=_fit_gru_arrays(Xtr,ytr,ptr,Xv,yv,pv,Xe,ye,pe,gru_cfg,seed,max_steps,warmup_steps,val_every,patience_checks)
    return {'fit':fit,'dev_metrics':evaluate_predictions(yv,pvhat,pv),'eval_metrics':evaluate_predictions(ye,pehat,pe),'n_train':len(ytr),
            'dev_predictions':pd.DataFrame({'STR_name':mv.STR_name,'Observation_Index':mv.Observation_Order,'y_true':yv,'y_pred':pvhat,'prev':pv.reshape(-1)}),
            'eval_predictions':pd.DataFrame({'STR_name':me.STR_name,'Observation_Index':me.Observation_Order,'y_true':ye,'y_pred':pehat,'prev':pe.reshape(-1)})}


def run_tdr_condition(master,update_frame,stable_cfg,seed=42,max_steps=2400,warmup_steps=100,val_every=50,patience_checks=6,seq_len=6):
    from v31_tdr import fit_tdr_step_budget,predict_tdr
    train_master=training_frame_with_update(master,update_frame); train_real=build_real_downstream_frame(train_master)
    Xtr,ytr,ptr,pktr,mtr=_arrays_for_target_years(train_real,range(2016,2020),seq_len); fixed=fixed_real_evaluation_arrays(master,seq_len)
    Xv,yv,pv,pkv,mv=fixed['dev2020']; Xe,ye,pe,pke,me=fixed['eval2021']
    fit=fit_tdr_step_budget(Xtr,ytr,ptr,pktr,Xv,yv,pv,pkv,stable_cfg,max_steps=max_steps,warmup_steps=warmup_steps,val_every=val_every,patience_checks=patience_checks,batch_size=64,seed=seed)
    pvh=predict_tdr(fit,Xv,pv,pkv); peh=predict_tdr(fit,Xe,pe,pke)
    return {'fit':fit,'dev_metrics':evaluate_predictions(yv,pvh,pv),'eval_metrics':evaluate_predictions(ye,peh,pe),'n_train':len(ytr),
            'dev_predictions':pd.DataFrame({'STR_name':mv.STR_name,'Observation_Index':mv.Observation_Order,'y_true':yv,'y_pred':pvh,'prev':pv.reshape(-1)}),
            'eval_predictions':pd.DataFrame({'STR_name':me.STR_name,'Observation_Index':me.Observation_Order,'y_true':ye,'y_pred':peh,'prev':pe.reshape(-1)})}


def run_tdr_real_bootstrap(master,retained_keys,stable_cfg,seed=42,allocation_seed=20260814,max_steps=2400,warmup_steps=100,val_every=50,patience_checks=6,seq_len=6):
    from v31_tdr import fit_tdr_step_budget,predict_tdr
    update=master[(master.Year==2019)&(master.Rutting_Observed==1)].copy(); mask=[(str(s),int(o)) in retained_keys for s,o in zip(update.STR_name,update.Observation_Index)]; retained=update.loc[mask].copy()
    train_master=training_frame_with_update(master,retained); trframe=build_real_downstream_frame(train_master); X,y,p,pk,meta=build_sequence_arrays_simple(trframe,seq_len)
    hmask=meta.Year.astype(int).le(2018).to_numpy(); umask=meta.Year.astype(int).eq(2019).to_numpy(); full=build_real_downstream_frame(master[master.Year<=2019].copy()); Xf,yf,pf,pkf,mf=build_sequence_arrays_simple(full,seq_len); n_target=int(mf.Year.astype(int).eq(2019).sum()); idxu=np.flatnonzero(umask)
    if len(idxu)==0: raise ValueError('real-bootstrap needs retained 2019 samples')
    chosen=np.random.default_rng(allocation_seed+seed).choice(idxu,size=n_target,replace=True)
    Xtr=np.concatenate([X[hmask],X[chosen]]); ytr=np.concatenate([y[hmask],y[chosen]]); ptr=np.concatenate([p[hmask],p[chosen]]); pktr=np.concatenate([pk[hmask],pk[chosen]])
    fixed=fixed_real_evaluation_arrays(master,seq_len); Xv,yv,pv,pkv,mv=fixed['dev2020']; Xe,ye,pe,pke,me=fixed['eval2021']
    fit=fit_tdr_step_budget(Xtr,ytr,ptr,pktr,Xv,yv,pv,pkv,stable_cfg,max_steps=max_steps,warmup_steps=warmup_steps,val_every=val_every,patience_checks=patience_checks,seed=seed)
    pvh=predict_tdr(fit,Xv,pv,pkv); peh=predict_tdr(fit,Xe,pe,pke)
    return {'fit':fit,'dev_metrics':evaluate_predictions(yv,pvh,pv),'eval_metrics':evaluate_predictions(ye,peh,pe),'n_train':len(ytr),
            'dev_predictions':pd.DataFrame({'STR_name':mv.STR_name,'Observation_Index':mv.Observation_Order,'y_true':yv,'y_pred':pvh,'prev':pv.reshape(-1)}),
            'eval_predictions':pd.DataFrame({'STR_name':me.STR_name,'Observation_Index':me.Observation_Order,'y_true':ye,'y_pred':peh,'prev':pe.reshape(-1)})}
