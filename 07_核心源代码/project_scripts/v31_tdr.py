from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import copy, random, sys
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader,TensorDataset
from v31_training import StepBudget,cosine_lr,overfitting_index

FIRST_MODEL=Path(__file__).resolve().parents[1]/'first_model'
if str(FIRST_MODEL) not in sys.path: sys.path.insert(0,str(FIRST_MODEL))
import TDR_PRSN as tdr

@dataclass(frozen=True)
class TDRStableConfig:
    d_model:int=32
    dropout:float=.20
    weight_decay:float=5e-4
    lr:float=8e-4

STABILITY_CANDIDATES={
    'TDR-A':TDRStableConfig(64,.10,1e-4),
    'TDR-B':TDRStableConfig(48,.15,5e-4),
    'TDR-C':TDRStableConfig(32,.20,5e-4),
    'TDR-D':TDRStableConfig(32,.25,1e-3),
}

def _scale(Xtr,Xv,ytr):
    sx=StandardScaler().fit(Xtr.reshape(-1,Xtr.shape[-1])); tx=lambda x:sx.transform(x.reshape(-1,x.shape[-1])).reshape(x.shape).astype('float32')
    sy=StandardScaler().fit(np.log1p(np.maximum(ytr,0)).reshape(-1,1))
    return sx,sy,tx(Xtr),tx(Xv)



def _evaluate_preserving_mode(model, X_scaled, X_raw, y_scaler, prev_y, prev_peak, y_mean_t, y_std_t, device, batch_size=512):
    """Run TDR evaluation without leaking eval() state back into an active training loop."""
    was_training = bool(model.training)
    try:
        return tdr.evaluate_tdr_model(
            model, X_scaled, X_raw, y_scaler, np.asarray(prev_y), np.asarray(prev_peak),
            y_mean_t, y_std_t, device, batch_size=batch_size
        )
    finally:
        model.train(was_training)

def fit_tdr_step_budget(X_train,y_train,prev_train,peak_train,X_val,y_val,prev_val,peak_val,stable:TDRStableConfig,
                        max_steps=2400,warmup_steps=100,val_every=50,patience_checks=6,batch_size=64,seed=42,device='auto',variant='full'):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev=torch.device('cuda' if device=='auto' and torch.cuda.is_available() else ('cpu' if device=='auto' else device))
    Xtr=np.asarray(X_train,np.float32); Xv=np.asarray(X_val,np.float32); ytr=np.asarray(y_train,np.float32); yv=np.asarray(y_val,np.float32)
    sx,sy,Xtrs,Xvs=_scale(Xtr,Xv,ytr)
    yts=sy.transform(np.log1p(np.maximum(ytr,0)).reshape(-1,1)).reshape(-1).astype('float32')
    ym=torch.tensor(float(sy.mean_[0]),device=dev); ys=torch.tensor(float(sy.scale_[0]),device=dev)
    cfg=tdr.Config(input='',seq_len=Xtr.shape[1],epochs=120,batch_size=batch_size,lr=stable.lr,weight_decay=stable.weight_decay,d_model=stable.d_model,dropout=stable.dropout,
                   run_xgb=False,run_dl_baselines=False,run_ablation=False,enable_residual_calibration=False,delta_loss_weight=.25,dual_state_loss_weight=.05,reversible_aux_weight=.03)
    model,_,_,_=tdr.build_neural_model('TDRPRSN',Xtr.shape[-1],cfg,variant,dev)
    ds=TensorDataset(torch.tensor(Xtrs),torch.tensor(Xtr),torch.tensor(yts.reshape(-1,1)),torch.tensor(ytr.reshape(-1,1)),torch.tensor(prev_train,dtype=torch.float32),torch.tensor(peak_train,dtype=torch.float32))
    loader=DataLoader(ds,batch_size=min(batch_size,len(ds)),shuffle=True,generator=torch.Generator().manual_seed(seed)); it=iter(loader)
    opt=torch.optim.AdamW(model.parameters(),lr=stable.lr,weight_decay=stable.weight_decay); budget=StepBudget(max_steps,warmup_steps,val_every,patience_checks)
    best=float('inf'); best_state=copy.deepcopy(model.state_dict()); bad=0; hist=[]; steps=0
    def pred(xsc,xraw,py,pp):
        d=_evaluate_preserving_mode(model,xsc,xraw,sy,py,pp,ym,ys,dev,batch_size=512)
        return d['y_pred_mm']
    while steps<max_steps:
        try: xs,xr,yss,yrr,py,pp=next(it)
        except StopIteration: it=iter(loader); xs,xr,yss,yrr,py,pp=next(it)
        steps+=1; lr=cosine_lr(steps,budget,stable.lr,stable.lr*.05)
        for pg in opt.param_groups: pg['lr']=lr
        xs,xr,yss,yrr,py,pp=[z.to(dev) for z in (xs,xr,yss,yrr,py,pp)]
        opt.zero_grad(set_to_none=True); out=model(xs,xr,py,pp,ym,ys)
        pseudo_epoch=max(1,int(round(120*steps/max_steps)))
        loss,_=tdr.tdr_loss(out,yss,yrr,cfg.risk_threshold_mm,pseudo_epoch,120,use_nll=False,use_risk=False,prev_y_raw=py,
                            delta_loss_weight=.25,delta_loss_scale_mm=8.,dual_state_loss_weight=.05,reversible_aux_weight=.03,reversible_reg_weight=5e-4)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if steps%val_every==0 or steps==max_steps:
            trp=pred(Xtrs,Xtr,prev_train,peak_train); vap=pred(Xvs,Xv,prev_val,peak_val)
            tr=float(np.sqrt(mean_squared_error(ytr,trp))); va=float(np.sqrt(mean_squared_error(yv,vap)))
            hist.append({'step':steps,'train_rmse':tr,'val_rmse':va,'oi':overfitting_index(tr,va),'lr':lr})
            if va<best-1e-10: best=va; best_state=copy.deepcopy(model.state_dict()); bad=0
            else:
                bad+=1
                if bad>=patience_checks: break
    model.load_state_dict(best_state)
    return {'model':model,'x_scaler':sx,'y_scaler':sy,'best_val_rmse':best,'steps_completed':steps,'history':hist,
            'parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad),'device':str(dev),'config':stable,'variant':str(variant)}

def predict_tdr(fit_result,X,prev,peak,batch_size=512):
    X=np.asarray(X,np.float32); sx=fit_result['x_scaler']; sy=fit_result['y_scaler']; dev=next(fit_result['model'].parameters()).device
    Xs=sx.transform(X.reshape(-1,X.shape[-1])).reshape(X.shape).astype('float32')
    ym=torch.tensor(float(sy.mean_[0]),dtype=torch.float32,device=dev); ys=torch.tensor(float(sy.scale_[0]),dtype=torch.float32,device=dev)
    return tdr.evaluate_tdr_model(fit_result['model'],Xs,X,sy,np.asarray(prev),np.asarray(peak),ym,ys,dev,batch_size=batch_size)['y_pred_mm']
