from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn

@dataclass(frozen=True)
class GRUConfig:
    input_dim:int
    hidden_dim:int=32
    num_layers:int=1
    dropout:float=.1

CAPACITY_LADDER={
'Tiny':dict(hidden_dim=16,num_layers=1), 'Small':dict(hidden_dim=32,num_layers=1),
'Medium':dict(hidden_dim=32,num_layers=2), 'Medium+':dict(hidden_dim=64,num_layers=2),
'Large':dict(hidden_dim=64,num_layers=3), 'XL':dict(hidden_dim=128,num_layers=3)}

class RuttingGRU(nn.Module):
    def __init__(self,cfg:GRUConfig):
        super().__init__(); self.cfg=cfg
        drop=cfg.dropout if cfg.num_layers>1 else 0.0
        self.gru=nn.GRU(cfg.input_dim,cfg.hidden_dim,num_layers=cfg.num_layers,batch_first=True,dropout=drop)
        self.head=nn.Sequential(nn.Linear(cfg.hidden_dim+1,cfg.hidden_dim),nn.ReLU(),nn.Dropout(cfg.dropout),nn.Linear(cfg.hidden_dim,1))
    def forward(self,x,prev_rutting):
        h,_=self.gru(x); last=h[:,-1,:]; z=torch.cat([last,prev_rutting],dim=1); delta=self.head(z); return prev_rutting+delta

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fit_gru_step_budget(X_train, y_train, prev_train, X_val, y_val, prev_val, config: GRUConfig,
                        max_steps=2400, warmup_steps=100, val_every=50, patience_checks=6,
                        batch_size=64, seed=42, device='auto', learning_rate=8e-4, weight_decay=5e-4):
    """Train a GRU with an exact optimizer-update budget for fair Real/Synthetic comparison."""
    import copy, math, random
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    from v31_training import cosine_lr, overfitting_index, StepBudget
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev=torch.device('cuda' if device=='auto' and torch.cuda.is_available() else ('cpu' if device=='auto' else device))
    # train-only scaling, flattened feature-wise
    sx=StandardScaler().fit(np.asarray(X_train).reshape(-1,X_train.shape[-1]))
    def tx(x): return sx.transform(np.asarray(x).reshape(-1,x.shape[-1])).reshape(x.shape).astype('float32')
    Xtr=tx(X_train); Xv=tx(X_val)
    ds=TensorDataset(torch.tensor(Xtr),torch.tensor(np.asarray(y_train).reshape(-1,1),dtype=torch.float32),torch.tensor(prev_train,dtype=torch.float32))
    gen=torch.Generator().manual_seed(seed)
    loader=DataLoader(ds,batch_size=min(batch_size,len(ds)),shuffle=True,generator=gen,drop_last=False)
    model=RuttingGRU(config).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    it=iter(loader); best=float('inf'); best_state=None; bad=0; hist=[]; steps=0
    def predict(x,p):
        model.eval(); out=[]
        with torch.no_grad():
            for a in range(0,len(x),512):
                xx=torch.tensor(x[a:a+512],dtype=torch.float32,device=dev)
                pp=torch.tensor(p[a:a+512],dtype=torch.float32,device=dev)
                out.append(model(xx,pp).cpu().numpy().reshape(-1))
        return np.concatenate(out)
    while steps < max_steps:
        try: xb,yb,pb=next(it)
        except StopIteration:
            it=iter(loader); xb,yb,pb=next(it)
        steps+=1
        lr=cosine_lr(steps,StepBudget(max_steps=max_steps,warmup_steps=warmup_steps,val_every=val_every,patience_checks=patience_checks),learning_rate,learning_rate*0.05)
        for pg in opt.param_groups: pg['lr']=lr
        model.train(); opt.zero_grad(set_to_none=True)
        pred=model(xb.to(dev),pb.to(dev)); loss=torch.nn.functional.smooth_l1_loss(pred,yb.to(dev)); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if steps % val_every==0 or steps==max_steps:
            trp=predict(Xtr,prev_train); vap=predict(Xv,prev_val)
            tr=float(np.sqrt(mean_squared_error(y_train,trp))); va=float(np.sqrt(mean_squared_error(y_val,vap)))
            hist.append({'step':steps,'train_rmse':tr,'val_rmse':va,'lr':lr,'oi':overfitting_index(tr,va)})
            if va < best-1e-10:
                best=va; best_state=copy.deepcopy(model.state_dict()); bad=0
            else:
                bad+=1
                if bad>=patience_checks: break
    if best_state is not None: model.load_state_dict(best_state)
    return {'model':model,'x_scaler':sx,'best_val_rmse':best,'history':hist,'steps_completed':steps,
            'parameter_count':count_parameters(model),'device':str(dev)}


def predict_gru(fit_result,X,prev,batch_size=512):
    import numpy as np, torch
    model=fit_result['model']; sx=fit_result['x_scaler']; dev=next(model.parameters()).device
    x=np.asarray(X); xs=sx.transform(x.reshape(-1,x.shape[-1])).reshape(x.shape).astype('float32'); p=np.asarray(prev,np.float32)
    model.eval(); out=[]
    with torch.no_grad():
        for a in range(0,len(xs),batch_size):
            out.append(model(torch.tensor(xs[a:a+batch_size],device=dev),torch.tensor(p[a:a+batch_size],device=dev)).cpu().numpy().reshape(-1))
    return np.concatenate(out) if out else np.array([],float)
