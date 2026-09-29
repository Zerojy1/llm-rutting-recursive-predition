from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json, joblib
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from v31_generator_arrays import windows_to_generator_arrays,joint_timegan_array,static_condition_from_runtime
from v31_timegan import train_timegan
from v31_timeweaver_lite import train_conditional_ddpm
from v31_master_data import STRUCTURE_COLUMNS

class TimeGANBaseline:
    """Classic joint TimeGAN plus nearest-condition retrieval for matched replacement.

    TimeGAN itself is unconditional; for a fair pavement conditional-use comparison we
    generate a joint bank and retrieve sequences whose generated metadata is nearest to the
    requested structure/load/temperature condition. This is explicitly reported as a
    retrieval-conditioned TimeGAN baseline, not a conditional TimeGAN architecture.
    """
    def __init__(self,model,scaler,n_features,seq_len): self.model=model; self.scaler=scaler; self.n_features=n_features; self.seq_len=seq_len; self.bank=None
    @classmethod
    def fit(cls,windows,future_steps=6,**kw):
        joint=joint_timegan_array(windows,future_steps); scaler=MinMaxScaler().fit(joint.reshape(-1,joint.shape[-1])); scaled=scaler.transform(joint.reshape(-1,joint.shape[-1])).reshape(joint.shape).astype('float32')
        model,h=train_timegan(scaled,**kw); return cls(model,scaler,joint.shape[-1],future_steps),h
    def sample_bank(self,n=5000,seed=42):
        torch.manual_seed(seed); dev=next(self.model.parameters()).device
        z=torch.rand(n,self.seq_len,self.n_features,device=dev)
        with torch.no_grad(): scaled=self.model.synthesize(z).cpu().numpy()
        arr=self.scaler.inverse_transform(scaled.reshape(-1,self.n_features)).reshape(scaled.shape)
        self.bank=arr; return arr
    def generator_fn(self,condition):
        if self.bank is None: self.sample_bank()
        L=len(condition['future_scenario']); st=static_condition_from_runtime(condition)
        dy=np.array([[x['cum_load_10k'],x['delta_load_10k'],x['temperature_c'],x['delta_temperature_c']] for x in condition['future_scenario']],float)
        if L<self.seq_len: dy=np.vstack([dy,np.repeat(dy[[-1]],self.seq_len-L,axis=0)])
        desired=np.concatenate([np.repeat(st[None,:],self.seq_len,axis=0),dy[:self.seq_len]],axis=1)
        bank_cond=self.bank[:,:,:-1]; # standardize distance by empirical bank dispersion
        scale=np.std(bank_cond.reshape(-1,bank_cond.shape[-1]),axis=0)+1e-6
        dist=np.mean(((bank_cond-desired[None,:,:])/scale)**2,axis=(1,2)); idx=int(np.argmin(dist))
        return self.bank[idx,:L,-1].tolist()

class ConditionalDDPMBaseline:
    def __init__(self,model,static_scaler,dynamic_scaler,target_scaler): self.model=model; self.ss=static_scaler; self.ds=dynamic_scaler; self.ys=target_scaler
    @classmethod
    def fit(cls,windows,future_steps=6,**kw):
        a=windows_to_generator_arrays(windows,future_steps); ss=StandardScaler().fit(a['static']); ds=StandardScaler().fit(a['dynamic'].reshape(-1,a['dynamic'].shape[-1])); ys=StandardScaler().fit(a['target'].reshape(-1,1))
        s=ss.transform(a['static']).astype('float32'); d=ds.transform(a['dynamic'].reshape(-1,a['dynamic'].shape[-1])).reshape(a['dynamic'].shape).astype('float32'); y=ys.transform(a['target'].reshape(-1,1)).reshape(a['target'].shape).astype('float32')
        model,h=train_conditional_ddpm(y,s,d,**kw); return cls(model,ss,ds,ys),h
    def generator_fn(self,condition):
        st=static_condition_from_runtime(condition)[None,:]; dy=np.array([[x['cum_load_10k'],x['delta_load_10k'],x['temperature_c'],x['delta_temperature_c']] for x in condition['future_scenario']],float)
        L=len(dy); cfg=self.model.cfg
        if L<cfg.seq_len: dy=np.vstack([dy,np.repeat(dy[[-1]],cfg.seq_len-L,axis=0)])
        s=torch.tensor(self.ss.transform(st),dtype=torch.float32,device=next(self.model.parameters()).device); d=self.ds.transform(dy[:cfg.seq_len]).reshape(1,cfg.seq_len,-1)
        with torch.no_grad(): y=self.model.sample(s,torch.tensor(d,dtype=torch.float32,device=s.device)).cpu().numpy().reshape(-1,1)
        return self.ys.inverse_transform(y).reshape(-1)[:L].tolist()

def save_baseline(obj,path,history):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); torch.save({'model_state':obj.model.state_dict(),'model_cfg':asdict(obj.model.cfg)},path)
    joblib.dump({k:v for k,v in obj.__dict__.items() if k!='model' and k!='bank'},path.with_suffix('.scalers.joblib'))
    path.with_suffix('.history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')

def load_timegan_baseline(path,device='auto'):
    from v31_timegan import TimeGANConfig,TimeGANModel
    path=Path(path); ck=torch.load(path,map_location='cpu',weights_only=False); cfg=TimeGANConfig(**ck['model_cfg']); model=TimeGANModel(cfg); model.load_state_dict(ck['model_state']); dev=torch.device('cuda' if device=='auto' and torch.cuda.is_available() else ('cpu' if device=='auto' else device)); model.to(dev).eval(); meta=joblib.load(path.with_suffix('.scalers.joblib')); return TimeGANBaseline(model,meta['scaler'],meta['n_features'],meta['seq_len'])

def load_ddpm_baseline(path,device='auto'):
    from v31_timeweaver_lite import ConditionalDiffusionConfig,MetadataConditionedDDPM
    path=Path(path); ck=torch.load(path,map_location='cpu',weights_only=False); cfg=ConditionalDiffusionConfig(**ck['model_cfg']); model=MetadataConditionedDDPM(cfg); model.load_state_dict(ck['model_state']); dev=torch.device('cuda' if device=='auto' and torch.cuda.is_available() else ('cpu' if device=='auto' else device)); model.to(dev).eval(); meta=joblib.load(path.with_suffix('.scalers.joblib')); return ConditionalDDPMBaseline(model,meta['ss'],meta['ds'],meta['ys'])
