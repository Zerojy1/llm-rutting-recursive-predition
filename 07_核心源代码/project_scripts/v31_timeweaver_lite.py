from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn

@dataclass(frozen=True)
class ConditionalDiffusionConfig:
    seq_len:int=6
    target_dim:int=1
    static_dim:int=18
    dynamic_dim:int=4
    hidden_dim:int=64
    diffusion_steps:int=100

class _Denoiser(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        self.static=nn.Sequential(nn.Linear(cfg.static_dim,cfg.hidden_dim),nn.SiLU(),nn.Linear(cfg.hidden_dim,cfg.hidden_dim))
        self.dynamic=nn.Linear(cfg.dynamic_dim,cfg.hidden_dim)
        self.target=nn.Linear(cfg.target_dim,cfg.hidden_dim)
        self.time=nn.Embedding(cfg.diffusion_steps+1,cfg.hidden_dim)
        layer=nn.TransformerEncoderLayer(d_model=cfg.hidden_dim,nhead=4,dim_feedforward=cfg.hidden_dim*2,batch_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(layer,num_layers=2); self.out=nn.Linear(cfg.hidden_dim,cfg.target_dim)
    def forward(self,noisy,t,static,dynamic):
        s=self.static(static)[:,None,:]; z=self.target(noisy)+self.dynamic(dynamic)+s+self.time(t)[:,None,:]
        return self.out(self.encoder(z))

class MetadataConditionedDDPM(nn.Module):
    """Time-Weaver-inspired heterogeneous-metadata conditional diffusion baseline.

    IMPORTANT: ICML 2024 Time Weaver has no public official implementation as of the
    package freeze. This module is a transparent conceptual reimplementation for the
    pavement benchmark, not a claim of byte-for-byte reproduction of the authors' model.
    """
    def __init__(self,cfg:ConditionalDiffusionConfig):
        super().__init__(); self.cfg=cfg; self.denoiser=_Denoiser(cfg)
        betas=torch.linspace(1e-4,0.02,cfg.diffusion_steps); alphas=1-betas; abar=torch.cumprod(alphas,0)
        self.register_buffer('betas',betas); self.register_buffer('alphas',alphas); self.register_buffer('abar',abar)
    def q_sample(self,y,t,noise=None):
        noise=torch.randn_like(y) if noise is None else noise; a=self.abar[t-1].view(-1,1,1)
        return torch.sqrt(a)*y+torch.sqrt(1-a)*noise,noise
    @torch.no_grad()
    def sample(self,static,dynamic):
        b=static.shape[0]; y=torch.randn(b,self.cfg.seq_len,self.cfg.target_dim,device=static.device)
        for step in range(self.cfg.diffusion_steps,0,-1):
            t=torch.full((b,),step,device=static.device,dtype=torch.long); eps=self.denoiser(y,t,static,dynamic); alpha=self.alphas[step-1]; abar=self.abar[step-1]; beta=self.betas[step-1]
            mean=(y-(beta/torch.sqrt(1-abar))*eps)/torch.sqrt(alpha)
            y=mean if step==1 else mean+torch.sqrt(beta)*torch.randn_like(y)
        return y


def train_conditional_ddpm(target,static,dynamic,steps=2000,batch_size=64,hidden=64,diffusion_steps=100,lr=1e-3,device='auto',seed=42):
    """Train the transparent Time-Weaver-inspired metadata-conditioned DDPM baseline."""
    import random, numpy as np, torch
    from torch.utils.data import DataLoader,TensorDataset
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev=torch.device('cuda' if device=='auto' and torch.cuda.is_available() else ('cpu' if device=='auto' else device))
    y=np.asarray(target,dtype='float32'); s=np.asarray(static,dtype='float32'); d=np.asarray(dynamic,dtype='float32')
    cfg=ConditionalDiffusionConfig(seq_len=y.shape[1],target_dim=y.shape[2],static_dim=s.shape[1],dynamic_dim=d.shape[2],hidden_dim=hidden,diffusion_steps=diffusion_steps)
    model=MetadataConditionedDDPM(cfg).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4); mse=nn.MSELoss()
    ds=TensorDataset(torch.tensor(y),torch.tensor(s),torch.tensor(d)); gen=torch.Generator().manual_seed(seed); loader=DataLoader(ds,batch_size=min(batch_size,len(ds)),shuffle=True,generator=gen)
    def cycle():
        while True:
            for b in loader: yield tuple(x.to(dev) for x in b)
    it=cycle(); hist=[]
    for step in range(1,steps+1):
        yy,ss,dd=next(it); t=torch.randint(1,cfg.diffusion_steps+1,(len(yy),),device=dev); noisy,noise=model.q_sample(yy,t)
        opt.zero_grad(); pred=model.denoiser(noisy,t,ss,dd); loss=mse(pred,noise); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); hist.append({'step':step,'loss':float(loss.detach().cpu())})
    return model,hist
