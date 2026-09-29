from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn

@dataclass(frozen=True)
class TimeGANConfig:
    seq_len:int
    n_features:int
    hidden_dim:int=24
    num_layers:int=3

class _RNNMap(nn.Module):
    def __init__(self,in_dim,hid,layers,out_dim,activation='sigmoid'):
        super().__init__(); self.rnn=nn.GRU(in_dim,hid,layers,batch_first=True); self.proj=nn.Linear(hid,out_dim); self.activation=activation
    def forward(self,x):
        h,_=self.rnn(x); y=self.proj(h)
        if self.activation=='sigmoid': y=torch.sigmoid(y)
        return y

class TimeGANModel(nn.Module):
    """Compact PyTorch implementation of the TimeGAN component topology.

    Training utilities below follow the original embedding/supervised/adversarial objectives,
    while keeping the code modern and CPU/GPU portable.
    """
    def __init__(self,cfg:TimeGANConfig):
        super().__init__(); self.cfg=cfg
        self.embedder=_RNNMap(cfg.n_features,cfg.hidden_dim,cfg.num_layers,cfg.hidden_dim)
        self.recovery=_RNNMap(cfg.hidden_dim,cfg.hidden_dim,cfg.num_layers,cfg.n_features)
        self.generator=_RNNMap(cfg.n_features,cfg.hidden_dim,cfg.num_layers,cfg.hidden_dim)
        self.supervisor=_RNNMap(cfg.hidden_dim,cfg.hidden_dim,max(1,cfg.num_layers-1),cfg.hidden_dim)
        self.discriminator=_RNNMap(cfg.hidden_dim,cfg.hidden_dim,cfg.num_layers,1,activation='none')
    def synthesize(self,z):
        return self.recovery(self.supervisor(self.generator(z)))


def moment_loss(real,fake):
    mr,sr=real.mean((0,1)),real.std((0,1),unbiased=False); mf,sf=fake.mean((0,1)),fake.std((0,1),unbiased=False)
    return torch.mean(torch.abs(mr-mf))+torch.mean(torch.abs(sr-sf))


def train_timegan(data,steps_embed=1000,steps_supervisor=1000,steps_joint=2000,batch_size=64,hidden_dim=24,num_layers=2,lr=1e-3,device='auto',seed=42):
    """Practical compact TimeGAN trainer for the V3.1 benchmark.

    It follows the original three-stage idea (embedding reconstruction, supervised latent
    dynamics, joint adversarial training) and is intentionally transparent rather than a
    dependency on the legacy TensorFlow reference implementation.
    """
    import random, numpy as np, torch
    from torch.utils.data import DataLoader,TensorDataset
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev=torch.device('cuda' if device=='auto' and torch.cuda.is_available() else ('cpu' if device=='auto' else device))
    arr=np.asarray(data,dtype='float32'); cfg=TimeGANConfig(arr.shape[1],arr.shape[2],hidden_dim,num_layers); m=TimeGANModel(cfg).to(dev)
    ds=TensorDataset(torch.tensor(arr)); g=torch.Generator().manual_seed(seed); loader=DataLoader(ds,batch_size=min(batch_size,len(ds)),shuffle=True,generator=g)
    def cycle():
        while True:
            for (x,) in loader: yield x.to(dev)
    it=cycle(); hist=[]; mse=nn.MSELoss(); bce=nn.BCEWithLogitsLoss()
    opt_er=torch.optim.Adam(list(m.embedder.parameters())+list(m.recovery.parameters()),lr=lr)
    for s in range(steps_embed):
        x=next(it); opt_er.zero_grad(); h=m.embedder(x); xr=m.recovery(h); loss=mse(xr,x); loss.backward(); opt_er.step(); hist.append({'stage':'embed','step':s+1,'loss':float(loss.detach().cpu())})
    opt_s=torch.optim.Adam(m.supervisor.parameters(),lr=lr)
    for s in range(steps_supervisor):
        x=next(it); opt_s.zero_grad(); h=m.embedder(x).detach(); hs=m.supervisor(h); loss=mse(h[:,1:,:],hs[:,:-1,:]); loss.backward(); opt_s.step(); hist.append({'stage':'supervisor','step':s+1,'loss':float(loss.detach().cpu())})
    opt_g=torch.optim.Adam(list(m.generator.parameters())+list(m.supervisor.parameters())+list(m.recovery.parameters()),lr=lr)
    opt_d=torch.optim.Adam(m.discriminator.parameters(),lr=lr)
    for s in range(steps_joint):
        x=next(it); b,L,F=x.shape; z=torch.rand(b,L,F,device=dev)
        # generator/supervisor/recovery update
        opt_g.zero_grad(); h_fake=m.supervisor(m.generator(z)); x_fake=m.recovery(h_fake); d_fake=m.discriminator(h_fake)
        adv=bce(d_fake,torch.ones_like(d_fake)); mom=moment_loss(x,x_fake); sup=mse(m.embedder(x).detach()[:,1:,:],m.supervisor(m.embedder(x).detach())[:,:-1,:])
        gl=adv+10.0*torch.sqrt(mom+1e-8)+sup; gl.backward(); opt_g.step()
        # discriminator update
        opt_d.zero_grad(); h_real=m.embedder(x).detach(); h_fake=m.supervisor(m.generator(z)).detach(); dr=m.discriminator(h_real); df=m.discriminator(h_fake)
        dl=bce(dr,torch.ones_like(dr))+bce(df,torch.zeros_like(df)); dl.backward(); opt_d.step()
        hist.append({'stage':'joint','step':s+1,'g_loss':float(gl.detach().cpu()),'d_loss':float(dl.detach().cpu())})
    return m,hist
