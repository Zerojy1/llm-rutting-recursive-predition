from __future__ import annotations
import numpy as np
from scipy.stats import wasserstein_distance


def _acf1(arr):
    vals=[]
    for x in np.asarray(arr,float):
        x=np.asarray(x).reshape(-1)
        if len(x)<2 or np.nanstd(x[:-1])==0 or np.nanstd(x[1:])==0: continue
        vals.append(np.corrcoef(x[:-1],x[1:])[0,1])
    return float(np.nanmean(vals)) if vals else float('nan')


def _rbf_mmd(x,y,gamma=None):
    x=np.asarray(x,float).reshape(len(x),-1); y=np.asarray(y,float).reshape(len(y),-1)
    z=np.vstack([x,y]);
    if gamma is None:
        # median heuristic, guarded for tiny samples
        d=((z[:,None,:]-z[None,:,:])**2).sum(-1); med=np.median(d[d>0]) if np.any(d>0) else 1.; gamma=1/max(med,1e-9)
    kxx=np.exp(-gamma*((x[:,None,:]-x[None,:,:])**2).sum(-1)).mean()
    kyy=np.exp(-gamma*((y[:,None,:]-y[None,:,:])**2).sum(-1)).mean()
    kxy=np.exp(-gamma*((x[:,None,:]-y[None,:,:])**2).sum(-1)).mean()
    return float(kxx+kyy-2*kxy)


def compare_real_synthetic(real,synthetic):
    r=np.asarray(real,float); s=np.asarray(synthetic,float)
    return {'wasserstein':float(wasserstein_distance(r.reshape(-1),s.reshape(-1))),
            'mmd_rbf':_rbf_mmd(r,s),'acf1_real':_acf1(r),'acf1_synthetic':_acf1(s),
            'mean_real':float(np.nanmean(r)),'mean_synthetic':float(np.nanmean(s)),
            'std_real':float(np.nanstd(r)),'std_synthetic':float(np.nanstd(s))}


def conditional_specificity(base_draws,changed_draws):
    a=np.asarray(base_draws,float); b=np.asarray(changed_draws,float)
    if a.shape!=b.shape: raise ValueError('paired conditional draws must have same shape')
    dif=b-a
    return {'mean_absolute_response':float(np.mean(np.abs(dif))), 'signed_mean_response':float(np.mean(dif)),
            'trajectory_response_l2':float(np.mean(np.linalg.norm(dif.reshape(len(dif),-1),axis=1)))}
