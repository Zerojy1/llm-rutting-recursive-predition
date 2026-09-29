from __future__ import annotations
import numpy as np


def observation_mask_for_gap(n:int,gap:int|None):
    m=np.zeros(n,dtype=bool); m[0]=True
    if gap is None: return m
    if gap<1: raise ValueError('gap must be >=1 or None')
    m[::gap]=True; return m


def recursive_rollout(predict_step,features,observed, gap:int|None):
    observed=np.asarray(observed,float); mask=observation_mask_for_gap(len(observed),gap); pred=np.full(len(observed),np.nan); state=observed[0]; pred[0]=state
    for t in range(1,len(observed)):
        state=float(predict_step(features[t],state)); pred[t]=state
        if mask[t] and np.isfinite(observed[t]): state=float(observed[t])
    return pred,mask
