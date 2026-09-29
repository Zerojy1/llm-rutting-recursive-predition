from __future__ import annotations
import numpy as np
import pandas as pd


def _stratified_take(frame:pd.DataFrame,n:int,seed:int)->pd.DataFrame:
    if n<0 or n>len(frame): raise ValueError('invalid requested sample size')
    if n==0: return frame.iloc[0:0].copy()
    rng=np.random.default_rng(seed)
    work=frame.copy(); work['_load_bin']=pd.qcut(pd.to_numeric(work.get('Cum_Load_10k',pd.Series(np.arange(len(work)),index=work.index)),errors='coerce').rank(method='first'),q=min(4,max(1,n)),labels=False,duplicates='drop')
    strata=work['STR_name'].astype(str)+'|'+work['_load_bin'].astype(str)
    # balanced proportional quotas with deterministic remainder
    counts=strata.value_counts(); raw=counts/counts.sum()*n; quotas=np.floor(raw).astype(int); rem=n-int(quotas.sum())
    for key in (raw-quotas).sort_values(ascending=False).index[:rem]: quotas.loc[key]+=1
    chosen=[]
    for key,q in quotas.items():
        idx=np.flatnonzero(strata.to_numpy()==key)
        if q: chosen.extend(rng.choice(idx,size=min(q,len(idx)),replace=False).tolist())
    if len(chosen)<n:
        remaining=np.setdiff1d(np.arange(len(work)),np.array(chosen,dtype=int)); chosen.extend(rng.choice(remaining,size=n-len(chosen),replace=False).tolist())
    return work.iloc[np.array(chosen[:n],dtype=int)].drop(columns=['_load_bin']).copy()


def build_equal_budget_update(real_pool:pd.DataFrame,synthetic_pool:pd.DataFrame,total_budget:int,synthetic_fraction:float,allocation_seed:int,control:str='synthetic')->pd.DataFrame:
    q=float(synthetic_fraction); n_syn=int(round(total_budget*q)); n_real=total_budget-n_syn
    if n_real>len(real_pool): raise ValueError('not enough real rows')
    real=_stratified_take(real_pool,n_real,allocation_seed); real['source_type']='real'
    if n_syn==0: return real.reset_index(drop=True)
    rng=np.random.default_rng(allocation_seed+991)
    if control=='real_bootstrap':
        if len(real)==0: raise ValueError('real-bootstrap undefined with zero retained real data')
        rep=real.iloc[rng.choice(len(real),size=n_syn,replace=True)].copy(); rep['source_type']='real_repeat'
        return pd.concat([real,rep],ignore_index=True)
    if control!='synthetic': raise ValueError(f'unknown control {control}')
    if n_syn>len(synthetic_pool): raise ValueError('not enough synthetic rows')
    syn=synthetic_pool.iloc[rng.choice(len(synthetic_pool),size=n_syn,replace=False)].copy(); syn['source_type']='synthetic'
    return pd.concat([real,syn],ignore_index=True)


def repeated_stratified_allocations(frame:pd.DataFrame,n_keep:int,n_allocations:int=5,base_seed:int=20260814):
    out=[]
    for i in range(n_allocations): out.append(_stratified_take(frame,n_keep,base_seed+i).index.to_numpy())
    return out
