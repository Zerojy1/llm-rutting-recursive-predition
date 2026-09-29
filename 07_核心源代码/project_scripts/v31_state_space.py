from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

@dataclass
class StateSupport:
    scaler: StandardScaler
    nn: NearestNeighbors
    train_z: np.ndarray
    sparse_threshold: float
    mins: np.ndarray
    maxs: np.ndarray
    padding: np.ndarray


def fit_state_support(train,k=5,sparse_quantile=.75,domain_padding=.10):
    X=np.asarray(train,float)
    if X.ndim!=2 or len(X)<2: raise ValueError('train support array must be 2D with >=2 rows')
    scaler=StandardScaler().fit(X); z=scaler.transform(X)
    kk=min(max(2,k+1),len(X)); nn0=NearestNeighbors(n_neighbors=kk).fit(z); d,_=nn0.kneighbors(z); kth=d[:,-1]
    nn=NearestNeighbors(n_neighbors=min(k,len(X))).fit(z)
    mins=X.min(axis=0); maxs=X.max(axis=0); padding=(maxs-mins)*domain_padding+1e-9
    return StateSupport(scaler,nn,z,float(np.quantile(kth,sparse_quantile)),mins,maxs,padding)


def in_domain_mask(support:StateSupport,points):
    X=np.asarray(points,float)
    if X.ndim==1: X=X.reshape(1,-1)
    return np.all((X>=support.mins-support.padding)&(X<=support.maxs+support.padding),axis=1)


def classify_support(support:StateSupport,points):
    """Backward-compatible training-threshold support labels."""
    X=np.asarray(points,float); in_domain=in_domain_mask(support,X)
    z=support.scaler.transform(X); d,_=support.nn.kneighbors(z); score=d[:,-1]
    return np.where(~in_domain,'ood',np.where(score>support.sparse_threshold,'sparse_in_domain','dense'))


def support_distance(support:StateSupport,points):
    X=np.asarray(points,float)
    if X.ndim==1: X=X.reshape(1,-1)
    z=support.scaler.transform(X); d,_=support.nn.kneighbors(z); return d[:,-1]


def relative_support_quartiles(distances,in_domain):
    """Rank in-domain future contexts by distance to historical support without using labels.

    Q1 contains the closest/highest-support future contexts and Q4 the farthest/lowest-support
    future contexts. True context-OOD points retain the label ``ood``. Rank-based allocation
    avoids degenerate all-sparse labels under chronological distribution shift.
    """
    d=np.asarray(distances,float); mask=np.asarray(in_domain,bool)
    if d.ndim!=1 or mask.shape!=d.shape: raise ValueError('distance/mask shape mismatch')
    out=np.full(len(d),'ood',dtype=object); idx=np.flatnonzero(mask)
    if len(idx)==0: return out
    order=idx[np.argsort(d[idx],kind='stable')]
    names=np.array(['Q1_high_support','Q2','Q3','Q4_low_support'],dtype=object)
    # Equal-count rank bins; no outcome labels enter the thresholding.
    bins=np.minimum((np.arange(len(order))*4)//len(order),3)
    out[order]=names[bins]
    return out


def progression_extrapolation_flags(train_progression,points_progression,eps=1e-12):
    """Report chronological stage extrapolation separately from context OOD."""
    tr=np.asarray(train_progression,float); pt=np.asarray(points_progression,float)
    if tr.ndim!=2 or pt.ndim!=2 or tr.shape[1]!=2 or pt.shape[1]!=2:
        raise ValueError('progression arrays must have shape (n,2): log_cum_load, previous_rutting')
    lo=np.nanmin(tr,axis=0); hi=np.nanmax(tr,axis=0)
    return {
        'load_stage_extrapolation':(pt[:,0]<lo[0]-eps)|(pt[:,0]>hi[0]+eps),
        'rutting_state_extrapolation':(pt[:,1]<lo[1]-eps)|(pt[:,1]>hi[1]+eps),
    }
