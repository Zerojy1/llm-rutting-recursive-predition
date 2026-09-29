from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reasons: tuple[str,...]
    reconstructed: tuple[float,...]


def empirical_increment_bounds(increments,lower_q=.005,upper_q=.995,pad=.05):
    x=np.asarray(increments,dtype=float); x=x[np.isfinite(x)]
    if len(x)<2: raise ValueError('need >=2 finite increments')
    lo=float(np.quantile(x,lower_q)-pad); hi=float(np.quantile(x,upper_q)+pad)
    return lo,hi


def prospective_rutting_bounds(historical_rutting, increment_bounds, horizon:int=6, pad:float=.25):
    """Leakage-safe absolute rutting envelope for prospective generation.

    Future deterioration is allowed to progress beyond the largest historical
    rutting depth. The hard envelope is derived only from historical observed
    states and the predeclared empirical increment bounds, never from holdout
    or future-year rutting labels.
    """
    x=np.asarray(historical_rutting,dtype=float); x=x[np.isfinite(x)]
    if len(x)==0: raise ValueError('need >=1 finite historical rutting value')
    lo_inc,hi_inc=map(float,increment_bounds); h=max(1,int(horizon)); pad=max(0.0,float(pad))
    lower=max(0.0,float(np.min(x)) + h*min(0.0,lo_inc) - pad)
    upper=float(np.max(x)) + h*max(0.0,hi_inc) + pad
    if not np.isfinite(lower) or not np.isfinite(upper) or upper<=lower:
        raise ValueError('invalid prospective rutting bounds')
    return lower,upper


def local_prospective_rutting_bounds(anchor:float, increment_bounds, horizon:int, base_bounds=(0.0,float('inf')), pad:float=1e-6):
    """Condition-specific hard envelope that never uses future truth.

    A retained real anchor in a future/update year may legitimately exceed the
    maximum rutting level observed in the historical generator-training years.
    The local upper envelope therefore expands from the *available anchor* by
    the predeclared empirical positive-increment bound over the requested
    horizon. This prevents a historical-level cap from misclassifying ordinary
    prospective deterioration as out-of-domain.
    """
    a=float(anchor); lo_inc,hi_inc=map(float,increment_bounds); h=max(1,int(horizon))
    blo,bhi=map(float,base_bounds); pad=max(0.0,float(pad))
    if not np.isfinite(a): raise ValueError('anchor must be finite')
    lower=0.0
    if np.isfinite(blo): lower=max(0.0,min(blo,a+h*min(0.0,lo_inc)-pad))
    reachable_upper=a+h*max(0.0,hi_inc)+pad
    upper=max(bhi if np.isfinite(bhi) else reachable_upper,reachable_upper)
    if not np.isfinite(upper) or upper<=lower: raise ValueError('invalid local prospective rutting bounds')
    return lower,upper


def hard_validate_trajectory(increments,anchor:float,increment_bounds,rutting_bounds):
    inc=np.asarray(increments,dtype=float); reasons=[]
    if inc.ndim!=1 or len(inc)==0: reasons.append('invalid_sequence')
    if not np.all(np.isfinite(inc)): reasons.append('non_finite')
    lo,hi=map(float,increment_bounds)
    if np.any(inc<lo) or np.any(inc>hi): reasons.append('increment_out_of_empirical_bounds')
    rec=np.asarray(anchor,dtype=float)+np.cumsum(inc)
    rlo,rhi=map(float,rutting_bounds)
    if np.any(rec<rlo) or np.any(rec>rhi): reasons.append('rutting_out_of_domain')
    return ValidationResult(not reasons,tuple(reasons),tuple(float(v) for v in rec))


def memorization_audit(real_trajectories,synthetic_trajectories,tol=1e-10):
    from sklearn.metrics import pairwise_distances
    R=np.asarray(real_trajectories,dtype=float); S=np.asarray(synthetic_trajectories,dtype=float)
    # Zero successful generations is a valid audit outcome, not a numerical error.
    if S.size==0 or len(S)==0:
        return {'exact_duplicate_rate':np.nan,'dcr':np.asarray([],dtype=float),
                'nn_distance_ratio':np.asarray([],dtype=float),'real_nn_scale':np.nan}
    if R.size==0 or len(R)==0:
        return {'exact_duplicate_rate':np.nan,'dcr':np.full(len(S),np.nan,dtype=float),
                'nn_distance_ratio':np.full(len(S),np.nan,dtype=float),'real_nn_scale':np.nan}
    if R.ndim==1: R=R.reshape(1,-1)
    if S.ndim==1: S=S.reshape(1,-1)
    if R.shape[1]!=S.shape[1]:
        raise ValueError(f'real/synthetic trajectory width mismatch: {R.shape[1]} vs {S.shape[1]}')
    D=pairwise_distances(S,R,metric='euclidean'); dcr=D.min(axis=1)
    exact=(dcr<=tol)
    # NN-distance ratio: synthetic-to-real / typical real-to-real NN
    if len(R)>1:
        RR=pairwise_distances(R,R); np.fill_diagonal(RR,np.inf); scale=float(np.median(RR.min(axis=1)))
    else: scale=np.nan
    ratio=dcr/scale if np.isfinite(scale) and scale>0 else np.full_like(dcr,np.nan)
    return {'exact_duplicate_rate':float(exact.mean()) if len(exact) else np.nan,'dcr':dcr,'nn_distance_ratio':ratio,'real_nn_scale':scale}


def paired_conditional_effect(base,changed):
    A=np.asarray(base,dtype=float); B=np.asarray(changed,dtype=float)
    if A.shape!=B.shape: raise ValueError('paired generation arrays must have same shape')
    diff=B-A
    return {'mean_endpoint_shift':float(np.mean(diff[:,-1])),'mean_absolute_path_shift':float(np.mean(np.abs(diff))),
            'paired_endpoint_sd':float(np.std(diff[:,-1],ddof=1)) if len(diff)>1 else 0.0}


def soft_plausibility_score(increments,local_center,local_scale,disagreement=0.0,nn_distance=0.0):
    inc=np.asarray(increments,float); c=np.asarray(local_center,float); s=np.maximum(np.asarray(local_scale,float),1e-6)
    z=float(np.mean(np.abs((inc-c)/s)))
    return float(np.exp(-z)*np.exp(-max(0,float(disagreement)))*np.exp(-0.1*max(0,float(nn_distance))))
