from __future__ import annotations
import numpy as np
import pandas as pd


def persistence_skill(model_rmse:float,persistence_rmse:float)->float:
    return float(1.0-model_rmse/persistence_rmse) if persistence_rmse>0 else np.nan


def maximum_synthetic_substitution_ratio(table:pd.DataFrame,margin=.10)->float:
    """Contiguous maximum synthetic substitution ratio (MSSR).

    Starting from the lowest evaluated non-zero replacement ratio, every intermediate ratio
    must satisfy the non-inferiority margin. An isolated pass at a higher ratio after an
    earlier failure is not reported as the maximum substitutable fraction.
    """
    t=table[['synthetic_fraction','upper_ci_relative_degradation']].copy()
    t['synthetic_fraction']=pd.to_numeric(t['synthetic_fraction'],errors='coerce')
    t['upper_ci_relative_degradation']=pd.to_numeric(t['upper_ci_relative_degradation'],errors='coerce')
    t=t.dropna().sort_values('synthetic_fraction')
    best=0.0
    for _,r in t.iterrows():
        q=float(r['synthetic_fraction'])
        if q<=0: continue
        if float(r['upper_ci_relative_degradation']) < float(margin):
            best=q
        else:
            break
    return float(best)


def cluster_paired_bootstrap(frame:pd.DataFrame,cluster_col:str,real_col:str,mix_col:str,n_boot=10000,seed=20260814):
    # aggregate within structure first; do not give repeated rows pseudo-independence
    agg=frame.groupby(cluster_col)[[real_col,mix_col]].mean(); diff=(agg[mix_col]-agg[real_col]).to_numpy(float)
    clusters=agg.index.to_numpy(); rng=np.random.default_rng(seed); boots=np.empty(n_boot,float)
    for b in range(n_boot):
        idx=rng.integers(0,len(diff),size=len(diff)); boots[b]=diff[idx].mean()
    return {'n_clusters':int(len(clusters)),'mean_difference':float(diff.mean()),'ci_low':float(np.quantile(boots,.025)),'ci_high':float(np.quantile(boots,.975))}


def noninferiority_relative_rmse(full_real_rmse,mix_rmse):
    return float((mix_rmse-full_real_rmse)/full_real_rmse)


def cluster_relative_rmse_bootstrap(frame:pd.DataFrame,cluster_col:str,y_col:str,real_pred_col:str,mix_pred_col:str,n_boot=10000,seed=20260814):
    """Structure-cluster bootstrap for relative RMSE degradation.

    Each sampled cluster contributes all of its transitions; structures are the inferential
    resampling unit, avoiding pseudo-replication from dense repeated measurements.
    """
    f=frame[[cluster_col,y_col,real_pred_col,mix_pred_col]].dropna().copy(); clusters=f[cluster_col].unique()
    if len(clusters)<2: raise ValueError('need at least 2 clusters')
    groups={c:f[f[cluster_col].eq(c)] for c in clusters}
    def stat(sampled):
        ys=[]; rr=[]; mm=[]
        for c in sampled:
            g=groups[c]; ys.extend(g[y_col].to_numpy(float)); rr.extend(g[real_pred_col].to_numpy(float)); mm.extend(g[mix_pred_col].to_numpy(float))
        y=np.asarray(ys); r=np.asarray(rr); m=np.asarray(mm); rrmse=np.sqrt(np.mean((r-y)**2)); mrmse=np.sqrt(np.mean((m-y)**2)); return (mrmse-rrmse)/rrmse
    point=stat(clusters); rng=np.random.default_rng(seed); boots=np.array([stat(rng.choice(clusters,size=len(clusters),replace=True)) for _ in range(n_boot)])
    return {'n_clusters':int(len(clusters)),'mean_relative_degradation':float(point),'ci_low':float(np.quantile(boots,.025)),'ci_high':float(np.quantile(boots,.975))}


def hierarchical_cluster_relative_bootstrap(frames:list[pd.DataFrame],cluster_col='STR_name',y_col='y_true',real_pred_col='real_pred',mix_pred_col='mix_pred',n_boot=10000,seed=20260814):
    """Two-level bootstrap over paired runs and pavement structures.

    Each frame is one paired allocation×seed comparison containing identical evaluation
    targets for full-real and mixed training. Bootstrap samples paired runs and, within each
    selected run, samples pavement structures with replacement.
    """
    if not frames: raise ValueError('no paired frames')
    prepared=[]; all_clusters=set()
    for f in frames:
        g=f[[cluster_col,y_col,real_pred_col,mix_pred_col]].dropna().copy(); groups={c:g[g[cluster_col].eq(c)] for c in g[cluster_col].unique()}; prepared.append(groups); all_clusters.update(groups)
    if len(all_clusters)<2: raise ValueError('need at least 2 clusters')
    def one(run_ids, rng=None, resample_clusters=False):
        rel=[]
        for rid in run_ids:
            groups=prepared[int(rid)]; cls=np.array(list(groups),dtype=object)
            sampled=(rng.choice(cls,size=len(cls),replace=True) if resample_clusters else cls)
            ys=[]; rr=[]; mm=[]
            for c in sampled:
                x=groups[c]; ys.extend(x[y_col]); rr.extend(x[real_pred_col]); mm.extend(x[mix_pred_col])
            y=np.asarray(ys,float); r=np.asarray(rr,float); m=np.asarray(mm,float); br=np.sqrt(np.mean((r-y)**2)); bm=np.sqrt(np.mean((m-y)**2)); rel.append((bm-br)/br)
        return float(np.mean(rel))
    point=one(range(len(prepared))); rng=np.random.default_rng(seed); boots=np.empty(n_boot)
    for b in range(n_boot):
        runs=rng.integers(0,len(prepared),size=len(prepared)); boots[b]=one(runs,rng,True)
    return {'n_runs':len(prepared),'n_clusters':len(all_clusters),'mean_relative_degradation':point,'ci_low':float(np.quantile(boots,.025)),'ci_high':float(np.quantile(boots,.975))}
