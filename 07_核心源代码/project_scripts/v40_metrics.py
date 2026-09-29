from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from v31_statistics import hierarchical_cluster_relative_bootstrap

RUN_COLS = ['model','method','synthetic_fraction','allocation','seed']
PAIR_KEYS = ['STR_name','origin_observation_index','target_observation_index','requested_horizon','step_h']


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(np.asarray(y,float), np.asarray(p,float))))


def _r2(y, p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    return float(r2_score(y,p)) if len(y)>=2 and np.std(y)>0 else np.nan


def summarize_prediction_rows(predictions: pd.DataFrame) -> dict[str,pd.DataFrame]:
    p = predictions.copy()
    required={'requested_horizon','step_h','STR_name','origin_observation_index','y_true','y_pred','persistence_pred'}
    miss=required-set(p.columns)
    if miss: raise KeyError(f'missing prediction columns: {sorted(miss)}')
    base=[c for c in RUN_COLS if c in p.columns]
    horizon_rows=[]
    for keys,g in p.groupby(base+['requested_horizon'], dropna=False, sort=True):
        if not isinstance(keys,tuple): keys=(keys,)
        rec=dict(zip(base+['requested_horizon'],keys))
        rec.update({
            'trajectory_rmse':_rmse(g.y_true,g.y_pred),
            'trajectory_mae':float(mean_absolute_error(g.y_true,g.y_pred)),
            'trajectory_r2':_r2(g.y_true,g.y_pred),
            'persistence_rmse':_rmse(g.y_true,g.persistence_pred),
            'n_rows':int(len(g)),
            'n_origins':int(g[['STR_name','origin_observation_index']].drop_duplicates().shape[0]),
        })
        rec['persistence_skill']=float(1-rec['trajectory_rmse']/rec['persistence_rmse']) if rec['persistence_rmse']>0 else np.nan
        horizon_rows.append(rec)
    step_rows=[]
    for keys,g in p.groupby(base+['requested_horizon','step_h'], dropna=False, sort=True):
        if not isinstance(keys,tuple): keys=(keys,)
        rec=dict(zip(base+['requested_horizon','step_h'],keys))
        rec.update({
            'rmse':_rmse(g.y_true,g.y_pred),
            'mae':float(mean_absolute_error(g.y_true,g.y_pred)),
            'r2':_r2(g.y_true,g.y_pred),
            'persistence_rmse':_rmse(g.y_true,g.persistence_pred),
            'n':int(len(g)),
        })
        rec['persistence_skill']=float(1-rec['rmse']/rec['persistence_rmse']) if rec['persistence_rmse']>0 else np.nan
        step_rows.append(rec)
    return {'horizon_metrics':pd.DataFrame(horizon_rows), 'step_metrics':pd.DataFrame(step_rows)}


def compute_error_growth(step_metrics: pd.DataFrame) -> pd.DataFrame:
    s=step_metrics.copy()
    group_cols=[c for c in RUN_COLS+['requested_horizon'] if c in s.columns]
    if not group_cols: group_cols=['requested_horizon'] if 'requested_horizon' in s.columns else []
    out=[]
    groups=s.groupby(group_cols,dropna=False,sort=False) if group_cols else [((),s)]
    for _,g in groups:
        g=g.copy(); one=g.loc[pd.to_numeric(g.step_h).eq(1),'rmse']
        base=float(one.iloc[0]) if len(one) else np.nan
        g['error_growth']=pd.to_numeric(g.rmse,errors='coerce')/base if np.isfinite(base) and base!=0 else np.nan
        out.append(g)
    return pd.concat(out,ignore_index=True) if out else s.assign(error_growth=np.nan)


def compute_persistence_skill(metrics: pd.DataFrame) -> pd.DataFrame:
    m=metrics.copy(); rmse=pd.to_numeric(m['rmse'],errors='coerce'); prmse=pd.to_numeric(m['persistence_rmse'],errors='coerce')
    m['persistence_skill']=np.where(prmse>0,1-rmse/prmse,np.nan)
    return m


def paired_noninferiority_by_horizon(full_real_rows: pd.DataFrame, mix_rows: pd.DataFrame,
                                      margin=.10, n_boot=10000, seed=20260814) -> pd.DataFrame:
    real=full_real_rows.copy(); mix=mix_rows.copy()
    if 'seed' not in real or 'seed' not in mix: raise KeyError('seed required for paired comparison')
    if 'allocation' not in mix: raise KeyError('allocation required for paired comparison')
    q_values=sorted(pd.to_numeric(mix.get('synthetic_fraction',pd.Series([np.nan]*len(mix))),errors='coerce').dropna().unique())
    if not q_values: q_values=[np.nan]
    horizons=sorted(pd.to_numeric(mix.requested_horizon,errors='coerce').dropna().astype(int).unique())
    records=[]
    for q in q_values:
        mq=mix if not np.isfinite(q) else mix[pd.to_numeric(mix.synthetic_fraction,errors='coerce').eq(q)]
        for h in horizons:
            frames=[]
            hmix=mq[pd.to_numeric(mq.requested_horizon).eq(h)]
            for (allocation,s),gm in hmix.groupby(['allocation','seed'],sort=True):
                gr=real[(pd.to_numeric(real.seed).eq(int(s))) & (pd.to_numeric(real.requested_horizon).eq(h))]
                if gr.empty: continue
                keys=[k for k in PAIR_KEYS if k in gm.columns and k in gr.columns]
                mm=gr[keys+['y_true','y_pred']].merge(
                    gm[keys+['y_true','y_pred']],on=keys,how='inner',suffixes=('_real','_mix'),validate='one_to_one'
                )
                if mm.empty: continue
                if not np.allclose(mm.y_true_real,mm.y_true_mix,atol=1e-9,rtol=0):
                    raise RuntimeError('paired target mismatch')
                frames.append(pd.DataFrame({
                    'STR_name':mm.STR_name.astype(str),
                    'y_true':mm.y_true_real.astype(float),
                    'real_pred':mm.y_pred_real.astype(float),
                    'mix_pred':mm.y_pred_mix.astype(float),
                }))
            if not frames: continue
            boot=hierarchical_cluster_relative_bootstrap(frames,n_boot=int(n_boot),seed=int(seed)+int(h)*100)
            records.append({
                'requested_horizon':int(h),
                'synthetic_fraction':float(q) if np.isfinite(q) else np.nan,
                'margin':float(margin),
                'n_paired_runs':int(boot['n_runs']),
                'n_clusters':int(boot['n_clusters']),
                'mean_relative_degradation':float(boot['mean_relative_degradation']),
                'lower_ci_relative_degradation':float(boot['ci_low']),
                'upper_ci_relative_degradation':float(boot['ci_high']),
                'noninferior':bool(float(boot['ci_high']) < float(margin)),
            })
    return pd.DataFrame(records)


def contiguous_mssr_by_horizon(noninferiority: pd.DataFrame, margins=(.10,.05)) -> pd.DataFrame:
    t=noninferiority.copy()
    group_cols=[c for c in ['model','method'] if c in t.columns]
    horizons=sorted(pd.to_numeric(t.requested_horizon,errors='coerce').dropna().astype(int).unique())
    out=[]
    base_groups=[((),t)] if not group_cols else list(t.groupby(group_cols,dropna=False,sort=False))
    for gkey,gdf in base_groups:
        if group_cols and not isinstance(gkey,tuple): gkey=(gkey,)
        gd=dict(zip(group_cols,gkey if group_cols else ()))
        for h in horizons:
            gh=gdf[pd.to_numeric(gdf.requested_horizon).eq(h)].copy()
            for margin in margins:
                qtab=(gh.groupby('synthetic_fraction',as_index=False)['upper_ci_relative_degradation'].max()
                      .sort_values('synthetic_fraction'))
                best=0.0
                for _,r in qtab.iterrows():
                    q=float(r.synthetic_fraction)
                    if q<=0: continue
                    if float(r.upper_ci_relative_degradation) < float(margin): best=q
                    else: break
                out.append({**gd,'requested_horizon':int(h),'margin':float(margin),'mssr':float(best),
                            'is_primary_horizon':bool(int(h)==6)})
    return pd.DataFrame(out)


def horizon_degradation_table(horizon_metrics: pd.DataFrame) -> pd.DataFrame:
    h=horizon_metrics.copy()
    if 'trajectory_rmse' not in h: raise KeyError('trajectory_rmse')
    base_cols=[c for c in RUN_COLS if c in h.columns]
    rows=[]
    groups=h.groupby(base_cols,dropna=False,sort=False) if base_cols else [((),h)]
    for _,g in groups:
        g=g.sort_values('requested_horizon').copy()
        h1=g.loc[pd.to_numeric(g.requested_horizon).eq(1),'trajectory_rmse']
        r1=float(h1.iloc[0]) if len(h1) else np.nan
        g['horizon_rmse_growth_vs_h1']=g['trajectory_rmse']/r1 if np.isfinite(r1) and r1!=0 else np.nan
        rows.append(g)
    return pd.concat(rows,ignore_index=True) if rows else h
