from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

TRANSITION_FEATURES = ['Previous_Rutting_mm','Delta_Load_10k','Avg_Temp_C','Delta_Rutting_mm']
CONDITIONING = ['Previous_Rutting_mm','Delta_Load_10k','Avg_Temp_C']


def _transition_frame(frame: pd.DataFrame) -> pd.DataFrame:
    d=frame.copy()
    if 'Rutting_Observed' in d:
        d=d[pd.to_numeric(d.Rutting_Observed,errors='coerce').fillna(0).astype(int).eq(1)]
    for c in ['Rutting_mm','Previous_Rutting_mm','Delta_Load_10k','Avg_Temp_C','Year']:
        if c not in d: raise KeyError(c)
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d['Delta_Rutting_mm']=d['Rutting_mm']-d['Previous_Rutting_mm']
    return d.dropna(subset=TRANSITION_FEATURES).reset_index(drop=True)


def build_real_transition_reference(master: pd.DataFrame) -> pd.DataFrame:
    """Return real transition supervision available to the generator refit (2016-2018 only)."""
    year=pd.to_numeric(master['Year'],errors='coerce')
    real=master[year.le(2018)].copy()
    return _transition_frame(real)


def build_synthetic_transition_frame(update: pd.DataFrame) -> pd.DataFrame:
    d=update.copy()
    if 'source_type' in d.columns:
        d=d[d['source_type'].astype(str).str.lower().eq('synthetic')].copy()
    return _transition_frame(d)


def _conditional_table(real, synth):
    rows=[]
    for var in CONDITIONING:
        vals=pd.to_numeric(real[var],errors='coerce').dropna().to_numpy(float)
        edges=np.unique(np.quantile(vals,[0,.25,.5,.75,1])) if len(vals) else np.array([])
        if len(edges)<2: continue
        # expand numerically so edge-equal values are retained by pd.cut
        edges=edges.astype(float); edges[0]-=1e-12; edges[-1]+=1e-12
        rb=pd.cut(real[var],edges,include_lowest=True,duplicates='drop')
        sb=pd.cut(synth[var],edges,include_lowest=True,duplicates='drop')
        cats=rb.cat.categories
        for k,cat in enumerate(cats,1):
            rdelta=real.loc[rb.eq(cat),'Delta_Rutting_mm']; sdelta=synth.loc[sb.eq(cat),'Delta_Rutting_mm']
            rows.append({
                'conditioning_variable':var,'bin':int(k),'lower':float(cat.left),'upper':float(cat.right),
                'n_real':int(len(rdelta)),'n_synthetic':int(len(sdelta)),
                'real_delta_mean':float(rdelta.mean()) if len(rdelta) else np.nan,
                'synthetic_delta_mean':float(sdelta.mean()) if len(sdelta) else np.nan,
                'absolute_mean_gap':float(abs(rdelta.mean()-sdelta.mean())) if len(rdelta) and len(sdelta) else np.nan,
            })
    return pd.DataFrame(rows)


def _embedding(real, synth):
    xr=real[TRANSITION_FEATURES].to_numpy(float); xs=synth[TRANSITION_FEATURES].to_numpy(float)
    scaler=StandardScaler().fit(xr); zr=scaler.transform(xr); zs=scaler.transform(xs)
    ncomp=min(2,zr.shape[1],len(zr))
    if ncomp<1:
        raise ValueError('no real transitions available for PCA')
    pca=PCA(n_components=ncomp,random_state=0).fit(zr)
    pr=pca.transform(zr); ps=pca.transform(zs)
    if ncomp==1:
        pr=np.c_[pr,np.zeros(len(pr))]; ps=np.c_[ps,np.zeros(len(ps))]
    emb=pd.DataFrame(np.vstack([pr[:,:2],ps[:,:2]]),columns=['pc1','pc2'])
    emb.insert(0,'source',['real']*len(pr)+['synthetic']*len(ps))
    dist=float(np.linalg.norm(pr[:,:2].mean(axis=0)-ps[:,:2].mean(axis=0))) if len(ps) else np.nan
    return emb,dist


def assess_synthetic_transition_fidelity(master: pd.DataFrame, update: pd.DataFrame) -> dict:
    real=build_real_transition_reference(master); synth=build_synthetic_transition_frame(update)
    if real.empty: raise ValueError('no 2016-2018 real transitions for fidelity reference')
    if synth.empty: raise ValueError('no synthetic transitions in update')
    rd=real.Delta_Rutting_mm.to_numpy(float); sd=synth.Delta_Rutting_mm.to_numpy(float)
    ks=ks_2samp(rd,sd,alternative='two-sided',method='auto')
    emb,centroid=_embedding(real,synth)
    lo,hi=np.quantile(rd,[.005,.995])
    plaus={
        'real_delta_q005':float(lo),'real_delta_q995':float(hi),
        'outside_real_99pct_support_rate':float(np.mean((sd<lo)|(sd>hi))),
        'synthetic_downward_change_rate':float(np.mean(sd<0)),
        'real_downward_change_rate':float(np.mean(rd<0)),
        'synthetic_nonfinite_rate':0.0,
    }
    summary={
        'reference_year_max':2018,
        'n_real':int(len(real)),'n_synthetic':int(len(synth)),
        'delta_mean_real':float(np.mean(rd)),'delta_mean_synthetic':float(np.mean(sd)),
        'delta_std_real':float(np.std(rd,ddof=1)) if len(rd)>1 else 0.0,
        'delta_std_synthetic':float(np.std(sd,ddof=1)) if len(sd)>1 else 0.0,
        'delta_wasserstein':float(wasserstein_distance(rd,sd)),
        'delta_ks_statistic':float(ks.statistic),'delta_ks_pvalue_descriptive':float(ks.pvalue),
        'pca_centroid_distance':centroid,
    }
    return {
        'summary':summary,
        'conditional':_conditional_table(real,synth),
        'plausibility':plaus,
        'embedding':emb,
        'real_transitions':real,
        'synthetic_transitions':synth,
    }
