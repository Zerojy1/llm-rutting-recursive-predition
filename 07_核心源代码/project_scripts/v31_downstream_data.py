from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from v31_master_data import STRUCTURE_COLUMNS

ENV_COLS=['Cum_Load_10k','Delta_Load_10k','Avg_Temp_C','Delta_Temp_C','Temp_MA3_C']
FEATURE_COLS=ENV_COLS+STRUCTURE_COLUMNS


def build_real_downstream_frame(master: pd.DataFrame) -> pd.DataFrame:
    """Convert V3.1 master panel to the naming contract of the first-paper TDR model.

    Only observed rutting targets are retained as supervised rows. Exposure history remains
    preserved in the master panel and is used upstream when generator windows are built.
    """
    # Keep every exposure row so load/temperature history remains complete even when
    # a rutting target was not measured. Supervised targets are selected later.
    df=master.copy()
    rename={
        'Load_Date':'Loading_Date','Observation_Index':'Observation_Order',
        'Temp_MA':'Temp_MA3_C','Rutting_mm':'Rutting_Depth_mm',
        'Previous_Rutting_mm':'Prev_Rutting_mm','Previous_Peak_mm':'Prev_Peak_Rutting_mm',
    }
    df=df.rename(columns=rename)
    if 'STR_num' not in df:
        df['STR_num']=df['STR_name'].astype(str).str.extract(r'(\d+)')[0].astype(int)
    if 'Struct_Class' not in df:
        # descriptive grouping only; no model target is derived from this field
        df['Struct_Class']='Unspecified'
    for col in FEATURE_COLS+['Rutting_Depth_mm','Prev_Rutting_mm','Prev_Peak_Rutting_mm']:
        if col not in df:
            raise KeyError(f'missing downstream field: {col}')
        df[col]=pd.to_numeric(df[col],errors='coerce')
    # first observed point has no true previous-state anchor; keep it in frame but callers may filter.
    return df.sort_values(['STR_num','Observation_Order'],kind='mergesort').reset_index(drop=True)


def split_downstream_frame(df: pd.DataFrame) -> dict[str,pd.DataFrame]:
    year=pd.to_numeric(df['Year'],errors='coerce').astype(int)
    return {
        'history': df.loc[year<=2018].copy(),
        'update2019': df.loc[year==2019].copy(),
        'dev2020': df.loc[year==2020].copy(),
        'eval2021': df.loc[year==2021].copy(),
    }


def build_sequence_arrays_simple(df: pd.DataFrame, seq_len: int=6):
    """Leakage-safe observed-target sequences for GRU/TDR downstream learners.

    The current row supplies the interval load/temperature covariates and target rutting;
    previous rutting state must already have been computed from earlier observed states.
    """
    xs=[]; ys=[]; prev=[]; peak=[]; metas=[]
    for sid,g in df.groupby('STR_name',sort=False):
        g=g.sort_values(['Observation_Order'],kind='mergesort').reset_index(drop=True)
        feat=g[FEATURE_COLS].to_numpy(dtype=np.float32)
        pad=np.vstack([np.repeat(feat[[0]],seq_len-1,axis=0),feat])
        for i,row in g.iterrows():
            observed=int(pd.to_numeric(pd.Series([row.get('Rutting_Observed',1)]),errors='coerce').fillna(0).iloc[0])==1
            if (not observed) or (not np.isfinite(row['Rutting_Depth_mm'])) or (not np.isfinite(row['Prev_Rutting_mm'])):
                continue
            xs.append(pad[i:i+seq_len]); ys.append(float(row['Rutting_Depth_mm']))
            prev.append(float(row['Prev_Rutting_mm'])); peak.append(float(row['Prev_Peak_Rutting_mm']))
            metas.append(row[['Year','STR_name','Observation_Order','Cycle_ID','Cum_Load_10k']].to_dict())
    return (np.asarray(xs,np.float32),np.asarray(ys,np.float32),np.asarray(prev,np.float32).reshape(-1,1),
            np.asarray(peak,np.float32).reshape(-1,1),pd.DataFrame(metas))
