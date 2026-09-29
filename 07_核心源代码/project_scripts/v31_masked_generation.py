from __future__ import annotations
from collections.abc import Callable
import math
import numpy as np
import pandas as pd
from v31_master_data import STRUCTURE_COLUMNS

DYN=['Cum_Load_10k','Delta_Load_10k','Avg_Temp_C','Delta_Temp_C','Temp_MA']


def generate_masked_update(panel:pd.DataFrame, retained_observation_indices:set, generator_fn:Callable[[dict],list[float]], year:int=2019, history_steps:int=4, max_horizon:int=6)->pd.DataFrame:
    """Sequentially fill withheld update-year rutting states without ever exposing their truth.

    The callback sees only fixed structure, exposure covariates, retained real rutting states,
    and previously generated states. Consecutive withheld rows may be generated in one block;
    a retained real observation re-anchors the state immediately.
    """
    def kept(sid,oi): return (sid,oi) in retained_observation_indices or oi in retained_observation_indices
    results=[]
    for sid,g0 in panel.groupby('STR_name',sort=False):
        g=g0.sort_values('Observation_Index',kind='mergesort').reset_index(drop=True).copy()
        available={}
        source={}
        # all pre-year observed states are legitimate history
        for i,r in g[g.Year.lt(year)].iterrows():
            if int(r.get('Rutting_Observed',1))==1 and pd.notna(r.Rutting_mm):
                available[int(r.Observation_Index)]=float(r.Rutting_mm); source[int(r.Observation_Index)]='historical_real'
        year_positions=list(g.index[g.Year.eq(year)])
        p=0
        while p < len(year_positions):
            idx=year_positions[p]; r=g.loc[idx]; oi=int(r.Observation_Index)
            if kept(sid,oi) and int(r.get('Rutting_Observed',1))==1 and pd.notna(r.Rutting_mm):
                available[oi]=float(r.Rutting_mm); source[oi]='real'; p+=1; continue
            # consecutive withheld block until retained anchor or max_horizon
            block=[]
            q=p
            while q<len(year_positions) and len(block)<max_horizon:
                j=year_positions[q]; rr=g.loc[j]; oij=int(rr.Observation_Index)
                if q>p and kept(sid,oij) and int(rr.get('Rutting_Observed',1))==1 and pd.notna(rr.Rutting_mm): break
                if kept(sid,oij): break
                block.append(j); q+=1
            # anchor = most recent available state, never withheld truth
            prior_ois=[k for k in available if k<oi]
            if not prior_ois: raise ValueError(f'{sid} has no available state before O{oi}')
            anchor_oi=max(prior_ois); anchor=float(available[anchor_oi])
            # history uses previous exposure rows but only available rutting states
            prev_idx=[j for j in range(max(0,idx-history_steps),idx)]
            history=[]
            for j in prev_idx:
                rr=g.loc[j]; oij=int(rr.Observation_Index)
                h={
                    'observation_index':oij,'cum_load_10k':float(rr.Cum_Load_10k),
                    'delta_load_10k':float(rr.Delta_Load_10k),'temperature_c':float(rr.Avg_Temp_C),
                    'delta_temperature_c':float(rr.Delta_Temp_C),'temp_ma_c':float(rr.Temp_MA),
                    'rutting_mm':available.get(oij,None),'state_source':source.get(oij,'unobserved')}
                history.append(h)
            future=[]
            for j in block:
                rr=g.loc[j]
                future.append({'observation_index':int(rr.Observation_Index),'cum_load_10k':float(rr.Cum_Load_10k),
                               'delta_load_10k':float(rr.Delta_Load_10k),'temperature_c':float(rr.Avg_Temp_C),
                               'delta_temperature_c':float(rr.Delta_Temp_C),'temp_ma_c':float(rr.Temp_MA)})
            cond={'STR_name':sid,'structure_cm':{c:float(r[c]) for c in STRUCTURE_COLUMNS},'anchor_rutting_mm':anchor,
                  'anchor_observation_index':anchor_oi,'history':history,'future_scenario':future}
            inc=list(generator_fn(cond))
            if len(inc)<len(block): raise ValueError('generator returned too few increments')
            state=anchor
            for j,d in zip(block,inc):
                state=float(state)+float(d); oij=int(g.loc[j,'Observation_Index']); available[oij]=state
                is_original_target=int(g.loc[j].get('Rutting_Observed',1))==1 and pd.notna(g.loc[j].get('Rutting_mm',np.nan))
                source[oij]='synthetic' if is_original_target else 'synthetic_latent'
            p += len(block)
        for idx in year_positions:
            r=g.loc[idx]; oi=int(r.Observation_Index)
            results.append({'STR_name':sid,'Observation_Index':oi,'Year':year,'true_rutting_mm':float(r.Rutting_mm) if pd.notna(r.Rutting_mm) else float('nan'),
                            'available_rutting_mm':available.get(oi,float('nan')),'state_source':source.get(oi,'missing')})
    return pd.DataFrame(results).sort_values(['STR_name','Observation_Index']).reset_index(drop=True)
