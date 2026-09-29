from __future__ import annotations
import numpy as np
import pandas as pd


def assemble_masked_update_frame(master:pd.DataFrame,generated_states:pd.DataFrame,year:int=2019)->pd.DataFrame:
    """Replace only update-year rutting labels by leakage-safe available states and rebuild memory."""
    key=['STR_name','Observation_Index']
    gs=generated_states[key+['available_rutting_mm','state_source']].copy()
    work=master.merge(gs,on=key,how='left')
    # build an available trajectory: all pre-year observed truth, update-year retained/synthetic states
    work=work.sort_values(['STR_name','Observation_Index'],kind='mergesort').copy()
    out_rows=[]
    for sid,g in work.groupby('STR_name',sort=False):
        prev=np.nan; prev_inc=np.nan; peak=np.nan
        for _,r in g.iterrows():
            yy=int(r.Year)
            if yy<year:
                val=float(r.Rutting_mm) if int(r.get('Rutting_Observed',1))==1 and pd.notna(r.Rutting_mm) else np.nan
                source='historical_real'
            elif yy==year:
                val=float(r.available_rutting_mm) if pd.notna(r.available_rutting_mm) else np.nan
                source=str(r.state_source) if pd.notna(r.state_source) else 'missing'
            else:
                continue
            old_prev=prev; old_peak=peak
            if np.isfinite(val):
                inc=val-old_prev if np.isfinite(old_prev) else np.nan
                peak=val if not np.isfinite(peak) else max(peak,val)
                prev=val
            else: inc=np.nan
            # Original missing-rutting exposures may carry a generated latent state for
            # recursive continuity, but they are never promoted to supervised targets.
            if yy==year and np.isfinite(val) and int(r.get('Rutting_Observed',0))==1:
                rr=r.copy(); rr['Rutting_mm']=val; rr['Rutting_Observed']=1
                rr['Previous_Rutting_mm']=old_prev; rr['Previous_Increment_mm']=prev_inc; rr['Previous_Peak_mm']=old_peak if np.isfinite(old_peak) else old_prev
                rr['source_type']='real' if source=='real' else ('synthetic' if source=='synthetic' else source)
                out_rows.append(rr)
            if np.isfinite(inc): prev_inc=inc
    if not out_rows: return master.iloc[0:0].copy()
    out=pd.DataFrame(out_rows).drop(columns=['available_rutting_mm','state_source'],errors='ignore')
    return out.reset_index(drop=True)
