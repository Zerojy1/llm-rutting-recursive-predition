from __future__ import annotations
import json
import numpy as np
import pandas as pd
from v31_master_data import STRUCTURE_COLUMNS

DYNAMIC_COLS=['Cum_Load_10k','Delta_Load_10k','Avg_Temp_C','Delta_Temp_C','Temp_MA']

def _json_number_or_none(value):
    try:
        x=float(value)
        return x if np.isfinite(x) else None
    except (TypeError,ValueError):
        return None


def build_generator_windows(panel:pd.DataFrame,history_steps:int=4,future_steps:int=6,allowed_target_year_max:int=2018,allowed_target_year_min:int|None=None)->pd.DataFrame:
    rows=[]
    for str_name,sub in panel.sort_values(['STR_name','Observation_Index']).groupby('STR_name',sort=False):
        sub=sub.reset_index(drop=True)
        for origin in range(history_steps-1,len(sub)-future_steps):
            hist=sub.iloc[origin-history_steps+1:origin+1]
            fut=sub.iloc[origin+1:origin+future_steps+1]
            # targets require observed rutting; histories can contain exposure rows without target observations
            if fut['Rutting_Observed'].min()!=1: continue
            target_year_min=int(fut['Year'].min()); target_year_max=int(fut['Year'].max())
            if target_year_max>allowed_target_year_max: continue
            if allowed_target_year_min is not None and target_year_min<allowed_target_year_min: continue
            # need at least latest observed state for anchor
            anchor_candidates=hist.loc[hist['Rutting_Observed'].eq(1),'Rutting_mm']
            if anchor_candidates.empty: continue
            anchor=float(anchor_candidates.iloc[-1])
            y=fut['Rutting_mm'].to_numpy(float); inc=np.diff(np.r_[anchor,y])
            record={'window_id':f'{str_name}_O{int(sub.loc[origin,"Observation_Index"])}_F{future_steps}', 'STR_name':str_name,
                    'origin_year':int(sub.loc[origin,'Year']),'target_year_min':target_year_min,'target_year_max':target_year_max,
                    'anchor_rutting_mm':anchor}
            for c in STRUCTURE_COLUMNS: record[c]=float(sub.loc[origin,c])
            for h in range(history_steps):
                r=hist.iloc[h]
                for c in DYNAMIC_COLS: record[f'h{h+1}_{c}']=float(r[c]) if pd.notna(r[c]) else np.nan
                record[f'h{h+1}_Rutting_mm']=float(r.Rutting_mm) if r.Rutting_Observed else np.nan
            for f in range(future_steps):
                r=fut.iloc[f]
                record[f'f{f+1}_Cum_Load_10k']=float(r.Cum_Load_10k)
                record[f'f{f+1}_Delta_Load_10k']=float(r.Delta_Load_10k)
                record[f'f{f+1}_Avg_Temp_C']=float(r.Avg_Temp_C)
                record[f'f{f+1}_Delta_Temp_C']=float(r.Delta_Temp_C)
                record[f'target_rutting_f{f+1}']=float(y[f])
                record[f'target_increment_f{f+1}']=float(inc[f])
            rows.append(record)
    return pd.DataFrame(rows)


def apply_observation_mask(sequence:pd.DataFrame, retained_mask, synthetic_values:dict[int,float], value_col:str='Rutting_mm')->pd.DataFrame:
    retained=np.asarray(retained_mask,dtype=bool)
    if len(retained)!=len(sequence): raise ValueError('retained mask length mismatch')
    out=sequence.copy(); available=[]; sources=[]
    last=np.nan
    for pos,(_,row) in enumerate(out.iterrows()):
        true=row[value_col]
        if retained[pos] and pd.notna(true):
            last=float(true); source='real'
        elif pos in synthetic_values:
            last=float(synthetic_values[pos]); source='synthetic'
        else:
            source='carried' if np.isfinite(last) else 'missing'
        available.append(last); sources.append(source)
    out['available_state']=available; out['state_source']=sources
    return out


def window_to_qwen_record(row:pd.Series,history_steps:int=4,future_steps:int=6)->dict:
    structure={c:float(row[c]) for c in STRUCTURE_COLUMNS}
    history=[]
    for h in range(1,history_steps+1):
        history.append({
            'cum_load_10k':_json_number_or_none(row.get(f'h{h}_Cum_Load_10k')), 'delta_load_10k':_json_number_or_none(row.get(f'h{h}_Delta_Load_10k')),
            'temperature_c':_json_number_or_none(row.get(f'h{h}_Avg_Temp_C')),'delta_temperature_c':_json_number_or_none(row.get(f'h{h}_Delta_Temp_C')),
            'temp_ma_c':_json_number_or_none(row.get(f'h{h}_Temp_MA')),'rutting_mm':_json_number_or_none(row.get(f'h{h}_Rutting_mm'))})
    future=[]
    for f in range(1,future_steps+1):
        future.append({'cum_load_10k':row[f'f{f}_Cum_Load_10k'],'delta_load_10k':row[f'f{f}_Delta_Load_10k'],
                       'temperature_c':row[f'f{f}_Avg_Temp_C'],'delta_temperature_c':row[f'f{f}_Delta_Temp_C']})
    increments=[round(float(row[f'target_increment_f{f}']),6) for f in range(1,future_steps+1)]
    system='You are a pavement deterioration trajectory generator. Return strict JSON only. Generate observed rutting increments; local negative increments are allowed when plausible.'
    user=json.dumps({'structure_cm':structure,'history':history,'future_scenario':future,'output':'six observed rutting increments in mm'},ensure_ascii=False)
    assistant=json.dumps({'rut_increment_mm':increments},ensure_ascii=False)
    return {'messages':[{'role':'system','content':system},{'role':'user','content':user},{'role':'assistant','content':assistant}],
            'metadata':{'window_id':row['window_id'],'STR_name':row['STR_name'],'origin_year':int(row['origin_year']), 'target_year_min':int(row['target_year_min']),'target_year_max':int(row['target_year_max'])}}
