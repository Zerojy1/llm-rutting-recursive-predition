from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

STRUCTURE_COLUMNS = [
    'Surf_AC13_I_SBS','Surf_AC13_II_SBS','Surf_SMA13','Surf_PAC13',
    'Mid_AC20_AH30','Mid_AC20_SBS','Mid_AC20_AH50',
    'Bot_AC25_AH30','Bot_AC25_AH50','Bot_AC25_AH70','Bot_AC25_Re_AH70','Bot_AC10_SBS',
    'Base_CBG_A','Base_CBG_B','Base_CS','Base_CC','Base_LCC','Base_GA'
]


def structure_table()->pd.DataFrame:
    names=[f'STR{i}' for i in range(1,20)]
    d=pd.DataFrame(0.0,index=names,columns=STRUCTURE_COLUMNS)
    rules={
    'STR1':(['Surf_AC13_I_SBS','Mid_AC20_AH30','Base_CBG_A','Base_CS'],[4,8,40,40]),
    'STR2':(['Surf_AC13_I_SBS','Mid_AC20_AH30','Base_CBG_A','Base_CS'],[4,8,40,20]),
    'STR3':(['Surf_AC13_I_SBS','Mid_AC20_AH30','Base_CBG_A','Base_GA'],[4,8,40,20]),
    'STR4':(['Surf_AC13_II_SBS','Mid_AC20_AH30','Bot_AC10_SBS','Base_LCC','Base_CBG_A','Base_CS'],[4,6,2,24,20,20]),
    'STR5':(['Surf_AC13_II_SBS','Mid_AC20_SBS','Bot_AC10_SBS','Base_CC','Base_CBG_A','Base_CS'],[4,6,2,24,20,20]),
    'STR6':(['Surf_AC13_II_SBS','Bot_AC25_AH30','Bot_AC10_SBS','Base_CBG_A','Base_CS'],[4,10,2,38,20]),
    'STR7':(['Surf_AC13_II_SBS','Mid_AC20_SBS','Bot_AC25_AH70','Base_CBG_A','Base_CS'],[4,6,8,38,20]),
    'STR8':(['Surf_AC13_II_SBS','Mid_AC20_SBS','Bot_AC25_AH70','Base_CBG_B','Base_CS'],[4,6,8,38,20]),
    'STR9':(['Surf_PAC13','Mid_AC20_SBS','Bot_AC25_AH70','Base_CBG_B','Base_CS'],[4,6,8,38,20]),
    'STR10':(['Surf_AC13_I_SBS','Mid_AC20_SBS','Bot_AC25_AH70','Bot_AC10_SBS','Base_GA','Base_CBG_B','Base_CS'],[4,6,16,2,20,20,20]),
    'STR12':(['Surf_AC13_I_SBS','Mid_AC20_SBS','Bot_AC25_AH70','Base_GA','Base_CBG_B','Base_CS'],[4,8,12,20,20,20]),
    'STR11':(['Surf_AC13_I_SBS','Mid_AC20_SBS','Bot_AC25_AH70','Base_CBG_A','Base_CS'],[4,6,18,40,20]),
    'STR13':(['Surf_AC13_I_SBS','Mid_AC20_SBS','Bot_AC25_AH70','Base_CBG_A','Base_CS'],[4,8,12,40,20]),
    'STR14':(['Surf_AC13_I_SBS','Mid_AC20_SBS','Bot_AC25_Re_AH70','Base_CBG_B','Base_CS'],[4,8,12,40,20]),
    'STR15':(['Surf_AC13_I_SBS','Mid_AC20_AH50','Bot_AC25_AH50','Base_CBG_A','Base_GA'],[4,8,24,20,44]),
    'STR16':(['Surf_SMA13','Mid_AC20_SBS','Bot_AC25_AH70','Base_CBG_A','Base_CS'],[4,8,24,20,20]),
    'STR17':(['Surf_SMA13','Mid_AC20_AH30','Bot_AC25_AH30','Base_CBG_A','Base_CS'],[4,8,24,20,20]),
    'STR18':(['Surf_SMA13','Mid_AC20_AH50','Bot_AC25_AH50','Bot_AC10_SBS','Base_GA'],[4,8,36,4,48]),
    'STR19':(['Surf_SMA13','Mid_AC20_AH50','Bot_AC25_AH30','Base_CBG_B'],[4,8,36,20]),
    }
    for k,(cols,vals) in rules.items(): d.loc[k,cols]=vals
    return d.rename_axis('STR_name').reset_index()


def _normalize_wide_columns(raw:pd.DataFrame)->pd.DataFrame:
    aliases={'Unnamed: 0':'Year','Unnamed: 1':'Load_Date','Unnamed: 2':'Cycle_ID','Unnamed: 3':'Cum_Load_10k','Unnamed: 4':'Avg_Temp_C'}
    out=raw.rename(columns=aliases).copy()
    if 'Year' not in out.columns:
        date_col=next((c for c in out.columns if str(c).lower() in {'load_date','date','loading date','加载日期','日期'}),None)
        if date_col:
            out['Year']=pd.to_datetime(out[date_col],errors='coerce').dt.year
        else:
            raise ValueError('Year column missing; preserve Year/Date in raw import before building V3.1 panel.')
    year_text=out['Year'].astype('string').str.extract(r'(\d{4})',expand=False)
    out['Year']=pd.to_numeric(year_text,errors='coerce').ffill().astype('Int64')
    if out['Year'].isna().any():
        raise ValueError('Year could not be reconstructed for every exposure row.')
    if 'Observation_Index' not in out.columns: out['Observation_Index']=np.arange(len(out),dtype=int)
    if 'Cycle_ID' not in out.columns: out['Cycle_ID']=[f'OBS_{i:04d}' for i in range(len(out))]
    if 'Load_Date' not in out.columns:
        date_col=next((c for c in out.columns if str(c).lower() in {'date','loading date','加载日期','日期'}),None)
        out['Load_Date']=out[date_col] if date_col else pd.NaT
    return out


def build_master_panel_from_frames(raw:pd.DataFrame)->pd.DataFrame:
    raw=_normalize_wide_columns(raw)
    str_cols=[f'STR{i}' for i in range(1,20)]
    missing=[c for c in ['Year','Cum_Load_10k','Avg_Temp_C',*str_cols] if c not in raw.columns]
    if missing: raise ValueError(f'missing required columns: {missing}')
    ids=['Year','Load_Date','Cycle_ID','Observation_Index','Cum_Load_10k','Avg_Temp_C']
    long=raw[ids+str_cols].melt(id_vars=ids,value_vars=str_cols,var_name='STR_name',value_name='Rutting_Source_0_1mm')
    long['Rutting_Observed']=long['Rutting_Source_0_1mm'].notna().astype(int)
    long['Rutting_mm']=pd.to_numeric(long['Rutting_Source_0_1mm'],errors='coerce')/10.0
    long=long.merge(structure_table(),on='STR_name',how='left',validate='many_to_one')
    long['STR_num']=long['STR_name'].str.replace('STR','',regex=False).astype(int)
    long=long.sort_values(['STR_num','Observation_Index'],kind='mergesort').reset_index(drop=True)
    g=long.groupby('STR_name',sort=False)
    long['Delta_Load_10k']=g['Cum_Load_10k'].diff().fillna(0.0)
    long['Delta_Temp_C']=g['Avg_Temp_C'].diff().fillna(0.0)
    long['Temp_MA']=g['Avg_Temp_C'].transform(lambda s:s.rolling(3,min_periods=1).mean())
    # memory uses only observed rutting; missing target is preserved but does not become a false observation
    prev=[]; inc=[]; peak=[]
    for _,sub in long.groupby('STR_name',sort=False):
        last=np.nan; last_inc=np.nan; max_seen=np.nan
        for y,obs in zip(sub['Rutting_mm'],sub['Rutting_Observed']):
            prev.append(last); inc.append(last_inc); peak.append(max_seen)
            if obs:
                new_inc=(y-last) if np.isfinite(last) else np.nan
                last=float(y); last_inc=new_inc
                max_seen=last if not np.isfinite(max_seen) else max(max_seen,last)
    long['Previous_Rutting_mm']=prev; long['Previous_Increment_mm']=inc; long['Previous_Peak_mm']=peak
    return long.drop(columns=['STR_num'])


def read_raw_xlsx(path: str|Path, header:int=1)->pd.DataFrame:
    # Runtime convenience for user; V3.1 tests operate on frames directly.
    return pd.read_excel(path,header=header)


def build_master_panel(raw_path: str|Path, output_csv: str|Path, header:int=1)->Path:
    raw=read_raw_xlsx(raw_path,header=header)
    panel=build_master_panel_from_frames(raw)
    out=Path(output_csv); out.parent.mkdir(parents=True,exist_ok=True); panel.to_csv(out,index=False,encoding='utf-8-sig')
    return out
