from __future__ import annotations
import numpy as np
import pandas as pd
from v31_master_data import STRUCTURE_COLUMNS

HISTORY_COVARS=['Cum_Load_10k','Delta_Load_10k','Avg_Temp_C','Delta_Temp_C','Temp_MA']
FUTURE_COVARS=['Cum_Load_10k','Delta_Load_10k','Avg_Temp_C','Delta_Temp_C']
HISTORY_STEPS=4


def _finite(value):
    try:
        x=float(value)
        return x if np.isfinite(x) else None
    except (TypeError,ValueError):
        return None


def static_condition_from_window(row,history_steps:int=HISTORY_STEPS):
    """Same eligible static/history information supplied to every generator baseline.

    Layout: 18 structure slots + anchor + for each history step
    [cum load, delta load, temp, delta temp, temp MA, rutting-filled, rutting-observed-mask].
    Missing historical rutting is filled by the latest anchor only for numeric model input;
    the explicit mask prevents the baseline from treating it as an observed measurement.
    """
    anchor=float(row['anchor_rutting_mm'])
    vals=[float(row[c]) for c in STRUCTURE_COLUMNS]+[anchor]
    for h in range(1,history_steps+1):
        for c in HISTORY_COVARS:
            x=_finite(row.get(f'h{h}_{c}'))
            vals.append(0.0 if x is None else x)
        rut=_finite(row.get(f'h{h}_Rutting_mm'))
        vals.extend([anchor if rut is None else rut,0.0 if rut is None else 1.0])
    return np.asarray(vals,dtype=np.float32)


def static_condition_from_runtime(condition:dict,history_steps:int=HISTORY_STEPS):
    anchor=float(condition['anchor_rutting_mm'])
    vals=[float(condition['structure_cm'][c]) for c in STRUCTURE_COLUMNS]+[anchor]
    hist=list(condition.get('history',[]))[-history_steps:]
    # Left-pad missing early history with an explicit unobserved row.
    hist=[None]*(history_steps-len(hist))+hist
    keys=['cum_load_10k','delta_load_10k','temperature_c','delta_temperature_c','temp_ma_c']
    for item in hist:
        item={} if item is None else item
        for k in keys:
            x=_finite(item.get(k)); vals.append(0.0 if x is None else x)
        rut=_finite(item.get('rutting_mm'))
        vals.extend([anchor if rut is None else rut,0.0 if rut is None else 1.0])
    return np.asarray(vals,dtype=np.float32)


def windows_to_generator_arrays(windows:pd.DataFrame,future_steps:int=6):
    static=np.stack([static_condition_from_window(r) for _,r in windows.iterrows()]).astype(np.float32)
    dyn=[]; target=[]
    for _,r in windows.iterrows():
        d=[]; y=[]
        for f in range(1,future_steps+1):
            d.append([r[f'f{f}_Cum_Load_10k'],r[f'f{f}_Delta_Load_10k'],r[f'f{f}_Avg_Temp_C'],r[f'f{f}_Delta_Temp_C']])
            y.append([r[f'target_increment_f{f}']])
        dyn.append(d); target.append(y)
    return {'target':np.asarray(target,np.float32),'static':static,'dynamic':np.asarray(dyn,np.float32),
            'anchor':windows.get('anchor_rutting_mm',pd.Series(np.nan,index=windows.index)).to_numpy(np.float32)}


def joint_timegan_array(windows:pd.DataFrame,future_steps:int=6):
    """Joint multivariate sequence for classic TimeGAN.

    The static/history conditioning vector is repeated over the future sequence and jointly
    generated with future load-temperature covariates and target increments. At inference,
    nearest-condition retrieval uses these same eligible conditioning variables.
    """
    a=windows_to_generator_arrays(windows,future_steps); st=np.repeat(a['static'][:,None,:],future_steps,axis=1)
    return np.concatenate([st,a['dynamic'],a['target']],axis=-1).astype(np.float32)
