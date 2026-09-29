from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from v31_master_data import STRUCTURE_COLUMNS
from v31_quality import hard_validate_trajectory,memorization_audit
from v31_quality_audit import compare_real_synthetic

def _finite_or_none(v):
    try:
        x=float(v); return x if np.isfinite(x) else None
    except (TypeError,ValueError): return None


def condition_from_window(row,future_steps=6,history_steps=4):
    history=[]
    for h in range(1,history_steps+1):
        history.append({'cum_load_10k':_finite_or_none(row.get(f'h{h}_Cum_Load_10k')),'delta_load_10k':_finite_or_none(row.get(f'h{h}_Delta_Load_10k')),'temperature_c':_finite_or_none(row.get(f'h{h}_Avg_Temp_C')),'delta_temperature_c':_finite_or_none(row.get(f'h{h}_Delta_Temp_C')),'temp_ma_c':_finite_or_none(row.get(f'h{h}_Temp_MA')),'rutting_mm':_finite_or_none(row.get(f'h{h}_Rutting_mm'))})
    future=[]
    for f in range(1,future_steps+1): future.append({'cum_load_10k':row[f'f{f}_Cum_Load_10k'],'delta_load_10k':row[f'f{f}_Delta_Load_10k'],'temperature_c':row[f'f{f}_Avg_Temp_C'],'delta_temperature_c':row[f'f{f}_Delta_Temp_C']})
    return {'STR_name':row.STR_name,'structure_cm':{c:float(row[c]) for c in STRUCTURE_COLUMNS},'anchor_rutting_mm':float(row.anchor_rutting_mm),'history':history,'future_scenario':future}


def evaluate_generator_on_windows(windows,generator_fn,future_steps=6,increment_bounds=None,rutting_bounds=None,max_windows=None):
    frame=windows.iloc[:max_windows].copy() if max_windows else windows
    rows=[]; real=[]; syn=[]
    for _,r in frame.iterrows():
        cond=condition_from_window(r,future_steps); truth=np.array([r[f'target_increment_f{i}'] for i in range(1,future_steps+1)],float)
        try:
            pred=np.asarray(generator_fn(cond),float).reshape(-1)
        except Exception as e:
            rows.append({'window_id':r.window_id,'STR_name':r.STR_name,'valid':False,
                         'reasons':'generation_error:'+type(e).__name__,'rmse_increment':np.nan})
            continue
        if len(pred)!=future_steps or not np.all(np.isfinite(pred)):
            reason='invalid_sequence_length' if len(pred)!=future_steps else 'non_finite_generation'
            rows.append({'window_id':r.window_id,'STR_name':r.STR_name,'valid':False,'reasons':reason,'rmse_increment':np.nan})
            continue
        valid=True; reasons=[]
        if increment_bounds is not None and rutting_bounds is not None:
            vr=hard_validate_trajectory(pred,float(r.anchor_rutting_mm),increment_bounds,rutting_bounds); valid=vr.valid; reasons=list(vr.reasons)
        rows.append({'window_id':r.window_id,'STR_name':r.STR_name,'valid':valid,'reasons':'|'.join(reasons),'rmse_increment':float(np.sqrt(np.mean((pred-truth)**2)))})
        real.append(truth); syn.append(pred)
    R=np.asarray(real,dtype=float).reshape(-1,future_steps); S=np.asarray(syn,dtype=float).reshape(-1,future_steps); summary=compare_real_synthetic(R,S) if len(R) else {}
    attempted=len(rows); generated=len(R)
    summary.update({'n_windows':attempted,'n_attempted':attempted,'n_generated':generated,
                    'generation_success_rate':float(generated/attempted) if attempted else np.nan,
                    'valid_rate':float(np.mean([bool(r['valid']) for r in rows])) if rows else np.nan})
    return pd.DataFrame(rows),summary,R,S


def save_generator_evaluation(out_dir,name,rows,summary,real=None,synthetic=None,training_real=None):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); rows.to_csv(out/f'{name}_window_metrics.csv',index=False,encoding='utf-8-sig')
    (out/f'{name}_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    if real is not None and synthetic is not None:
        audit=memorization_audit(np.asarray(training_real if training_real is not None else real),np.asarray(synthetic)); np.savez_compressed(out/f'{name}_memorization.npz',dcr=audit['dcr'],nn_distance_ratio=audit['nn_distance_ratio']);
        m={k:v for k,v in audit.items() if k not in {'dcr','nn_distance_ratio'}}; (out/f'{name}_memorization.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
