from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from v31_windows import build_generator_windows, window_to_qwen_record


def _write_jsonl(path:Path, records:list[dict]):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for r in records: f.write(json.dumps(r,ensure_ascii=False)+'\n')


def _make_records(w:pd.DataFrame,history_steps:int,future_steps:int)->list[dict]:
    return [window_to_qwen_record(r,history_steps=history_steps,future_steps=future_steps) for _,r in w.iterrows()]


def build_chronological_qwen_datasets(master:pd.DataFrame,out_dir:str|Path,history_steps=4,future_steps=6):
    """Create V3.1 chronological generator datasets with whole-target-horizon guards."""
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    # Training: all targets <=2017. Development: all targets in 2018.
    train=build_generator_windows(master,history_steps,future_steps,allowed_target_year_max=2017)
    dev=build_generator_windows(master,history_steps,future_steps,allowed_target_year_min=2018,allowed_target_year_max=2018)
    refit=build_generator_windows(master,history_steps,future_steps,allowed_target_year_max=2018)
    hold=build_generator_windows(master,history_steps,future_steps,allowed_target_year_min=2019,allowed_target_year_max=2019)
    paths={
        'train':out/'qwen_train_2016_2017.jsonl','dev':out/'qwen_dev_2018.jsonl',
        'refit':out/'qwen_refit_2016_2018.jsonl','holdout':out/'qwen_holdout_2019.jsonl'}
    frames={'train':train,'dev':dev,'refit':refit,'holdout':hold}
    for k,frame in frames.items():
        _write_jsonl(paths[k],_make_records(frame,history_steps,future_steps)); frame.to_csv(out/f'{k}_windows.csv',index=False,encoding='utf-8-sig')
    manifest={k:{'n':len(v),'target_year_min':None if len(v)==0 else int(v.target_year_min.min()),'target_year_max':None if len(v)==0 else int(v.target_year_max.max()),'path':paths[k].name} for k,v in frames.items()}
    (out/'qwen_chronological_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return frames
