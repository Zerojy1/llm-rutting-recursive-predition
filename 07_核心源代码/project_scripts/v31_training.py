from __future__ import annotations
from dataclasses import dataclass
import math
import pandas as pd

@dataclass(frozen=True)
class StepBudget:
    max_steps:int=2400
    warmup_steps:int=100
    val_every:int=50
    patience_checks:int=6
    @property
    def max_validation_checks(self): return self.max_steps//self.val_every


def overfitting_index(train_rmse:float,val_rmse:float)->float:
    if val_rmse<=0: return float('nan')
    return float((val_rmse-train_rmse)/val_rmse)


def choose_stable_candidate(candidates:pd.DataFrame)->str:
    required={'name','mean_val_rmse','se_val_rmse','seed_sd','gap','params'}
    if not required.issubset(candidates.columns): raise ValueError('missing stability columns')
    best=candidates.loc[candidates.mean_val_rmse.idxmin()]
    threshold=float(best.mean_val_rmse+best.se_val_rmse)
    eligible=candidates[candidates.mean_val_rmse<=threshold].copy()
    eligible=eligible.sort_values(['seed_sd','gap','params','mean_val_rmse'],kind='mergesort')
    return str(eligible.iloc[0]['name'])


def cosine_lr(step:int,budget:StepBudget,base_lr:float,min_lr:float=0.0)->float:
    if step<budget.warmup_steps: return base_lr*(step+1)/max(1,budget.warmup_steps)
    progress=(step-budget.warmup_steps)/max(1,budget.max_steps-budget.warmup_steps)
    progress=min(max(progress,0.0),1.0)
    return float(min_lr+(base_lr-min_lr)*0.5*(1+math.cos(math.pi*progress)))
