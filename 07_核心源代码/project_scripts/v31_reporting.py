from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from v31_statistics import maximum_synthetic_substitution_ratio


def summarize_substitution_table(table:pd.DataFrame,margin=.10):
    return {'mssr':maximum_synthetic_substitution_ratio(table,margin),'margin':float(margin),'n_ratios':int(len(table))}


def plot_substitution_curve(table,out_path,margin=.10):
    import matplotlib.pyplot as plt
    p=Path(out_path); p.parent.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(6.5,4.5))
    x=100*pd.to_numeric(table.synthetic_fraction); y=100*pd.to_numeric(table.get('relative_degradation',table.get('mean_relative_degradation')))
    ax.plot(x,y,marker='o'); ax.axhline(100*margin,linestyle='--'); ax.axhline(0,linewidth=.8)
    ax.set_xlabel('Synthetic replacement ratio (%)'); ax.set_ylabel('Relative RMSE degradation (%)'); ax.set_title('Equal-budget synthetic substitution')
    fig.tight_layout(); fig.savefig(p,dpi=300); plt.close(fig); return p


def plot_monitoring_frontier(table,out_path):
    import matplotlib.pyplot as plt
    p=Path(out_path); p.parent.mkdir(parents=True,exist_ok=True); fig,ax=plt.subplots(figsize=(6.5,4.5))
    for name,g in table.groupby('condition'):
        ax.plot(g['observation_frequency'],g['rmse'],marker='o',label=name)
    ax.set_xlabel('Real rutting observation frequency'); ax.set_ylabel('RMSE (mm)'); ax.legend(); ax.set_title('Monitoring Efficiency Frontier'); fig.tight_layout(); fig.savefig(p,dpi=300); plt.close(fig); return p


def plot_capacity_map(table,out_path,value_col='relative_degradation'):
    import matplotlib.pyplot as plt
    p=Path(out_path); p.parent.mkdir(parents=True,exist_ok=True)
    pivot=table.pivot_table(index='capacity',columns='synthetic_fraction',values=value_col,aggfunc='mean')
    fig,ax=plt.subplots(figsize=(7,4.8)); im=ax.imshow(pivot.to_numpy(),aspect='auto')
    ax.set_yticks(range(len(pivot.index)),pivot.index); ax.set_xticks(range(len(pivot.columns)),[f'{100*x:.0f}%' for x in pivot.columns]); ax.set_xlabel('Synthetic replacement ratio'); ax.set_ylabel('GRU capacity')
    fig.colorbar(im,ax=ax,label=value_col); fig.tight_layout(); fig.savefig(p,dpi=300); plt.close(fig); return p
