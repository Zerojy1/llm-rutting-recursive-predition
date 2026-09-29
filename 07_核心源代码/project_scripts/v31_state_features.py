from __future__ import annotations
import numpy as np
import pandas as pd
from v31_master_data import STRUCTURE_COLUMNS

# Full state is retained for descriptive diagnostics only. Absolute cumulative load and
# previous rutting are progression coordinates in a chronological deterioration problem;
# they are therefore not used as hard OOD gates in V3.2.
FULL_STATE_COLS=['log_cum_load','Delta_Load_10k','Avg_Temp_C','Delta_Temp_C','Previous_Rutting_mm']+STRUCTURE_COLUMNS
CONTEXT_SUPPORT_COLS=['Delta_Load_10k','Avg_Temp_C','Delta_Temp_C']+STRUCTURE_COLUMNS
PROGRESSION_COLS=['log_cum_load','Previous_Rutting_mm']
# Backward-compatible alias.
STATE_COLS=FULL_STATE_COLS


def _numeric_frame(frame:pd.DataFrame):
    x=frame.copy()
    x['log_cum_load']=np.log1p(pd.to_numeric(x['Cum_Load_10k'],errors='coerce').clip(lower=0))
    needed=['Delta_Load_10k','Avg_Temp_C','Delta_Temp_C','Previous_Rutting_mm']+STRUCTURE_COLUMNS
    for c in needed:
        x[c]=pd.to_numeric(x[c],errors='coerce')
    return x


def _meta(x,valid):
    cols=[c for c in ['STR_name','Observation_Index','Year','Cum_Load_10k','Previous_Rutting_mm'] if c in x.columns]
    return x.loc[valid,cols].copy().reset_index(drop=True)


def state_feature_frame(frame:pd.DataFrame):
    """Legacy/full-state representation, retained for descriptive distance diagnostics."""
    x=_numeric_frame(frame); valid=x[FULL_STATE_COLS].notna().all(axis=1)
    return x.loc[valid,FULL_STATE_COLS].to_numpy(float),_meta(x,valid)


def context_support_frame(frame:pd.DataFrame):
    """Context representation used for V3.2 in-domain support/OOD analysis.

    The hard support dimensions are transition exposure and pavement structure. Absolute
    cumulative load and previous rutting are excluded because they necessarily advance
    during chronological deterioration and would otherwise label all future years OOD.
    """
    x=_numeric_frame(frame); valid=x[CONTEXT_SUPPORT_COLS].notna().all(axis=1)
    return x.loc[valid,CONTEXT_SUPPORT_COLS].to_numpy(float),_meta(x,valid)


def progression_frame(frame:pd.DataFrame):
    """Chronological progression coordinates reported separately from context OOD."""
    x=_numeric_frame(frame); valid=x[PROGRESSION_COLS].notna().all(axis=1)
    return x.loc[valid,PROGRESSION_COLS].to_numpy(float),_meta(x,valid)
