from __future__ import annotations
import importlib.util
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from v31_downstream_data import build_real_downstream_frame, build_sequence_arrays_simple
from v40_openloop import build_eval_arrays, iter_forecast_origins, ForecastOrigin


def _static_features(X, prev):
    x=np.asarray(X,float); p=np.asarray(prev,float).reshape(-1,1)
    return np.concatenate([x[:,-1,:],p],axis=1)


def fit_static_regressor(X,y,prev,backend='auto',seed=20260817):
    backend=str(backend).lower()
    if backend=='auto': backend='xgboost' if importlib.util.find_spec('xgboost') is not None else 'histgb'
    z=_static_features(X,prev); y=np.asarray(y,float)
    if backend=='xgboost':
        from xgboost import XGBRegressor
        model=XGBRegressor(n_estimators=400,max_depth=3,learning_rate=.05,subsample=.8,colsample_bytree=.8,
                           reg_lambda=1.0,objective='reg:squarederror',random_state=int(seed),n_jobs=1)
    elif backend=='histgb':
        model=HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,max_leaf_nodes=15,
                                            l2_regularization=1.0,random_state=int(seed))
    else: raise ValueError(f'unknown static benchmark backend: {backend}')
    model.fit(z,y)
    return {'model':model,'backend':backend,'seed':int(seed),'feature_dim':int(z.shape[1])}


def predict_static(fit,X,prev):
    return np.asarray(fit['model'].predict(_static_features(X,prev)),float).reshape(-1)


def forecast_static_origin(fit, origin: ForecastOrigin, predictor=predict_static):
    state=float(origin.y_t); rows=[]
    for h,j in enumerate(origin.future_positions,1):
        pred=float(predictor(fit,origin.arrays.X[j:j+1],np.array([[state]],dtype=np.float32))[0])
        meta=origin.arrays.meta.iloc[j]
        rows.append({
            'STR_name':origin.structure,'origin_observation_index':int(origin.origin_observation_index),
            'target_observation_index':int(meta['Observation_Order']),'requested_horizon':int(origin.requested_horizon),
            'step_h':int(h),'y_origin':float(origin.y_t),'y_true':float(origin.arrays.y[j]),
            'y_pred':pred,'persistence_pred':float(origin.y_t),
        })
        state=pred
    return rows



def forecast_static_all_origins(fit, eval_arrays, horizon, predictor=predict_static):
    """Recursive static benchmark, batched across all eligible origins at each step."""
    origins=list(iter_forecast_origins(eval_arrays,int(horizon)))
    if not origins:
        return pd.DataFrame()
    states=np.asarray([o.y_t for o in origins],dtype=np.float32)
    rows_by_origin=[[] for _ in origins]
    for h in range(1,int(horizon)+1):
        positions=np.asarray([o.future_positions[h-1] for o in origins],dtype=int)
        preds=np.asarray(predictor(fit,eval_arrays.X[positions],states.reshape(-1,1)),float).reshape(-1)
        if len(preds)!=len(origins):
            raise ValueError('predictor returned a batch with unexpected length')
        for i,(origin,j,pred) in enumerate(zip(origins,positions,preds)):
            meta=eval_arrays.meta.iloc[int(j)]
            rows_by_origin[i].append({
                'STR_name':origin.structure,
                'origin_observation_index':int(origin.origin_observation_index),
                'target_observation_index':int(meta['Observation_Order']),
                'requested_horizon':int(origin.requested_horizon),
                'step_h':int(h),
                'y_origin':float(origin.y_t),
                'y_true':float(eval_arrays.y[int(j)]),
                'y_pred':float(pred),
                'persistence_pred':float(origin.y_t),
            })
        states=preds.astype(np.float32,copy=False)
    return pd.DataFrame([r for rs in rows_by_origin for r in rs])

def fit_full_real_static_benchmark(master: pd.DataFrame,seq_len=6,backend='auto',seed=20260817):
    real=build_real_downstream_frame(master)
    X,y,prev,peak,meta=build_sequence_arrays_simple(real,seq_len=seq_len)
    mask=pd.to_numeric(meta.Year,errors='coerce').le(2019).to_numpy()
    return fit_static_regressor(X[mask],y[mask],prev[mask],backend=backend,seed=seed)


def run_static_full_real_benchmark(master: pd.DataFrame,horizons=(1,3,6,12),seq_len=6,backend='auto',seed=20260817):
    fit=fit_full_real_static_benchmark(master,seq_len=seq_len,backend=backend,seed=seed)
    arrays=build_eval_arrays(master,seq_len=seq_len); frames=[]
    for h in horizons:
        rr=forecast_static_all_origins(fit,arrays,int(h))
        if len(rr):
            rr=rr.copy()
            rr['model']=fit['backend']; rr['method']='full_real_static'; rr['synthetic_fraction']=0.0
            rr['allocation']=0; rr['seed']=int(seed)
            frames.append(rr)
    return fit,(pd.concat(frames,ignore_index=True) if frames else pd.DataFrame())
