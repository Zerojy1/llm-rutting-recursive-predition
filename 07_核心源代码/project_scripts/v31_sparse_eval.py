from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from v31_experiment_engine import fixed_real_evaluation_arrays


def rollout_values(y,predict_fn,gap):
    y=np.asarray(y,float); state=float(y[0]); peak=state; preds=[]
    for j in range(1,len(y)):
        pred=float(predict_fn(state,peak)); preds.append(pred)
        if gap is not None and j%gap==0: state=float(y[j]); peak=max(peak,state)
        else: state=pred; peak=max(peak,state)
    return {'pred':np.asarray(preds),'truth':y[1:]}


def sparse_eval_fit(master,fit_result,model_type='gru',gaps=(1,2,4,6,12,None),seq_len=6):
    X,y,p,pk,meta=fixed_real_evaluation_arrays(master,seq_len)['eval2021']; rows=[]
    # index into full 2021 target arrays by structure; first 2021 target becomes the initial observation anchor
    for gap in gaps:
        se=[]; n=0
        for sid,idxs in meta.groupby('STR_name',sort=False).groups.items():
            idx=np.asarray(list(idxs),int); order=np.argsort(meta.loc[idx,'Observation_Order'].to_numpy()); idx=idx[order]
            if len(idx)<2: continue
            state=float(y[idx[0]]); peak=max(float(pk[idx[0]].reshape(-1)[0]),state)
            for local,j in enumerate(idx[1:],start=1):
                if model_type=='gru':
                    from v31_gru import predict_gru
                    pred=float(predict_gru(fit_result,X[j:j+1],np.array([[state]],np.float32))[0])
                else:
                    from v31_tdr import predict_tdr
                    pred=float(predict_tdr(fit_result,X[j:j+1],np.array([[state]],np.float32),np.array([[peak]],np.float32))[0])
                se.append((pred-float(y[j]))**2); n+=1
                if gap is not None and local%gap==0: state=float(y[j]); peak=max(peak,state)
                else: state=pred; peak=max(peak,state)
        rows.append({'gap':'open' if gap is None else gap,'observation_frequency':0.0 if gap is None else 1.0/gap,'rmse':float(np.sqrt(np.mean(se))) if se else np.nan,'n_predictions':n})
    return pd.DataFrame(rows)
