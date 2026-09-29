from __future__ import annotations

from copy import deepcopy
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


def _clip(value, bounds):
    x=float(value)
    if bounds is None:
        return x
    lo,hi=bounds
    return float(np.clip(x,float(lo),float(hi)))


def make_condition_variant(condition, kind, limits=None, alternate_structure=None,
                           load_scale=1.15, temperature_shift_c=2.0, state_shift_mm=0.2):
    """Create one paired counterfactual condition without using future rutting truth.

    The perturbations are deliberately modest. They are used to measure whether a generator
    responds to conditioning information; the audit does *not* assume a universal monotonic
    engineering direction for the generated rutting response.
    """
    c=deepcopy(condition); limits=limits or {}; kind=str(kind).lower()
    if kind=='load':
        future=c.get('future_scenario',[])
        if not future: return c
        history=c.get('history',[])
        last_cum=None
        for r in reversed(history):
            v=r.get('cum_load_10k')
            if v is not None and np.isfinite(float(v)):
                last_cum=float(v); break
        if last_cum is None:
            first=future[0]
            last_cum=float(first.get('cum_load_10k',0.0))-float(first.get('delta_load_10k',0.0))
        running=last_cum
        for r in future:
            d=_clip(float(r.get('delta_load_10k',0.0))*float(load_scale),limits.get('delta_load_10k'))
            running=_clip(running+d,limits.get('cum_load_10k'))
            r['delta_load_10k']=d; r['cum_load_10k']=running
        return c
    if kind=='temperature':
        for r in c.get('future_scenario',[]):
            if r.get('temperature_c') is not None:
                r['temperature_c']=_clip(float(r['temperature_c'])+float(temperature_shift_c),limits.get('temperature_c'))
        return c
    if kind=='state':
        c['anchor_rutting_mm']=_clip(float(c['anchor_rutting_mm'])+float(state_shift_mm),limits.get('rutting_mm'))
        for r in c.get('history',[]):
            if r.get('rutting_mm') is not None:
                r['rutting_mm']=_clip(float(r['rutting_mm'])+float(state_shift_mm),limits.get('rutting_mm'))
        return c
    if kind=='structure':
        if alternate_structure is None:
            raise ValueError('alternate_structure is required for structure variant')
        c['structure_cm']={str(k):float(v) for k,v in dict(alternate_structure).items()}
        c['STR_name']=str(c.get('STR_name',''))+'__counterfactual_structure'
        return c
    raise ValueError(f'unknown controllability variant: {kind}')


def _draw(generator_fn, condition, replicates):
    draws=[]
    for _ in range(int(replicates)):
        x=np.asarray(generator_fn(deepcopy(condition)),dtype=float).reshape(-1)
        if not np.isfinite(x).all():
            raise ValueError('generator returned non-finite trajectory during controllability audit')
        draws.append(x)
    if not draws: raise ValueError('replicates must be positive')
    a=np.stack(draws)
    return a,a.mean(axis=0)


def conditional_controllability_audit(conditions: Iterable[Mapping], generator_fn,
                                      variant_kinds=('load','temperature','state'), replicates=2,
                                      limits=None, alternate_structures=None):
    """Quantify paired conditional sensitivity without imposing a direction-of-effect claim.

    Generation failures are part of the audit rather than fatal to the whole experiment.
    A pair contributes a response metric only when both the base and modified conditions
    produce the requested number of finite draws.
    """
    rows=[]; conditions=list(conditions); alternate_structures=list(alternate_structures or [])
    base_failures=0; variant_failures=0
    for i,cond in enumerate(conditions):
        try:
            base_draws,base_mean=_draw(generator_fn,cond,replicates)
        except Exception:
            base_failures+=1; continue
        base_noise=float(np.sqrt(np.mean((base_draws-base_mean[None,:])**2)))
        for kind in variant_kinds:
            alt=None
            if kind=='structure':
                if not alternate_structures: continue
                alt=alternate_structures[i % len(alternate_structures)]
                if dict(alt)==dict(cond.get('structure_cm',{})) and len(alternate_structures)>1:
                    alt=alternate_structures[(i+1) % len(alternate_structures)]
            changed=make_condition_variant(cond,kind,limits=limits,alternate_structure=alt)
            try:
                var_draws,var_mean=_draw(generator_fn,changed,replicates)
            except Exception:
                variant_failures+=1; continue
            if var_mean.shape!=base_mean.shape:
                variant_failures+=1; continue
            var_noise=float(np.sqrt(np.mean((var_draws-var_mean[None,:])**2)))
            response=float(np.sqrt(np.mean((var_mean-base_mean)**2)))
            noise=max((base_noise+var_noise)/2.0,1e-12)
            rows.append({'condition_index':i,'variant':kind,'response_l2':response,
                         'endpoint_abs_shift':float(abs(var_mean[-1]-base_mean[-1])),
                         'within_condition_noise':float((base_noise+var_noise)/2.0),
                         'response_noise_ratio':float(response/noise)})
    frame=pd.DataFrame(rows)
    summary={}
    if len(frame):
        for kind,g in frame.groupby('variant'):
            summary[str(kind)]={'n_pairs':int(len(g)),'mean_response_l2':float(g.response_l2.mean()),
                                'median_response_l2':float(g.response_l2.median()),
                                'mean_endpoint_abs_shift':float(g.endpoint_abs_shift.mean()),
                                'mean_response_noise_ratio':float(g.response_noise_ratio.mean())}
    summary['_audit']={'n_conditions':int(len(conditions)),'n_base_failures':int(base_failures),
                       'n_variant_failures':int(variant_failures),'n_successful_pairs':int(len(frame))}
    return frame,summary

