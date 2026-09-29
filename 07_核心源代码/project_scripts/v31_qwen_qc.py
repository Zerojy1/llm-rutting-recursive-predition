from __future__ import annotations
import numpy as np
from v31_qwen_generate import consensus_generate
from v31_quality import hard_validate_trajectory,local_prospective_rutting_bounds,soft_plausibility_score

class QualityControlledQwen:
    def __init__(self,generator,increment_bounds,rutting_bounds,k=5,max_attempts=3,temperature=.7,
                 local_center=None,local_scale=None):
        self.generator=generator; self.increment_bounds=increment_bounds; self.rutting_bounds=rutting_bounds
        self.k=k; self.max_attempts=max_attempts; self.temperature=temperature
        self.local_center=None if local_center is None else np.asarray(local_center,float)
        self.local_scale=None if local_scale is None else np.asarray(local_scale,float)
        self.audit=[]
    def __call__(self,condition):
        accepted=[]; errors=[]; mads=[]
        for attempt in range(1,self.max_attempts+1):
            try:
                c=consensus_generate(self.generator,condition,k=self.k,temperature=self.temperature); inc=c['increments']
            except Exception as e:
                errors.append(f'generation_error:{type(e).__name__}:{e}')
                continue
            local_rb=local_prospective_rutting_bounds(condition['anchor_rutting_mm'],self.increment_bounds,len(inc),self.rutting_bounds)
            vr=hard_validate_trajectory(inc,condition['anchor_rutting_mm'],self.increment_bounds,local_rb)
            if vr.valid:
                accepted.append(np.asarray(inc,float)); mads.append(np.asarray(c['mad'],float)); break
            errors.extend(vr.reasons)
        if not accepted: raise RuntimeError('Qwen QC failed after regeneration: '+','.join(map(str,errors))+f" | anchor={condition.get('anchor_rutting_mm')} horizon={len(condition.get('future_scenario',[]))} increment_bounds={self.increment_bounds} base_rutting_bounds={self.rutting_bounds}")
        x=accepted[0]; mean_mad=float(np.mean(mads[0])) if mads else np.nan
        score=np.nan
        if self.local_center is not None and self.local_scale is not None:
            center=np.resize(self.local_center,x.shape); scale=np.resize(self.local_scale,x.shape)
            score=soft_plausibility_score(x,center,scale,disagreement=mean_mad if np.isfinite(mean_mad) else 0.0)
        self.audit.append({'accepted':True,'attempts':attempt,'mean_mad':mean_mad,
                           'soft_plausibility_score':float(score) if np.isfinite(score) else np.nan,'reasons':errors})
        return x.tolist()
