from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib, json

@dataclass(frozen=True)
class V31Protocol:
    protocol_name: str = 'Qwen-TDR-V3.1'
    generator_train_years: tuple[int,int] = (2016, 2017)
    generator_development_year: int = 2018
    generator_refit_years: tuple[int,int,int] = (2016, 2017, 2018)
    generator_holdout_year: int = 2019
    downstream_history_max_year: int = 2018
    downstream_update_year: int = 2019
    downstream_development_year: int = 2020
    chronological_evaluation_year: int = 2021
    history_steps: int = 4
    future_steps: int = 6
    replacement_fractions: tuple[float,...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    allocation_seeds: tuple[int,...] = (202608141,202608142,202608143,202608144,202608145)
    neural_seeds: tuple[int,...] = (42,43,44)
    noninferiority_margin: float = 0.10
    sensitivity_margin: float = 0.05
    max_optimizer_steps: int = 2400
    warmup_steps: int = 100
    validation_every_steps: int = 50
    patience_checks: int = 6


def validate_protocol(p: V31Protocol) -> None:
    if tuple(p.generator_train_years)!=(2016,2017): raise ValueError('generator training years changed')
    if p.generator_development_year!=2018 or p.generator_holdout_year!=2019: raise ValueError('generator chronology changed')
    if (p.downstream_history_max_year,p.downstream_update_year,p.downstream_development_year,p.chronological_evaluation_year)!=(2018,2019,2020,2021): raise ValueError('downstream chronology changed')
    if not all(0 <= q <= 1 for q in p.replacement_fractions): raise ValueError('invalid replacement fraction')
    if sorted(p.replacement_fractions)!=list(p.replacement_fractions): raise ValueError('replacement fractions must be ordered')
    if p.noninferiority_margin<=0 or p.max_optimizer_steps<=0: raise ValueError('invalid frozen numeric settings')


def protocol_signature(p: V31Protocol|None=None)->str:
    p=p or V31Protocol(); validate_protocol(p)
    payload=json.dumps(asdict(p),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def write_protocol_json(path: str|Path, p: V31Protocol|None=None)->Path:
    p=p or V31Protocol(); validate_protocol(p)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    payload=asdict(p); payload['protocol_signature']=protocol_signature(p)
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    return path
