from __future__ import annotations
from pathlib import Path
import importlib.util, json, os, sys
import pandas as pd
from v31_protocol import V31Protocol, validate_protocol, protocol_signature

CORE_IMPORTS=('numpy','pandas','sklearn','torch','matplotlib','openpyxl')
GPU_IMPORTS=('transformers','peft','bitsandbytes')


def _present(mod):
    return importlib.util.find_spec(mod) is not None


def preflight_report(project_root, require_qwen=False):
    root=Path(project_root)
    p=V31Protocol(); validate_protocol(p)
    required=[
        root/'data/raw/足尺环道车辙数据.xlsx',
        root/'first_model/TDR_PRSN.py',
    ]
    master_path=root/'data/processed/rutting_master_v31.csv'
    if master_path.exists():
        m=pd.read_csv(master_path,usecols=['Rutting_Observed'])
        rows=len(m); observed=int(pd.to_numeric(m.Rutting_Observed,errors='coerce').fillna(0).sum())
    else:
        rows=observed=0
    missing=max(0,rows-observed)
    deps={x:_present(x) for x in CORE_IMPORTS}
    gpu_deps={x:_present(x) for x in GPU_IMPORTS}
    model_dir=Path(os.environ.get('QWEN_BASE_MODEL_DIR',r'K:\ollama\Qwen3.5-9B-hf'))
    qwen_ready=all(gpu_deps.values()) and model_dir.exists()
    report={
        'protocol':'V3.1','protocol_signature':protocol_signature(p),
        'python':sys.version.split()[0],
        'required_files_ok':all(x.exists() for x in required),
        'required_files':{str(x.relative_to(root)):x.exists() for x in required},
        'master_exists':master_path.exists(),'master_rows':rows,
        'observed_rutting':observed,'missing_rutting':missing,
        'core_dependencies':deps,'core_dependencies_ok':all(deps.values()),
        'qwen_dependencies':gpu_deps,'qwen_model_dir':str(model_dir),'qwen_ready':qwen_ready,
        'require_qwen':bool(require_qwen),
    }
    report['ok']=report['required_files_ok'] and report['core_dependencies_ok'] and (qwen_ready if require_qwen else True)
    return report


def run_preflight(project_root,require_qwen=False):
    root=Path(project_root); r=preflight_report(root,require_qwen=require_qwen)
    out=root/'reports/v31/00_preflight.json'; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(r,ensure_ascii=False,indent=2))
    if not r['ok']:
        raise RuntimeError('V3.1 preflight failed; see reports/v31/00_preflight.json')
    return r
