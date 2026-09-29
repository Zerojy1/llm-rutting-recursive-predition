from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent/'project'
sys.path.insert(0,str(ROOT/'scripts'))
from v31_preflight import run_preflight
if __name__=='__main__':
    run_preflight(ROOT,require_qwen=False)
