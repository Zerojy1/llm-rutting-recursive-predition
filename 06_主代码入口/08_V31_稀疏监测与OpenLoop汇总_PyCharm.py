from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
SCRIPTS=ROOT/"project"/"scripts"
if str(SCRIPTS) not in sys.path: sys.path.insert(0,str(SCRIPTS))
from v31_stages import stage08_sparse_monitoring

if __name__=="__main__":
    print("=== 汇总H=1/2/4/6/12/open-loop监测效率前沿 ===")
    stage08_sparse_monitoring(ROOT/"project")
