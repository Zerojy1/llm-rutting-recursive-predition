from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
SCRIPTS=ROOT/"project"/"scripts"
if str(SCRIPTS) not in sys.path: sys.path.insert(0,str(SCRIPTS))
from v31_stages import stage09_report

if __name__=="__main__":
    print("=== structure-cluster-aware MSSR统计与论文主图 ===")
    stage09_report(ROOT/"project")
