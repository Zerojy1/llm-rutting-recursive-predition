from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
SCRIPTS=ROOT/"project"/"scripts"
if str(SCRIPTS) not in sys.path: sys.path.insert(0,str(SCRIPTS))
from v31_stages import stage07_capacity_and_state

if __name__=="__main__":
    print("=== 状态空间稀疏机制+GRU容量适用边界 ===")
    stage07_capacity_and_state(ROOT/"project")
