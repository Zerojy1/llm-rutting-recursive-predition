from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
SCRIPTS=ROOT/"project"/"scripts"
if str(SCRIPTS) not in sys.path: sys.path.insert(0,str(SCRIPTS))
from v31_stages import stage06_equal_budget

if __name__=="__main__":
    print("=== 固定2019更新预算，运行真实/模拟替代与real-bootstrap控制 ===")
    stage06_equal_budget(ROOT/"project")
