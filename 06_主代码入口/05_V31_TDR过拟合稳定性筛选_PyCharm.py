from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
SCRIPTS=ROOT/"project"/"scripts"
if str(SCRIPTS) not in sys.path: sys.path.insert(0,str(SCRIPTS))
from v31_stages import stage05_tdr_stability

if __name__=="__main__":
    print("=== 仅用2020development选择稳定TDR配置 ===")
    stage05_tdr_stability(ROOT/"project")
