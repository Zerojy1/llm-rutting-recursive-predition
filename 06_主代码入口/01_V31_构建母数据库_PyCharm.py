from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
SCRIPTS=ROOT/"project"/"scripts"
if str(SCRIPTS) not in sys.path: sys.path.insert(0,str(SCRIPTS))
from v31_stages import stage01_master

if __name__=="__main__":
    print("=== 构建并审计V3.1母数据库 ===")
    stage01_master(ROOT/"project")
