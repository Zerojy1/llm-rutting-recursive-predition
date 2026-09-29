from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
SCRIPTS=ROOT/"project"/"scripts"
if str(SCRIPTS) not in sys.path: sys.path.insert(0,str(SCRIPTS))
from v31_stages import stage03_qwen_train

if __name__=="__main__":
    print("=== 2016-17训练+2018选择，再2016-18固定epoch重拟合 ===")
    stage03_qwen_train(ROOT/"project")
