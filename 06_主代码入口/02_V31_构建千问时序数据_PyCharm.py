from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
SCRIPTS=ROOT/"project"/"scripts"
if str(SCRIPTS) not in sys.path: sys.path.insert(0,str(SCRIPTS))
from v31_stages import stage02_qwen_data

if __name__=="__main__":
    print("=== 构建防泄漏Qwen训练/验证/重拟合/2019保留集 ===")
    stage02_qwen_data(ROOT/"project")
