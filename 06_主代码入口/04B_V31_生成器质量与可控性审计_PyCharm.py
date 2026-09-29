from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
SCRIPTS=ROOT/"project"/"scripts"
if str(SCRIPTS) not in sys.path: sys.path.insert(0,str(SCRIPTS))
from v31_stages import stage04_generator_quality

if __name__=="__main__":
    print("=== Base/FT/QC Qwen与生成基线质量审计 ===")
    stage04_generator_quality(ROOT/"project")
