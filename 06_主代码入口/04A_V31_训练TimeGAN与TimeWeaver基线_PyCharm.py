from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
SCRIPTS=ROOT/"project"/"scripts"
if str(SCRIPTS) not in sys.path: sys.path.insert(0,str(SCRIPTS))
from v31_stages import stage04_train_generator_baselines

if __name__=="__main__":
    print("=== 训练TimeGAN与Time-Weaver-inspired条件扩散基线 ===")
    stage04_train_generator_baselines(ROOT/"project")
