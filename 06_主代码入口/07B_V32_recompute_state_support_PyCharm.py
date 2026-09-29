from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'project/scripts'))
from v32_reanalysis import recompute_state_mechanism
print('=== V3.2 状态空间机制重算：无需重训任何模型 ===')
audit=recompute_state_mechanism(ROOT/'project')
print(audit)
