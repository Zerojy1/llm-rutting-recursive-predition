from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'project/scripts'))
from v32_reanalysis import recompute_contiguous_mssr
print('=== V3.2 Contiguous MSSR 重算：无需重训任何模型 ===')
print(recompute_contiguous_mssr(ROOT/'project'))
