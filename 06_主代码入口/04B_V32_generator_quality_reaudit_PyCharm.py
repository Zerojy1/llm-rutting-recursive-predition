from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'project/scripts'))
from v31_stages import stage04_generator_quality
print('=== V3.2 生成器质量审计：使用前瞻性QC边界；无需重训Qwen/TimeGAN/TimeWeaver ===')
stage04_generator_quality(ROOT/'project')
