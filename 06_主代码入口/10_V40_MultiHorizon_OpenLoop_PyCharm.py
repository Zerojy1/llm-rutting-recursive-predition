from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "project" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from v40_stage10 import stage10


if __name__ == "__main__":
    phase = os.environ.get("V40_PHASE", "B").strip().upper() or "A"
    print("=== V4.0 Stage 10: multi-horizon rolling-origin open-loop evaluation ===")
    print(f"phase={phase}; primary H=6; secondary H=1/3/12")
    stage10(ROOT / "project", phase=phase)
