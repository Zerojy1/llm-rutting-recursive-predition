from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "project" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from v43_stage13 import main


if __name__ == "__main__":
    main(ROOT / "project")
