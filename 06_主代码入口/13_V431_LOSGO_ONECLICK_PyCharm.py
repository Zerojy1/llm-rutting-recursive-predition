from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "project" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from v431_stage13_oneclick import main


if __name__ == "__main__":
    # Formal one-click entry. No V43_MODE, V43_FOLD or V43_MAX_STEPS edits are needed.
    main(ROOT / "project")
