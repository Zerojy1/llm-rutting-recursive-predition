from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parent
SCRIPTS = PACKAGE_ROOT / "project" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from v41_stage11 import stage11


def main(root=None):
    package_root = Path(root) if root is not None else PACKAGE_ROOT
    return stage11(package_root / "project")


if __name__ == "__main__":
    main()
