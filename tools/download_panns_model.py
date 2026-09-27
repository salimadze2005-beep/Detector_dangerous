from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.bootstrap import install_panns


def main() -> int:
    try:
        install_panns()
        return 0
    except Exception as exc:
        print(f"[ERROR] PANNs download failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
