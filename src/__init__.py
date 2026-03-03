from __future__ import annotations

import sys
from pathlib import Path

# Support both `python src/main.py` (top-level imports) and `import src.*` usage.
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

