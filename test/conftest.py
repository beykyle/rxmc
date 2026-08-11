"""Make the shared test helpers importable under any pytest import mode."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
