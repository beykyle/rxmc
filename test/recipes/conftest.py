"""Make the shared test helpers importable from the recipe suite."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
