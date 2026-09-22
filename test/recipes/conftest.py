"""Make the recipe helpers and the shared test helpers importable."""

import pathlib
import sys

_here = pathlib.Path(__file__).resolve().parent
for path in (_here, _here.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
