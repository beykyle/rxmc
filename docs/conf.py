"""Sphinx configuration for rxmc documentation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "rxmc"
copyright = "2026, rxmc contributors"
author = "rxmc contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "alabaster"
