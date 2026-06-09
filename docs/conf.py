import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))

project = "rxmc"
author = "Kyle Beyer"
copyright = "2024, Kyle Beyer"

try:
    from rxmc.__version__ import version as release
except ImportError:
    release = "unknown"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_nb",
]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/beykyle/rxmc",
    "use_edit_page_button": False,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}
html_static_path = ["_static"]

# Napoleon settings
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False
napoleon_use_param = False

# Autosummary
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}

# myst-nb — notebooks are pre-executed; never re-run them during the build
nb_execution_mode = "off"

# MathJax — enable $...$ inline and \[...\] display delimiters used in docstrings
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
    }
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

exclude_patterns = ["_build", "**.ipynb_checkpoints", "generated"]
