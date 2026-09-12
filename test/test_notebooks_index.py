"""The example notebooks name the recipes they teach, and every one listed exists.

Design document section 9 item 8: a notebook must cite at least one recipe;
its first cell carries ``Recipes: N, M, ...`` and every number is a heading
of ``docs/recipes.md``.  Nothing here executes a notebook.
"""

import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
RECIPES = ROOT / "docs" / "recipes.md"

# design document section 9, item 8: the nine notebooks and their recipes
NOTEBOOKS = {
    "linear_calibration": {1, 17},
    "error_models": {2, 4, 19},
    "sharing_error_models": {5},
    "normalization_and_covariance_structure": {3, 6, 27},
    "gp_discrepancy": {7, 8, 36},
    "robust_likelihoods": {9, 34},
    "local_optical_model_calibration": {12, 14, 15, 16, 21, 26},
    "alpha_ca_error_model_comparison": {10, 11, 13, 18},
    "hierarchical_calibration": {22, 24, 30, 35, 38},
}

RECIPE_LINE = re.compile(r"^Recipes?:\s*(.+)$", re.M)


def _headings() -> set:
    return {int(n) for n in re.findall(r"^## (\d+)\. ", RECIPES.read_text(), re.M)}


def _present() -> list:
    return sorted(p.stem for p in EXAMPLES.glob("*.ipynb")) if EXAMPLES.is_dir() else []


def _cited(name) -> set:
    nb = json.loads((EXAMPLES / f"{name}.ipynb").read_text())
    first = nb["cells"][0]
    assert first["cell_type"] == "markdown", f"{name}: the first cell must be markdown"
    m = RECIPE_LINE.search("".join(first["source"]))
    assert m, f"{name}: the first cell needs a line 'Recipes: N, M, ...'"
    return {int(n) for n in re.findall(r"\d+", m.group(1))}


@pytest.mark.parametrize("name", sorted(NOTEBOOKS))
def test_each_notebook_exists_and_cites_its_recipes(name):
    assert name in _present(), f"examples/{name}.ipynb is in NOTEBOOKS but missing"
    cited = _cited(name)
    assert cited, f"{name} cites no recipe"
    missing = cited - _headings()
    assert not missing, f"{name} cites recipe(s) {sorted(missing)} with no heading"
    assert cited == NOTEBOOKS[name], (
        f"{name} cites {sorted(cited)}; the design's section 9 says "
        f"{sorted(NOTEBOOKS[name])} (update both or neither)"
    )


def test_no_stray_notebooks():
    stray = sorted(set(_present()) - set(NOTEBOOKS))
    assert not stray, f"notebooks not in the design's list: {stray}"
