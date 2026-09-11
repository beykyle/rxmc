"""docs/recipes.md and test/recipes/ cannot drift apart.

Every ``## N. Title`` heading in the recipes document has exactly one test
file ``test/recipes/test_recipe_NN_<slug>.py`` whose docstring quotes the
recipe, and the error-model legend of recipe 18 is the one the tests build.
"""

import pathlib
import re

import pytest

from helpers import STUDY_LEGEND

ROOT = pathlib.Path(__file__).resolve().parents[1]
RECIPES = ROOT / "docs" / "recipes.md"
TEST_DIR = ROOT / "test" / "recipes"

HEADING = re.compile(r"^## (\d+)\. (.+)$", re.M)
FILE = re.compile(r"^test_recipe_(\d+)_[a-z0-9_]+\.py$")
DOCSTRING = re.compile(r'^"""Recipe (\d+): (.+)$')
LEGEND_ROW = re.compile(r"^\| `(\w+)` \| (.+) \|$", re.M)


def _headings() -> dict:
    return {int(n): title for n, title in HEADING.findall(RECIPES.read_text())}


def _files() -> list:
    return sorted(p for p in TEST_DIR.glob("test_recipe_*.py") if FILE.match(p.name))


def _normalise(text: str) -> str:
    """Case, backticks, colons, commas, a trailing stop and spacing do not count."""
    text = text.replace("`", "").replace(":", "").replace(",", "")
    return " ".join(text.lower().rstrip(".").split())


def test_every_heading_has_one_test_file_and_vice_versa():
    headings = _headings()
    numbers = [int(FILE.match(p.name).group(1)) for p in _files()]
    duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
    assert not duplicates, f"more than one test file for recipe(s) {duplicates}"
    missing = sorted(set(headings) - set(numbers))
    stray = sorted(set(numbers) - set(headings))
    assert (
        not missing
    ), f"recipe(s) {missing} in docs/recipes.md have no test/recipes/test_recipe_NN_*.py"
    assert not stray, f"test file(s) for recipe(s) {stray} have no ## NN. heading"


@pytest.mark.parametrize("path", _files(), ids=lambda p: p.name)
def test_each_test_file_quotes_its_recipe(path):
    n = int(FILE.match(path.name).group(1))
    first = path.read_text().splitlines()[0]
    m = DOCSTRING.match(first)
    assert m, f'{path.name} must open with a docstring """Recipe {n}: <title>'
    assert int(m.group(1)) == n, f"{path.name} quotes recipe {m.group(1)}, not {n}"
    heading = _normalise(_headings()[n])
    assert heading.startswith(
        _normalise(m.group(2))
    ), f"{path.name} quotes {m.group(2)!r}; the heading is {_headings()[n]!r}"


def test_the_error_model_legend_matches_the_recipe_table():
    text = RECIPES.read_text()
    start = text.index("## 18. ")
    end = text.index("## 19. ")
    table = {label: desc for label, desc in LEGEND_ROW.findall(text[start:end])}
    assert table, "recipe 18 must carry the error-model legend table"
    # L0t is a likelihood choice, not a covariance form the study builder makes;
    # "custom" is a test-only spelling of the L1 form
    forms = set(table) - {"L0t"}
    built = set(STUDY_LEGEND) - {"custom"}
    assert forms == built, f"table {sorted(forms)} vs tests {sorted(built)}"
    for label in forms:
        assert _normalise(table[label]) == _normalise(
            STUDY_LEGEND[label]
        ), f"{label}: table says {table[label]!r}, tests say {STUDY_LEGEND[label]!r}"
