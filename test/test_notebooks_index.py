"""The example notebooks name the recipes they teach, and every one listed exists.

Design document section 9 item 8: a notebook must cite at least one recipe;
its first cell carries ``Recipes: N, M, ...`` and every number is a heading
of ``docs/recipes.md``.  The committed notebooks are full runs (the
documentation shows them as committed), executed top to bottom.  Nothing
here executes a notebook.
"""

import ast
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
RECIPES = ROOT / "docs" / "recipes.md"
# examples/budget.py's quick-run banner, read from its source so importing it
# here cannot print it
BANNER = re.search(r'^BANNER = "(.+)"$', (EXAMPLES / "budget.py").read_text(), re.M)[1]

# design document section 9, item 8: the eight notebooks and their recipes
NOTEBOOKS = {
    "linear_calibration": {1, 2, 17, 33, 40},
    "normalization_and_covariance_structure": {2, 3, 4, 5, 6, 17, 19, 27},
    "gp_discrepancy": {2, 7, 10, 16, 17, 18, 33, 40},
    "robust_likelihoods": {12, 34, 39},
    "local_optical_model_calibration": {10, 11, 14, 15, 16, 17, 18, 21, 40},
    "alpha_ca_error_model_comparison": {7, 10, 11, 13, 17, 18, 40},
    "hierarchical_calibration": {11, 17, 22, 24, 30, 35, 38, 40, 41},
    "gradient_based_sampling": {7, 16, 33, 38, 41},
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


def _python(source) -> str:
    """The cell's source with IPython magics and shell escapes dropped.

    ``%%time`` and friends are not Python, and ``ast.parse`` chokes on them.
    """
    lines = [ln for ln in source if not ln.lstrip().startswith(("%", "!"))]
    return "".join(lines)


@pytest.mark.parametrize("name", sorted(NOTEBOOKS))
def test_no_latex_escape_lands_in_a_plain_string(name):
    r"""``f"$\rho$"`` is a carriage return, and matplotlib then fails to parse it.

    Every LaTeX macro in a notebook must sit in a raw string, or the Python
    escapes ``\r \t \a \b \f \v`` silently eat the backslash and the label.  An
    ``ast`` walk sees f-strings too, which a ``tokenize`` pass does not.
    """
    control = {
        "\r": r"\r",
        "\t": r"\t",
        "\a": r"\a",
        "\b": r"\b",
        "\f": r"\f",
        "\v": r"\v",
    }
    nb = json.loads((EXAMPLES / f"{name}.ipynb").read_text())
    bad = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse(_python(cell["source"]))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for ch, shown in control.items():
                    if ch in node.value:
                        bad.append(f"cell {i}: {shown} in {node.value[:40]!r}")
    assert not bad, f"{name}: a LaTeX escape outside a raw string: {bad}"


@pytest.mark.parametrize("name", sorted(NOTEBOOKS))
def test_committed_outputs_are_one_full_run(name):
    """Every code cell ran, in order, in one kernel, without an error, and not
    in CI's quick mode: ``python examples/run.py <name>`` makes such a run."""
    nb = json.loads((EXAMPLES / f"{name}.ipynb").read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    counts = [c["execution_count"] for c in code]
    assert counts == list(range(1, len(code) + 1)), (
        f"{name}: execution counts {counts} are not one top-to-bottom run; "
        f"rerun it with examples/run.py"
    )
    for i, cell in enumerate(code):
        for out in cell["outputs"]:
            assert out["output_type"] != "error", f"{name}: code cell {i} failed"
            text = "".join(out.get("text", [])) + "".join(
                out.get("data", {}).get("text/plain", [])
            )
            assert BANNER not in text, f"{name}: committed from a quick run"


def test_no_stray_notebooks():
    stray = sorted(set(_present()) - set(NOTEBOOKS))
    assert not stray, f"notebooks not in the design's list: {stray}"
