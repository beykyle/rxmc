"""The README's Python blocks run, so the front door cannot drift from the API.

The blocks share one namespace and execute in order: the linear quickstart,
the normalisation term, and the reaction example (a mock measurement, a
short nested-sampling run at ``lmax=10``).
"""

import pathlib
import re

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"
BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def test_readme_python_blocks_execute(capsys):
    blocks = BLOCK.findall(README.read_text())
    assert len(blocks) == 3, "the README carries three Python blocks"
    namespace = {}
    for block in blocks:
        exec(compile(block, str(README), "exec"), namespace)  # noqa: S102
    out = capsys.readouterr().out
    assert "['m', 'b']" in out and "['m', 'b', 'log_eta']" in out
    assert "['V', 'W', 'a']" in out
