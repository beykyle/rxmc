"""The package installs and reports a version."""

import rxmc


def test_import_and_version():
    assert isinstance(rxmc.__version__, str)
    assert rxmc.__version__
