# Copilot instructions

This repository treats validation as part of the implementation contract.

## Required local validation

Install the validation dependencies before making substantive changes:

```bash
pip install -ve '.[validation]'
```

Then run the same checks that CI runs for pull requests into `main`:

```bash
python -m isort --check-only src test
python -m black --check src test
python -m ruff check src test
python -m nbqa isort --check examples/*.ipynb
python -m black --check --ipynb examples/*.ipynb
python -m nbqa ruff examples/*.ipynb
python -m pytest
```

## Formatting expectations

- `src/` and `test/` must remain formatted with `isort` and `black`.
- `examples/` notebooks must remain clean under `nbqa isort`, native Black
  notebook formatting (`black --ipynb`), and `nbqa ruff`.
- Only used imports should remain in both Python modules and notebook code cells.

## Notebook expectations

- All notebooks in `examples/` are part of the validation surface.
- Notebook changes are not complete unless `python -m pytest examples` passes.
- If a notebook needs extra runtime dependencies, they must be included in the
  `examples` optional dependency group in `pyproject.toml`.

## Packaging expectations

- Example runtime dependencies belong in the `examples` optional extra.
- Validation tools belong in the `validation` optional extra.
- Keep the README and CI workflow aligned with the actual validation commands.
