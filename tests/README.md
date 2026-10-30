# Test layout

Runtime tests are grouped by responsibility:

- `algebra/`: dimensions, exact factors, monomials and numerical helpers.
- `units/`: unit identity, definitions, parsing, registries and catalogue collection.
- `quantities/`: linear quantities, affine/logarithmic coordinates and boundaries.
- `catalogues/`: built-in unit systems and spellings.
- `numpy/`: optional array behavior, including standard-library and CGS integration.
- `display/`: text, Unicode and export formatting.
- `stdlib/`: scalar mathematics, statistics, time and optional SciPy adapters.

Cross-cutting import, immutability, serialization, warning, plotting, example and typing
checks remain at the root. `typing/` and `numpy_typing/` contain static consumers
run by `test_typing.py`; pytest does not collect them as runtime tests.

Run all checks with `python -m pytest -q`, or select a group such as
`python -m pytest -q tests/numpy`. Source doctests are collected separately
from `src/unitx`. See [contributing](../CONTRIBUTING.md) for the full matrix.
