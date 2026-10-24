# Contributing and validation

Use Python 3.12 or later. Install the checkout into your development environment:

```sh
python -m pip install -e '.[test,numpy]' build twine
```

NumPy is optional at runtime; including it here enables array tests. SymPy is
an optional factorization backend. Changes must also preserve scalar use
without NumPy and the built-in factorization path without SymPy.

## Run checks

```sh
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m pyright --pythonpath /path/to/your/python
```

Pytest collects runtime tests and source doctests. `tests/test_typing.py` runs
Pyright against scalar and optional NumPy consumers, including scalar checks
with NumPy absent. The [typing guide](docs/typing.md) explains the boundaries.
CI covers Python 3.12–3.15, built-in/SymPy factorization and minimum/current NumPy.
An additional CI job builds the distributions and checks their wheel with the
consumers in `.github/consumers/`. It copies those scripts and their independent
Pyright configuration outside the checkout, first checking scalar runtime and
types without NumPy, then installing NumPy and checking array runtime and types.
The release workflow runs the same consumers before uploading distributions.

Run the executable Markdown examples as well:

```sh
python -m doctest README.md docs/*.md docs/catalogues/*.md
```

Use tracked public guides for release validation; local ignored design notes
are not release documentation. Documentation examples should be reproducible
and avoid modifying source files. Notebook source is checked and executed by
the tests when its optional dependencies are present; see the
[notebook policy](examples/README.md#keep-git-history-small).
With NumPy and Matplotlib installed, the tests also execute the README's
`python` code blocks in order and check the blackbody plot's numerical data.

## Keep changes reviewable

Implement the agreed public behavior in code, annotations, tests and docs
together. Keep exact unit algebra distinct from approximate numerical
backends. Include the relevant boundary cases rather than merely checking
that a call succeeds. Preserve unrelated changes in the working tree.

Use focused commits, normally `feat:`, `fix:`, `docs:` or `chore:`, with a
breaking-change marker when appropriate. Features and their regression tests
belong together; release metadata is a separate change. Version-specific
migration guidance belongs in [migration notes](docs/migration.md), rather
than the current runtime API reference.

## Build and release

```sh
python -m build
python -m twine check --strict dist/*
```

Check wheel/sdist contents, version metadata, `py.typed`, and an installed
consumer outside the checkout. Source-tree success alone does not verify
packaging. Use a clean export of the intended commit when separating several
changes into independently reviewable commits.

The [release workflow](.github/workflows/release.yml) validates distributions
and uploads them to PyPI after a published GitHub Release. The tag must match
`v{version}` from pyproject.toml. A pushed tag alone does not trigger upload;
a manual workflow run builds and validates without publishing.

Release verification includes the exact remote commit/tag, the CI matrix,
the Release workflow, and independently available wheel/sdist files on PyPI.
A local tag, build or changelog is not confirmation of publication.
