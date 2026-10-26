# Contributing and validation

Use Python 3.12 or later. Install the checkout into your development environment:

```sh
python -m pip install -e '.[test,numpy,scipy]' build twine
```

NumPy and SciPy are optional at runtime; including them here enables array
and SciPy integration tests. SymPy is
an optional factorization backend. Changes must also preserve scalar use
without NumPy and the built-in factorization path without SymPy.

## Run checks

```sh
python -m pytest -q
python -m ruff check .
python -m ruff check tests conftest.py --select ANN --ignore ANN401
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
It then installs Matplotlib and checks the optional axis converter with the
installed plotting consumer, then checks the optional SciPy integration consumer. The NumPy minimum job also checks Matplotlib 3.9;
the current job uses current Matplotlib. These jobs also exercise minimum/current
SciPy for the finite jv and Simpson integration support.
The release workflow runs the same consumers before uploading distributions.
Test functions and helpers have annotated parameters and return values. The
dedicated Ruff command checks those signatures; static inference regressions
use the separate Pyright consumers rather than annotating expected results.

Run the executable Markdown examples as well:

```sh
python -m doctest README.md docs/*.md docs/catalogues/*.md docs/numpy/*.md
```

Use tracked public guides for release validation; local ignored design notes
are not release documentation. Documentation examples should be reproducible
and avoid modifying source files. Notebook source is checked and executed by
the tests when its optional dependencies are present; see the
[notebook policy](examples/README.md#keep-git-history-small).
With NumPy and Matplotlib installed, the tests also execute the README's
`python` code blocks in order and check the blackbody plot's numerical data.

## Documentation site

Build or preview the existing Markdown guides with MkDocs:

```sh
python -m pip install -r docs/requirements.txt
python -m mkdocs build --strict
python -m mkdocs serve
```

The build writes to the ignored `site/` directory. CI checks the strict build,
including internal links and anchors. The [documentation site](https://daitthu.github.io/unitx/)
is hosted on GitHub Pages. `.github/workflows/pages.yml` builds and deploys
changes to `docs/`, `mkdocs.yml` or the workflow on main; it also supports
manual runs. Repository Settings → Pages must use GitHub Actions as its source.
`.readthedocs.yaml` retains the same pinned MkDocs requirement for a future
Read the Docs deployment. Do not connect or publish that project until
ownership of `unitx.readthedocs.io` has transferred to the project owner.
Mermaid diagrams load a pinned JavaScript module from jsDelivr in the browser;
building the HTML does not require that CDN. Repository files outside `docs/`
use GitHub links so they also work on a hosted documentation site.

## Performance measurements

Run the [scalar and NumPy benchmarks](benchmarks/README.md) when investigating
performance changes. Their JSON reports record the environment and per-call
timings; compare runs under the same conditions rather than setting CI timing
thresholds.

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
