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
Before publishing optional-backend or documentation changes, also run the full
pytest suite and Markdown doctests in a clean environment installed with
`.[test]` and no NumPy. Confirm `find_spec('numpy') is None`. This checks source
doctest collection as well as examples; installed-wheel scalar consumers do
not replace it. NumPy backend modules must remain excluded from source doctest
collection when NumPy is absent.
Installed-wheel CI jobs cover minimum and current optional dependencies using
the consumers in `.github/consumers/`. Each job copies those scripts and their
independent Pyright configuration outside the checkout, first checking scalar
runtime and types without NumPy, then installing NumPy and checking array
runtime and types. It then installs Matplotlib and checks the optional axis
converter, followed by the SciPy integration consumer. The minimum job uses
NumPy 2.1, Matplotlib 3.9 and SciPy 1.14; the current job uses current versions.
The release workflow runs the same consumers with current dependencies before
uploading distributions. Source tests also cover both dependency combinations.

`tests/numpy/test_numpy_workflows.py` checks complete coordinate-statistics,
circuit-solve and sampled-spectrum flows. It verifies physical results across
unit choices, reconstruction/energy identities and staged output writes,
including overlapping storage and unchanged buffers on failure.
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

Build release artifacts from a clean checkout or Git export of the intended
commit. Git's local `.git/info/exclude` is not a distribution manifest:
building a shared working tree can include locally ignored notes in an sdist.
Check wheel/sdist contents, version metadata, `py.typed`, and installed
consumers outside the checkout. Source-tree success alone does not verify
packaging. When separating changes, also validate the actual committed
snapshots rather than only the final working tree.

The [release workflow](.github/workflows/release.yml) validates distributions
and uploads them to PyPI after a published GitHub Release. The tag must match
`v{version}` from pyproject.toml. A pushed tag alone does not trigger upload;
a manual workflow run builds and validates without publishing.

Release verification includes the exact remote commit/tag, the CI matrix,
the Release workflow, and independently available wheel/sdist files on PyPI.
A local tag, build or changelog is not confirmation of publication.
