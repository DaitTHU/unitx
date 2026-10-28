"""Keep scientific examples runnable and notebooks free of generated Git payloads."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ._notebooks import notebook_source

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / 'examples').glob('*.ipynb'))


@pytest.mark.parametrize(
    'guide', ['compatibility', 'creation', 'fft', 'histograms', 'methods', 'writes']
)
def test_numpy_documentation(guide: str) -> None:
    pytest.importorskip('numpy')
    path = Path(__file__).resolve().parents[1] / f'docs/numpy/{guide}.md'
    blocks = re.findall(r'^```python\n(.*?)^```', path.read_text(encoding='utf-8'), re.M | re.S)
    assert blocks, 'Keep runnable Python examples in the NumPy guides.'
    namespace = {'__name__': '__main__'}
    for index, source in enumerate(blocks, start=1):
        exec(compile(source, f'{path}:python block {index}', 'exec'), namespace)


def test_readme_python_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    np = pytest.importorskip('numpy')
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    from unitx import Quantity, Unit

    path = Path(__file__).resolve().parents[1] / 'README.md'
    blocks = re.findall(r'^```python\n(.*?)^```', path.read_text(encoding='utf-8'), re.M | re.S)
    assert blocks, 'Keep runnable Python examples in the README.'
    namespace = {'__name__': '__main__'}
    monkeypatch.setattr(plt, 'show', lambda: None)
    try:
        # Later blocks may reuse imports and variables introduced earlier.
        for index, source in enumerate(blocks, start=1):
            exec(compile(source, f'{path}:python block {index}', 'exec'), namespace)

        radiance = namespace['radiance']
        assert isinstance(radiance, Quantity) and radiance.shape == (100, 3)
        assert radiance.unit == Unit('kW/sr/m2/nm')
        assert np.all(np.isfinite(radiance.value)) and np.all(radiance.value >= 0)
        lines = namespace['ax'].get_lines()
        assert len(lines) == 3
        for index, line in enumerate(lines):
            np.testing.assert_array_equal(line.get_xdata(), namespace['wavelength'].value[:, 0])
            np.testing.assert_array_equal(line.get_ydata(), radiance.value[:, index])
    finally:
        plt.close('all')


@pytest.mark.parametrize('path', NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_clean_source(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding='utf-8'))
    assert not notebook['metadata'].get('widgets'), 'Clear saved widget state before committing.'
    for cell in notebook['cells']:
        assert not cell.get('attachments'), 'Keep generated images out of notebooks in Git.'
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            assert r'\[' not in source and r'\]' not in source, 'Use $$ for display equations.'
        if cell['cell_type'] == 'code':
            assert cell['outputs'] == [], 'Clear notebook outputs before committing.'
            assert cell['execution_count'] is None, (
                'Clear notebook execution counts before committing.'
            )


@pytest.mark.parametrize(
    'path', [path for path in NOTEBOOKS if path.stem != 'blackbody'], ids=lambda path: path.stem
)
def test_notebook_physics(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip('numpy')
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    # Match Jupyter's access to companion modules beside the notebook.
    monkeypatch.syspath_prepend(str(path.parent))
    monkeypatch.setattr(plt, 'show', lambda: None)
    try:
        exec(compile(notebook_source(path), str(path), 'exec'), {'__name__': '__main__'})
    finally:
        plt.close('all')


def test_nonlinear_fit_example() -> None:
    pytest.importorskip('scipy')
    path = Path(__file__).resolve().parents[1] / 'examples/nonlinear_fit.py'
    exec(compile(path.read_text(), str(path), 'exec'), {'__name__': '__main__'})


@pytest.mark.parametrize('guide', ['array-storage', 'scientific-boundaries'])
def test_scientific_documentation(guide: str) -> None:
    pytest.importorskip('numpy')
    path = Path(__file__).resolve().parents[1] / f'docs/{guide}.md'
    blocks = re.findall(r'^```python\n(.*?)^```', path.read_text(), re.M | re.S)
    assert blocks
    namespace = {'__name__': '__main__'}
    for source in blocks:
        exec(compile(source, str(path), 'exec'), namespace)
