"""Keep physics notebooks reproducible and free of generated Git payloads."""

import json
from pathlib import Path

import pytest

from ._notebooks import notebook_source

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / 'examples').glob('*.ipynb'))


@pytest.mark.parametrize('path', NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_clean_source(path):
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
def test_notebook_physics(path, monkeypatch):
    pytest.importorskip('numpy')
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    monkeypatch.setattr(plt, 'show', lambda: None)
    try:
        exec(compile(notebook_source(path), str(path), 'exec'), {'__name__': '__main__'})
    finally:
        plt.close('all')
