"""Expose static type checks to pytest and VS Code's test explorer."""

import os
import shlex
import subprocess
import sys
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

import pytest

from ._notebooks import notebook_source


def _check_type_hints(*paths: str, pythonpath: str = sys.executable) -> None:
    assert find_spec('pyright') is not None, (
        f'Pyright is not installed for {sys.executable}.\n'
        f'Install it with: "{sys.executable}" -m pip install "pyright>=1.1.414"'
    )
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'pyright',
            '--project',
            str(root / 'pyrightconfig.json'),
            '--pythonpath',
            pythonpath,
            *paths,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_pyright_type_hints() -> None:
    # Full optional NumPy annotations need NumPy; scalar consumers do not.
    if find_spec('numpy') is None:
        _check_type_hints('tests/typing')
    else:
        _check_type_hints()


def test_numpy_type_hints() -> None:
    if find_spec('numpy') is None:
        pytest.skip('NumPy is optional')
    _check_type_hints('tests/numpy_typing')


def test_numpy_example_type_hints(tmp_path: Path) -> None:
    if find_spec('numpy') is None or find_spec('matplotlib') is None:
        pytest.skip('NumPy and Matplotlib are optional')
    examples = Path(__file__).resolve().parents[1] / 'examples'
    for path in examples.glob('*.ipynb'):
        (tmp_path / f'{path.stem}.py').write_text(notebook_source(path), encoding='utf-8')
    _check_type_hints(str(tmp_path))


def test_asarray_type_hints() -> None:
    if find_spec('numpy') is None:
        pytest.skip('NumPy is optional')
    if tuple(map(int, version('numpy').split('.')[:2])) < (2, 5):
        pytest.skip('Older NumPy stubs can infer Any for asarray(...) * unit')
    _check_type_hints('tests/example_typing')


def test_scalar_optional_type_hints(tmp_path: Path) -> None:
    if os.name == 'nt':
        pytest.skip('The isolated interpreter launcher uses a POSIX shell')
    launcher = tmp_path / 'python-no-site'
    launcher.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -I -S "$@"\n')
    launcher.chmod(0o755)
    # Remove site-packages for import resolution without creating an environment
    # or installing dependencies. Pyright itself runs in the normal interpreter.
    subprocess.run(
        [
            str(launcher),
            '-c',
            'import importlib.util; assert importlib.util.find_spec("numpy") is None',
        ],
        check=True,
        timeout=20,
    )
    _check_type_hints('tests/typing', pythonpath=str(launcher))
