"""Expose static type checks to pytest and VS Code's test explorer."""

import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path


def test_pyright_type_hints() -> None:
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
            sys.executable,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_numpy_type_hints() -> None:
    if find_spec('numpy') is None:
        import pytest

        pytest.skip('NumPy is optional')
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'pyright',
            '--project',
            str(root / 'pyrightconfig.numpy.json'),
            '--pythonpath',
            sys.executable,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
