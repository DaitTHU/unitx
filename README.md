# unitx

[![PyPI version](https://img.shields.io/pypi/v/unitx)](https://pypi.org/project/unitx/)
[![Python versions](https://img.shields.io/pypi/pyversions/unitx)](https://pypi.org/project/unitx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**A lightweight Python library for physical units, quantities, and dimensional analysis.**

`unitx` provides a small core for composing unit expressions, converting quantities, and working with physical dimensions.
It has no third-party runtime dependencies and is designed around explicit, composable operations.

> **Status: Alpha.** The public API and supported unit set may change between releases.
Please validate the behavior you rely on before using `unitx` in critical scientific or production workflows.

## Features

- **Lightweight installation** — no third-party runtime dependencies.

## Requirements

- Python 3.11 or newer

## Installation

Install the latest release from PyPI:

```bash
python -m pip install unitx
```

## Development

Clone the repository and install the development dependencies:

```bash
git clone https://github.com/DaitTHU/unitx.git
cd unitx

python -m venv .venv
```

Activate the virtual environment, then run:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install -e ".[test]"
pytest
ruff check .
```

## Contributing

Bug reports, focused feature proposals, and pull requests are welcome.

When reporting a bug, include:

- The Python version and `unitx` version.
- A minimal reproducible example.
- The expected result and the actual result.

For changes to unit parsing or arithmetic, please add tests that cover both the intended behavior and relevant edge cases.

## License

`unitx` is distributed under the [MIT License](LICENSE).