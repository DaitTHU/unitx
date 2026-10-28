"""Time the complete weighted fit recipe, excluding imports and data generation."""

import argparse
import json
import platform
import runpy
import statistics
import timeit
from pathlib import Path

import numpy as np
import scipy

from unitx import Quantity
from unitx.si import si


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--number', type=int, default=10)
    options = parser.parse_args()
    if options.repeat < 1 or options.number < 1:
        parser.error('--repeat and --number must be positive integers.')
    fit = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'examples/nonlinear_fit.py'))[
        'fit_decay'
    ]
    time = Quantity(np.linspace(0.0, 6.0, 41), si.s)
    signal = 2.5 * si.V * np.exp(-time / (1.7 * si.s))
    sigma = 0.02 * si.V

    def operation() -> None:
        fit(time, signal, sigma)

    operation()
    seconds = (
        statistics.median(timeit.repeat(operation, repeat=options.repeat, number=options.number))
        / options.number
    )
    print(
        json.dumps(
            {
                'python': platform.python_version(),
                'platform': platform.platform(),
                'numpy': np.__version__,
                'scipy': scipy.__version__,
                'samples': time.size,
                'repeat': options.repeat,
                'number': options.number,
                'cases': {'weighted_decay_fit': {'seconds': seconds}},
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
