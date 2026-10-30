"""Time the complete weighted fit recipe, excluding imports and data generation."""

import argparse
import cProfile
import hashlib
import json
import platform
import runpy
import statistics
import timeit
from pathlib import Path

import numpy as np
import scipy

from unitx import Quantity
from unitx.factor import factorint
from unitx.si import si


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--number', type=int, default=10)
    options = parser.parse_args()
    if options.repeat < 1 or options.number < 1:
        parser.error('--repeat and --number must be positive integers.')
    source = Path(__file__).resolve().parents[1] / 'examples/nonlinear_fit.py'
    fit = runpy.run_path(str(source))['fit_decay']
    time = Quantity(np.linspace(0.0, 6.0, 41), si.s)
    signal = 2.5 * si.V * np.exp(-time / (1.7 * si.s))
    sigma = 0.02 * si.V

    finite = fit(time, signal, sigma, analytic_jacobian=False)
    analytic = fit(time, signal, sigma)
    for actual, expected in zip(analytic[:2], finite[:2], strict=True):
        np.testing.assert_allclose(actual.value_in(expected.unit), expected.value, rtol=1e-8)
    np.testing.assert_allclose(analytic[2], finite[2], rtol=1e-7)
    results = {}
    for use_analytic in (False, True):

        def operation() -> None:
            fit(time, signal, sigma, analytic_jacobian=use_analytic)

        samples = [
            elapsed / options.number
            for elapsed in timeit.repeat(operation, repeat=options.repeat, number=options.number)
        ]
        # Count actual callbacks, including finite-difference evaluations omitted
        # from some SciPy nfev reports. Profiling is separate from timing.
        profile = cProfile.Profile()
        profile.runcall(operation)
        calls = {
            entry.code.co_name: entry.callcount
            for entry in profile.getstats()
            if not isinstance(entry.code, str) and entry.code.co_filename == str(source)
        }
        name = 'analytic' if use_analytic else 'finite_difference'
        results[name] = {
            'seconds': statistics.median(samples),
            'samples': samples,
            'residual_calls': calls.get('residual', 0),
            'jacobian_calls': calls.get('decay_jacobian', 0),
        }
    print(
        json.dumps(
            {
                'python': platform.python_version(),
                'platform': platform.platform(),
                'numpy': np.__version__,
                'scipy': scipy.__version__,
                'factorization_backend': (
                    'sympy' if factorint.__module__.startswith('sympy') else 'builtin'
                ),
                'example_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'samples': time.size,
                'repeat': options.repeat,
                'number': options.number,
                'cases': results,
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
