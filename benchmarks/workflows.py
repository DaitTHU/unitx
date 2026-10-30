"""Time prepared contractions and a circuit solve-to-energy workflow.

Inputs and fixed units are prepared before timing. Physical results are checked
first, and call profiles and allocation tracing run separately from timing.
"""

import argparse
import cProfile
import hashlib
import json
import platform
import statistics
import timeit
import tracemalloc
from pathlib import Path

import numpy as np

import unitx
from unitx import Quantity
from unitx.factor import Factor, factorint, pi
from unitx.si import si


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--number', type=int, default=1000)
    parser.add_argument('--label', default='working tree')
    options = parser.parse_args()
    if options.repeat < 1 or options.number < 1:
        parser.error('--repeat and --number must be positive integers.')

    resistance = Quantity(np.array([[3.0, -1.0], [-1.0, 2.0]]), si.ohm)
    voltage = Quantity(np.array([5.0, 0.0]), si.V)
    time = Quantity(np.linspace(0.0, 2.0, 11), si.s)
    current = Quantity(np.array([2.0, 1.0]), si.A)
    series = np.broadcast_to(current, (time.size, 2))
    transformed_resistance = resistance.to(Factor(2) * pi * si.ohm)
    transformed_voltage = voltage.to(si.mV)

    def circuit(matrix: Quantity, source: Quantity) -> Quantity:
        solution = np.linalg.solve(matrix, source)
        grid = np.broadcast_to(solution, (time.size, 2))
        power = np.einsum('ti,ij,tj->t', grid, matrix, grid)
        return np.trapezoid(power, time).to(si.J)

    cases = {
        'trace': (lambda: np.trace(resistance), 5.0, si.ohm),
        'einsum_view': (lambda: np.einsum('ij->ji', resistance), resistance.value.T, si.ohm),
        'dot': (lambda: np.dot(resistance, current), [5.0, 0.0], si.V),
        'dot_bare_first': (lambda: np.dot(resistance.value, current), [5.0, 0.0], si.A),
        'einsum_power': (
            lambda: np.einsum('ti,ij,tj->t', series, resistance, series),
            np.full(time.size, 10.0),
            si.W,
        ),
        'circuit': (lambda: circuit(resistance, voltage), 20.0, si.J),
        'circuit_scaled': (
            lambda: circuit(transformed_resistance, transformed_voltage),
            20.0,
            si.J,
        ),
        'convert': (lambda: voltage.to(si.mV), [5000.0, 0.0], si.mV),
        'array_expression': (
            lambda: np.exp(-time / (2.0 * si.s)).mean(),
            np.exp(-time.value / 2.0).mean(),
            si.one,
        ),
    }
    results = {}
    for name, (operation, expected, unit) in cases.items():
        np.testing.assert_allclose(operation().value_in(unit), expected, rtol=1e-12, atol=1e-12)
        samples = [
            elapsed / options.number
            for elapsed in timeit.repeat(operation, repeat=options.repeat, number=options.number)
        ]
        tracemalloc.start()
        result = operation()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del result
        profile = cProfile.Profile()
        profile.runcall(operation)
        metadata_calls = sum(
            entry.callcount
            for entry in profile.getstats()
            if not isinstance(entry.code, str)
            and entry.code.co_name == '_from_monomial'
            and Path(entry.code.co_filename).name == 'unit.py'
        )
        results[name] = {
            'seconds': statistics.median(samples),
            'samples': samples,
            'peak_bytes': peak,
            'unit_metadata_calls': metadata_calls,
        }
    source = Path(unitx.__file__).resolve().parent
    print(
        json.dumps(
            {
                'python': platform.python_version(),
                'platform': platform.platform(),
                'numpy': np.__version__,
                'factorization_backend': 'sympy'
                if factorint.__module__.startswith('sympy')
                else 'builtin',
                'repeat': options.repeat,
                'number': options.number,
                'label': options.label,
                'source_sha256': {
                    f'unitx/{path.relative_to(source)}': hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in sorted(source.rglob('*'))
                    if path.suffix in ('.py', '.pyi')
                },
                'cases': results,
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
