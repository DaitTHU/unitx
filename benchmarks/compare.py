"""Compare shared float64 operations, with units and registries prepared first.

NumPy is a numerical reference without unit checks. Quantity construction uses
public, validated constructors; array sharing is checked rather than assumed.
Exact arithmetic and staged-output guarantees are outside this common subset.
"""

import argparse
import importlib.metadata
import json
import platform
import statistics
import timeit
import tracemalloc
from collections.abc import Callable
from typing import Any

import numpy as np


def operations(library: str, size: str) -> tuple[dict[str, Any], bool | None]:
    values = 1.5 if size == 'scalar' else np.linspace(1.0, 2.0, int(size))
    if library == 'numpy':
        metre, centimetre, square_metre, dimensionless = 1, 100, 1, 1

        def construct(value, unit):
            return value

        def convert(q, unit):
            return q * unit

        def read(q, unit):
            return q
    elif library == 'unitx':
        from unitx import Quantity
        from unitx.si import si

        metre, centimetre, square_metre, dimensionless = si.m, si.cm, si.m**2, si.one
        construct = Quantity

        def convert(q, unit):
            return q.to(unit)

        def read(q, unit):
            return q.value_in(unit)
    elif library == 'pint':
        import pint

        registry = pint.UnitRegistry()
        metre, centimetre = registry.meter, registry.centimeter
        square_metre, dimensionless = metre**2, registry.dimensionless
        construct = registry.Quantity

        def convert(q, unit):
            return q.to(unit)

        def read(q, unit):
            return q.to(unit).magnitude
    elif library == 'astropy':
        from astropy import units

        metre, centimetre, square_metre, dimensionless = units.m, units.cm, units.m**2, units.one

        def construct(value, unit):
            return units.Quantity(
                value, unit, copy=False if isinstance(value, np.ndarray) else None
            )

        def convert(q, unit):
            return q.to(unit)

        def read(q, unit):
            return q.to_value(unit)
    elif library == 'unyt':
        import unyt

        metre, centimetre, square_metre = unyt.m, unyt.cm, unyt.m**2
        dimensionless = unyt.Unit('dimensionless')

        def construct(value, unit):
            return (
                unyt.unyt_array(value, unit)
                if isinstance(value, np.ndarray)
                else unyt.unyt_quantity(value, unit)
            )

        def convert(q, unit):
            return q.to(unit)

        def read(q, unit):
            if not isinstance(q, unyt.unyt_array) and unit == dimensionless:
                return q
            return q.to_value(unit)
    else:
        error_message = f'Unknown library: {library}'
        raise ValueError(error_message)
    q = construct(values, metre)
    scaled = construct(values * 100, centimetre)
    denominator = construct(2.0, metre)
    cases = {
        'construct': (lambda: construct(values, metre), values, metre),
        'convert': (lambda: convert(q, centimetre), values * 100, centimetre),
        'add_same': (lambda: q + q, values * 2, metre),
        'add_scaled': (
            lambda: q + scaled if library != 'numpy' else q + scaled / 100,
            values * 2,
            metre,
        ),
        'multiply': (lambda: q * q, values**2, square_metre),
        'exp_ratio': (lambda: np.exp(q / denominator), np.exp(values / 2), dimensionless),
    }
    if size != 'scalar':
        cases['mean'] = (lambda: np.mean(q), np.mean(values), metre)
    for operation, expected, unit in cases.values():
        result = operation()
        # Each result is converted to the expected unit before comparison.
        actual = result if library == 'numpy' else read(result, unit)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)
    raw = q.value if library == 'unitx' else q.magnitude if library == 'pint' else q
    sharing = bool(np.shares_memory(raw, values)) if size != 'scalar' else None
    return {name: case[0] for name, case in cases.items()}, sharing


def measure(operation: Callable[[], Any], repeat: int, minimum: float) -> dict[str, Any]:
    timer = timeit.Timer(operation)
    number = 1
    while timer.timeit(number=number) < minimum:
        number *= 2
    samples = [elapsed / number for elapsed in timer.repeat(repeat=repeat, number=number)]
    tracemalloc.start()
    result = operation()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del result
    return {
        'seconds': statistics.median(samples),
        'samples': samples,
        'number': number,
        'peak_bytes': peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--libraries', nargs='+', default=['numpy', 'unitx'])
    parser.add_argument('--sizes', nargs='+', default=['scalar', '1', '1000', '1000000'])
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--min-time', type=float, default=0.02)
    parser.add_argument('--label', default='working tree')
    options = parser.parse_args()
    if options.repeat < 1 or not np.isfinite(options.min_time) or options.min_time <= 0:
        parser.error('--repeat and --min-time must be positive.')
    if any(size != 'scalar' and (not size.isdecimal() or int(size) < 1) for size in options.sizes):
        parser.error('--sizes accepts scalar or positive array lengths.')
    versions = {'numpy': np.__version__}
    versions.update({name: importlib.metadata.version(name) for name in options.libraries})
    if 'unitx' in options.libraries:
        from unitx.factor import factorint

        backend = 'sympy' if factorint.__module__.startswith('sympy') else 'builtin'
        if backend == 'sympy':
            versions['sympy'] = importlib.metadata.version('sympy')
    else:
        backend = None
    results = {}
    for library in options.libraries:
        results[library] = {}
        for size in options.sizes:
            cases, sharing = operations(library, size)
            results[library][size] = {
                'construction_shares_memory': sharing,
                'cases': {
                    name: measure(operation, options.repeat, options.min_time)
                    for name, operation in cases.items()
                },
            }
    print(
        json.dumps(
            {
                'python': platform.python_version(),
                'platform': platform.platform(),
                'versions': versions,
                'label': options.label,
                'factorization_backend': backend,
                'repeat': options.repeat,
                'min_time': options.min_time,
                'results': results,
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
