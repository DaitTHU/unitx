"""Reproducible timings and traced peak allocations for controlled array writes.

Run from a source checkout with its environment: python benchmarks/numpy_writes.py
Results describe this process and machine, not an allocation-free guarantee.
"""

import argparse
import json
import platform
import statistics
import timeit
import tracemalloc

import numpy as np

from unitx import Quantity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=1_000_000)
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--number', type=int, default=20)
    options = parser.parse_args()
    source = Quantity(np.ones(options.size), 'm')
    target = Quantity(np.empty(options.size), 'm')
    converted = Quantity(np.empty(options.size), 'cm')
    cases = {
        'add_out': lambda: np.add(source, source, out=target),
        'converted_out': lambda: np.add(source, source, out=converted),
        'add_inplace': lambda: target.__iadd__(source),
        'scalar_assignment': lambda: target.__setitem__(slice(None), Quantity(1, 'm')),
        'overlapping_assignment': lambda: target.__setitem__(slice(1, None), target[:-1]),
    }
    results = {}
    for name, operation in cases.items():
        operation()
        seconds = (
            statistics.median(
                timeit.repeat(operation, repeat=options.repeat, number=options.number)
            )
            / options.number
        )
        tracemalloc.start()
        operation()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results[name] = {'seconds': seconds, 'peak_bytes': peak}
    print(
        json.dumps(
            {
                'python': platform.python_version(),
                'platform': platform.platform(),
                'numpy': np.__version__,
                'size': options.size,
                'cases': results,
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
