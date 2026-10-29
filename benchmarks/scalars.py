"""Time scalar parsing, conversion, comparison and exact factor operations.

Run from an installed checkout: python benchmarks/scalars.py
Each parse creates a new UnitParser; conversion inputs are prepared beforehand.
Results describe this process and machine, not a cross-machine performance claim.
"""

import argparse
import json
import platform
import statistics
import timeit
from decimal import Decimal, getcontext
from fractions import Fraction

from unitx import Quantity, Unit, pi
from unitx.factor import Factor, factorint
from unitx.si import si


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--number', type=int, default=1000)
    options = parser.parse_args()
    if options.repeat < 1 or options.number < 1:
        parser.error('--repeat and --number must be positive integers.')

    rational = Quantity(Fraction(1, 3), si.m)
    decimal = Quantity(Decimal('1.25'), si.m)
    equivalent = Quantity(Fraction(100, 3), si.cm)
    greater = Quantity(Fraction(101, 3), si.cm)
    base = Factor(2) * pi
    radical = base ** Fraction(1, 3)
    dimension = Unit('dB*m/s2').dimension
    cases = {
        'parse_expression': lambda: Unit('kg m2/s3'),
        'convert_fraction': lambda: rational.value_in(si.cm),
        'convert_decimal': lambda: decimal.value_in(si.cm),
        'equal_cross_unit': lambda: rational == equivalent,
        'order_cross_unit': lambda: rational < greater,
        'factor_construct': lambda: Factor(Fraction(123456, 1001)),
        'factor_rational_power': lambda: base ** Fraction(1, 3),
        'factor_cancel': lambda: radical**3 / base,
        'dimension_physical': lambda: dimension.physical,
        'dimension_repr': lambda: repr(dimension),
        'dimension_dict': lambda: dimension.as_dict(),
    }
    results = {}
    for name, operation in cases.items():
        operation()
        samples = timeit.repeat(operation, repeat=options.repeat, number=options.number)
        results[name] = {'seconds': statistics.median(samples) / options.number}
    backend = 'sympy' if factorint.__module__.startswith('sympy') else 'builtin'
    print(
        json.dumps(
            {
                'python': platform.python_version(),
                'platform': platform.platform(),
                'factorization_backend': backend,
                'decimal_precision': getcontext().prec,
                'repeat': options.repeat,
                'number': options.number,
                'cases': results,
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
