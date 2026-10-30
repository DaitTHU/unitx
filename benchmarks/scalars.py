"""Time scalar parsing, conversion, comparison and exact factor operations.

Run from an installed checkout: python benchmarks/scalars.py
Each parse creates a new UnitParser; conversion inputs are prepared beforehand.
Results describe this process and machine, not a cross-machine performance claim.
"""

import argparse
import hashlib
import json
import platform
import statistics
import timeit
import tracemalloc
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

import unitx
from unitx import Quantity, Unit, pi
from unitx.factor import Factor, factorint
from unitx.si import si


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', type=int, default=7)
    parser.add_argument('--number', type=int, default=1000)
    parser.add_argument('--label', default='working tree')
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
    length = 1.5 * si.m
    square_metre, square_centimetre = si.m**2, si.cm**2
    force = si.kg * si.m / si.s**2
    scaled = Unit(si.cm, scale=pi)
    half = Fraction(1, 2)
    mixed = Unit('mg*mm/us2', scale=pi)
    other = Unit('mV*ohm', scale=Factor(Fraction(7, 3)))

    def formula_inline_units() -> object:
        return (length * length + Quantity(10000.0, si.cm**2)).value_in(si.m**2)

    def formula_reused_units() -> object:
        return (length * length + Quantity(10000.0, square_centimetre)).value_in(square_metre)

    assert formula_inline_units() == formula_reused_units() == 3.25
    assert force * si.m == Unit('kg*m2/s2')
    assert scaled**half == Unit('cm^(1/2)', scale=Factor(pi) ** half)
    assert mixed * other == Unit('mg*mm*mV*ohm/us2', scale=pi * Factor(Fraction(7, 3)))
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
        'formula_inline_units': formula_inline_units,
        'formula_reused_units': formula_reused_units,
        'unit_square': lambda: si.m**2,
        'unit_fractional_power': lambda: scaled**half,
        'unit_product': lambda: force * si.m,
        'unit_mixed_product': lambda: mixed * other,
        'unit_copy': lambda: Unit(force),
        'unit_empty': lambda: Unit(),
    }
    results = {}
    for name, operation in cases.items():
        operation()
        samples = [
            elapsed / options.number
            for elapsed in timeit.repeat(operation, repeat=options.repeat, number=options.number)
        ]
        tracemalloc.start()
        result = operation()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del result
        results[name] = {
            'seconds': statistics.median(samples),
            'samples': samples,
            'peak_bytes': peak,
        }
    backend = 'sympy' if factorint.__module__.startswith('sympy') else 'builtin'
    source = Path(unitx.__file__).resolve().parent
    print(
        json.dumps(
            {
                'python': platform.python_version(),
                'platform': platform.platform(),
                'factorization_backend': backend,
                'decimal_precision': getcontext().prec,
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
