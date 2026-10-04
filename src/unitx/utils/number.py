from decimal import Decimal
from fractions import Fraction

ZERO = Fraction(0)


def common_fraction(value: int | float | Decimal | Fraction, *,
                    float_max_denominator=1000000,
                    stacklevel=1) -> Fraction:
    '''
    Convert a number to a Fraction with a small denominator, like 1, 42, -2/3...

    - For `int` and `Decimal`, the conversion is accurate.
    - For `float`, the denominator is limited (under 1,000,000 by default).

    >>> Fraction(1.47)
    Fraction(6620291452234629, 4503599627370496)
    >>> common_fraction(1.47)
    Fraction(147, 100)
    '''
    if isinstance(value, Fraction):
        return value
    frac = Fraction(value)
    if isinstance(value, float):
        frac_limited = frac.limit_denominator(float_max_denominator)
        if frac_limited != frac:
            import warnings
            warnings.warn(
                f'Converting float {value!r} to approximate Fraction {frac_limited}, use Fraction for exact values.',
                FloatConversionWarning, stacklevel=stacklevel+1)
        return frac_limited
    return frac


class FloatConversionWarning(UserWarning):
    '''Warning issued when a float is converted to an approximate Fraction.'''
