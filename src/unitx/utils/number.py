from decimal import Decimal
from fractions import Fraction

ZERO = Fraction(0)
ONE = Fraction(1)
TWO = Fraction(2)
THREE = Fraction(3)
TEN = Fraction(10)


def common_fraction(value: int | float | Decimal | Fraction) -> Fraction:
    '''
    Convert a number to a Fraction with a small denominator, like 1, 42, -2/3...

    >>> Fraction(1.47)
    Fraction(6620291452234629, 4503599627370496)
    >>> common_fraction(1.47)
    Fraction(147, 100)
    '''
    if isinstance(value, Fraction):
        return value
    frac = Fraction(value)
    return frac.limit_denominator() if isinstance(value, float) else frac


class FloatConversionWarning(UserWarning):
    '''Warning issued when a float is converted to an approximate Fraction.'''

