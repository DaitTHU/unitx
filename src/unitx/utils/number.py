from decimal import Decimal
from fractions import Fraction

from ..exceptions import InexactFloatWarning

ZERO = Fraction(0)


def common_fraction(value: int | float | Decimal | Fraction, *,
                    float_max_denominator=1000000,
                    stacklevel=1) -> Fraction:
    '''
    Convert a number to a Fraction.

    - For `int` and `Decimal`, the conversion is always exact.
    - For `float`, the denominator is limited (under 1,000,000 by default).

    The conversion is useful for float results that are _expected_ to be
    rational numbers like 1/3 or 2/5, but may be inexact due to floating-point
    representation. However, such conversion may not always be as expected,
    especially for very large or small numbers like 1e30 or 1e-10.
    Therefore, when inexact conversion occurs, a warning is issued to inform
    the user of the potential inaccuracy.

    >>> Fraction(1.47)
    Fraction(6620291452234629, 4503599627370496)
    >>> common_fraction(1.47)  # InexactFloatWarning
    Fraction(147, 100)
    >>> common_fraction(1/3)  # InexactFloatWarning
    Fraction(1, 3)
    >>> common_fraction(1e30)  # InexactFloatWarning
    Fraction(1000000000000000019884624838656, 1)
    '''
    if isinstance(value, Fraction):
        return value
    frac = Fraction(value)
    if isinstance(value, float):
        frac_limited = frac.limit_denominator(float_max_denominator)
        if frac_limited != frac:
            import warnings
            warnings.warn(
                f'The float number {value!r} = {frac} is likely inexact, so '
                f'it was converted to the approximate Fraction {frac_limited}. '
                'For exact values, use Fraction or Decimal instead.',
                InexactFloatWarning, stacklevel=stacklevel+1)
        return frac_limited
    return frac
