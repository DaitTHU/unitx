"""Numerical utilities for unitx."""

from __future__ import annotations

import warnings
from decimal import Decimal
from fractions import Fraction
from typing import TypeAlias

from .exceptions import _WARNING_SKIP_PREFIXES, InexactFloatWarning

__all__ = ['Fractionable', 'as_fraction']

Fractionable: TypeAlias = int | float | Decimal | Fraction


def as_fraction(value: Fractionable, /, *, float_max_denominator: int = 1000000) -> Fraction:
    """Convert a number to a Fraction.

    - For `int` and `Decimal`, the conversion is always exact.
    - For `float`, the denominator is limited (under 1,000,000 by default).

    The conversion is useful for float results that are _expected_ to be
    rational numbers like 1/3 or 2/5, but may be inexact due to floating-point
    representation.

    However, such conversion may not always be as expected, especially for
    very large or small numbers like 1e30 or 1e-10. Therefore, when inexact
    conversion occurs, an `InexactFloatWarning` is issued to inform the user
    of the potential inaccuracy.

    >>> Fraction(1.47)  # exact but not expected result
    Fraction(6620291452234629, 4503599627370496)
    >>> as_fraction(1.47)  # InexactFloatWarning
    Fraction(147, 100)
    >>> as_fraction(1 / 3)  # InexactFloatWarning
    Fraction(1, 3)
    >>> as_fraction(1e-10)  # InexactFloatWarning
    Fraction(0, 1)
    >>> as_fraction(1e-10, float_max_denominator=10**12)  # InexactFloatWarning
    Fraction(1, 10000000000)
    >>> as_fraction(1e30)  # no warning, but not expected result
    Fraction(1000000000000000019884624838656, 1)
    """
    if type(value) is Fraction:
        return value
    if not isinstance(value, Fractionable) or type(value) is bool:
        error_message = f'Cannot convert {type(value).__name__} to Fraction.'
        raise TypeError(error_message)
    fraction = Fraction(value)
    if isinstance(value, float):
        limited_fraction = fraction.limit_denominator(float_max_denominator)
        if limited_fraction != fraction:
            warnings.warn(
                f'The denominator of the float number {value!r} = {fraction} is '
                'too big and may be not expected, so it was converted to '
                f'the approximate Fraction {limited_fraction}. '
                'The conversion is inexact and may not be as expected either. '
                'For exact values, use Fraction or Decimal instead.',
                InexactFloatWarning,
                skip_file_prefixes=_WARNING_SKIP_PREFIXES,
            )
        return limited_fraction
    return fraction
