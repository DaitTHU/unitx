"""Apply exact factors at scalar and array numerical boundaries."""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import Any, cast

from ..factor import Factor, one
from . import Magnitude, Scalar, is_numpy_object


def scale_value(value: Magnitude, factor: Factor) -> Magnitude:
    """Apply a unit ratio in the magnitude's numerical domain."""
    if is_numpy_object(value):
        from .numpy_operands import scale_magnitude

        return cast(Magnitude, scale_magnitude(value, factor))
    value = cast(Scalar, value)
    if factor == one:
        return value
    if isinstance(value, Decimal):
        if not value.is_finite() or value.is_zero():
            return value * 1
        rational, irrational = factor.decompose()
        scaled = Fraction(value) * rational
        if irrational != one:
            scaled *= Fraction(Decimal(str(irrational.value)))
        # Round the final value once, avoiding intermediate Decimal range loss.
        return Decimal(scaled.numerator) / Decimal(scaled.denominator)
    return cast(Scalar, value * cast(Scalar, factor.value))


def _coordinate_scalar(value: object, name: str) -> Fraction:
    """Validate a finite real defining value and preserve its exact stored value."""
    if (
        is_numpy_object(value)
        or isinstance(value, bool)
        or not isinstance(value, (int, float, Fraction, Decimal))
    ):
        error_message = f'{name} must have a real Python scalar magnitude.'
        raise TypeError(error_message)
    if not _finite(value):
        error_message = f'{name} must be finite.'
        raise ValueError(error_message)
    return Fraction(value)


def affine_value(value: Any, a: Any, b: Any) -> Any:
    if a == 1 and b == 0:
        return value
    if is_numpy_object(value):
        from .numpy import affine_magnitude

        return affine_magnitude(value, a, b)
    if isinstance(value, Decimal):
        if not value.is_finite():
            return value * 1
        result = Fraction(value) * Fraction(a) + Fraction(b)
        return Decimal(result.numerator) / Decimal(result.denominator)
    return value * a + b


def _finite(value: Any) -> bool:
    if isinstance(value, Decimal):
        return value.is_finite()
    if isinstance(value, (int, Fraction)):
        return True
    return math.isfinite(value)
