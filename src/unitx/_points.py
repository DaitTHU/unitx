"""Numerical point operand alignment, comparison and tolerance helpers."""

from __future__ import annotations

import math
import operator
from decimal import Decimal
from fractions import Fraction
from typing import Any

from ._coordinates import _Coordinate
from ._coordinates import convert_point as convert
from ._numerics import is_numpy_object
from ._operands import _Point
from ._scaling import _finite
from ._scaling import affine_value as _affine
from .exceptions import DimensionError
from .quantity import Quantity
from .unit import Unit


def _addition_parts(
    left: Any, right: Any, *, subtract: bool
) -> tuple[Any, Any, Unit | _Coordinate]:

    if not isinstance(left, _Point):
        if not subtract and isinstance(left, Quantity) and isinstance(right, _Point):
            b, a, unit = _addition_parts(right, left, subtract=False)
            return a, b, unit
        error_message = 'A difference cannot subtract an affine point.'
        raise TypeError(error_message)
    if isinstance(right, _Point):
        if not subtract:
            error_message = 'Two affine points cannot be added.'
            raise TypeError(error_message)
        if not left.unit._compatible_with(right.unit):
            error_message = 'Point subtraction requires the same coordinate convention.'
            raise DimensionError(error_message)
        raw = (
            Decimal(right.value)
            if isinstance(left.value, Decimal) and isinstance(right.value, int)
            else right.value
        )
        return left.value, convert(raw, right.unit, left.unit), left.unit.delta_unit
    if not isinstance(right, Quantity):
        error_message = 'Point arithmetic requires an explicit Quantity difference.'
        raise TypeError(error_message)
    delta = convert(right.value, right.unit, left.unit.delta_unit)
    if isinstance(left.value, Decimal) and isinstance(delta, (int, Fraction)):
        delta = _affine(Decimal(0), 1, delta)
    return left.value, delta, left.unit


def _add_sub(left: Any, right: Any, *, subtract: bool) -> tuple[Any, Unit | _Coordinate]:

    a, b, unit = _addition_parts(left, right, subtract=subtract)
    # Avoid rounding the aligned coordinate before cancellation in a Decimal difference.
    if (
        isinstance(a, Decimal)
        and _finite(a)
        and isinstance(right.value, (Decimal, int))
        and _finite(right.value)
    ):
        target = left.unit if isinstance(right, _Point) else left.unit.delta_unit
        exact = (operator.sub if subtract else operator.add)(
            Fraction(a), convert(Fraction(right.value), right.unit, target)
        )
        value = (
            Decimal(exact.numerator) / Decimal(exact.denominator)
            if isinstance(exact, Fraction)
            else Decimal(str(exact))
        )
    else:
        value = (operator.sub if subtract else operator.add)(a, b)
    return value, unit


def _compare(left: Any, right: Any, op: Any) -> Any:
    equal = op in (operator.eq, operator.ne)
    mismatch = not isinstance(right, _Point) or not left.unit._compatible_with(right.unit)
    if mismatch:
        if equal:
            raw = getattr(right, 'value', right)
            if is_numpy_object(left.value) or is_numpy_object(raw):
                from ._numpy import comparison_constant

                return comparison_constant(left.value, raw, op is operator.ne)
            return op is operator.ne
        error_message = 'Point comparison requires compatible coordinate readings.'
        raise DimensionError(error_message)
    return left.unit._mapping._compare_values(left.value, right.value, right.unit._mapping, op)


def _isclose(left: Any, right: Any, rel_tol: Any, abs_tol: Any, *, equal_nan: bool = False) -> Any:

    if not isinstance(left, _Point) or not isinstance(right, _Point):
        error_message = 'Point isclose requires two coordinate readings.'
        raise TypeError(error_message)
    unit = left.unit
    if not left.unit._compatible_with(right.unit):
        error_message = 'Point comparison requires compatible coordinates.'
        raise DimensionError(error_message)
    if isinstance(rel_tol, Quantity):
        rel_tol = rel_tol.value_in(Unit())
    if is_numpy_object(rel_tol):
        from ._numpy import zero_tolerance

        zero_tolerance(rel_tol)
    elif not isinstance(rel_tol, (int, float, Fraction, Decimal)):
        error_message = 'Relative tolerance must be real.'
        raise TypeError(error_message)
    elif rel_tol != 0:
        error_message = (
            'Point isclose requires rel_tol=0; use a difference abs_tol '
            'or convert to physical units.'
        )
        raise ValueError(error_message)
    if isinstance(abs_tol, Quantity):
        tolerance = convert(abs_tol.value, abs_tol.unit, unit.delta_unit)
    elif is_numpy_object(abs_tol):
        from ._numpy import zero_tolerance

        zero_tolerance(abs_tol)
        tolerance = abs_tol
    elif isinstance(abs_tol, (int, float, Fraction, Decimal)) and abs_tol == 0:
        tolerance = 0
    else:
        error_message = 'Absolute tolerance must be a Quantity difference or real zero.'
        raise TypeError(error_message)
    if not is_numpy_object(tolerance) and not isinstance(
        tolerance, (int, float, Fraction, Decimal)
    ):
        error_message = 'Absolute tolerance must be real.'
        raise TypeError(error_message)
    if not is_numpy_object(tolerance) and tolerance == 0:
        if not any(is_numpy_object(x) for x in (left.value, right.value, rel_tol)):
            return _compare(left, right, operator.eq)
    a = convert(left.value, left.unit, unit)
    b = convert(right.value, right.unit, unit)
    if any(is_numpy_object(x) for x in (a, b, tolerance, rel_tol)):
        from ._numpy import affine_close_magnitude

        return affine_close_magnitude(a, b, tolerance, rel_tol, equal_nan=equal_nan)
    if not _finite(tolerance) or tolerance < 0:
        error_message = 'Absolute tolerance must be finite and nonnegative.'
        raise ValueError(error_message)
    if a == b:
        return True
    if not _finite(a) or not _finite(b):
        return equal_nan and math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tolerance
