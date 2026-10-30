"""Optional NumPy operand validation, normalization and unit alignment.

Shared by operation adapters without importing their dispatch or write handlers.
Magnitude validation remains at each public boundary; array storage can change.
"""

from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np

from .._coordinates.resolve import _Coordinate
from .._coordinates.resolve import convert_point as convert
from .._operands import _Point
from ..exceptions import DimensionError, UnitOperationError
from ..factor import Factor, one
from ..quantity import Quantity
from ..unit import Unit
from .conversion import compatible

_UNITLESS = Unit()


def _wrap(value: Any, unit: Unit | _Coordinate) -> Any:
    return unit.__rmatmul__(value) if isinstance(unit, _Coordinate) else Quantity(value, unit)


def _is_numpy_magnitude(value: Any) -> bool:
    return (
        type(value) is np.ndarray
        or isinstance(value, np.number)
        and type(value).__module__ == 'numpy'
    ) and value.dtype.kind in 'iufc'


def validate_magnitude(value: Any) -> None:
    if not _is_numpy_magnitude(value):
        error_message = 'Only plain numeric ndarrays and NumPy numeric scalars are supported.'
        raise TypeError(error_message)


def astype_magnitude(value: Any, dtype: Any) -> Any:
    _validate_dtype(dtype)
    return value.astype(dtype)


def _is_supported_operand(value: Any) -> bool:
    if isinstance(value, (Quantity, _Point)):
        value = value.value
    return type(value) in (int, float, complex, Decimal, Fraction) or _is_numpy_magnitude(value)


def _numeric_value(value: Any) -> Any:
    # NumPy has no native Decimal/Fraction dtype. Enter its floating domain at
    # this boundary, leaving scalar Quantity arithmetic and constants intact.
    return float(value) if isinstance(value, (Decimal, Fraction)) else value


def scale_magnitude(value: Any, factor: Factor, *, where: Any = True) -> Any:
    if not _is_supported_operand(value):
        error_message = 'NumPy operations require numeric arrays or supported numeric scalars.'
        raise TypeError(error_message)
    value = _numeric_value(value)
    if factor == one:
        return value
    multiplier = factor.value
    if isinstance(multiplier, Fraction):
        multiplier = float(multiplier)
    # Do not pass a Fraction to NumPy (it would create object dtype).
    return np.multiply(value, multiplier, where=where, out=None)


def _operand_parts(value: Any) -> tuple[Any, Unit | _Coordinate]:
    if isinstance(value, (Quantity, _Point)):
        return _numeric_value(value.value), value.unit
    if isinstance(value, (Unit, _Coordinate)):
        return 1, value
    return _numeric_value(value), _UNITLESS


def _linear_parts(value: Any) -> tuple[Any, Unit]:
    if isinstance(value, Quantity):
        return _numeric_value(value.value), value.unit
    if isinstance(value, Unit):
        return 1, value
    if isinstance(value, (_Point, _Coordinate)):
        error_message = 'This operation requires linear quantities or differences.'
        raise UnitOperationError(error_message)
    return _numeric_value(value), _UNITLESS


def _align_operands(left: Any, right: Any) -> tuple[Any, Any, Unit]:
    a, u = _linear_parts(left)
    b, v = _linear_parts(right)
    if not compatible(u, v):
        error_message = 'Cannot operate on incompatible dimensions or logarithmic degrees.'
        raise DimensionError(error_message)
    # Addition retains the first Quantity's unit, even with a bare left input.
    target = u if isinstance(left, Quantity) else v
    return (
        scale_magnitude(a, u.factor / target.factor),
        scale_magnitude(b, v.factor / target.factor),
        target,
    )


def _selection_parts(left: Any, right: Any) -> tuple[Any, Any, Unit | _Coordinate]:
    """Align selection operands without crossing point/difference boundaries."""
    if not all(_is_supported_operand(x) for x in (left, right)):
        error_message = 'Selection requires numeric scalars, ndarrays, quantities or points.'
        raise TypeError(error_message)
    if isinstance(left, _Point) or isinstance(right, _Point):
        if not isinstance(left, _Point) or not isinstance(right, _Point):
            error_message = 'Selection cannot mix points and differences.'
            raise UnitOperationError(error_message)
        if not left.unit._compatible_with(right.unit):
            error_message = 'Selection requires compatible points with the same convention.'
            raise DimensionError(error_message)
        return (
            _numeric_value(left.value),
            _numeric_value(convert(_numeric_value(right.value), right.unit, left.unit)),
            left.unit,
        )
    return _align_operands(left, right)


def _delta_unit(unit: Unit | _Coordinate) -> Unit:
    return unit.delta_unit if isinstance(unit, _Coordinate) else unit


def _validate_dtype(dtype: Any) -> None:
    if dtype is not None and np.dtype(dtype).kind not in 'iufc':
        error_message = 'Only numeric output dtypes are supported.'
        raise TypeError(error_message)
