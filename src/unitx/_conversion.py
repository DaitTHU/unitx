"""Linear unit compatibility and exact magnitude scaling."""

from __future__ import annotations

from ._numerics import Magnitude
from ._scaling import scale_value
from .exceptions import DimensionError
from .unit import Unit


def compatible(source: Unit, target: Unit) -> bool:
    return source.dimension == target.dimension


def convert(value: Magnitude, source: Unit, target: Unit) -> Magnitude:
    if not isinstance(source, Unit) or not isinstance(target, Unit):
        error_message = (
            'Linear conversion requires Unit operands; '
            'use point conversion methods for coordinates.'
        )
        raise TypeError(error_message)
    if not compatible(source, target):
        error_message = (
            f'Cannot convert quantity with dimension {source.dimension} '
            f'to unit with incompatible algebraic dimension {target.dimension}.'
        )
        raise DimensionError(error_message)
    return scale_value(value, source.factor / target.factor)
