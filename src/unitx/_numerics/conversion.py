"""Linear unit compatibility and exact magnitude scaling."""

from __future__ import annotations

from ..exceptions import DimensionError
from ..unit import Unit
from . import Magnitude
from .scaling import scale_value


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
        if source.dimension.physical == target.dimension.physical:
            error_message = (
                f'Cannot convert unit {str(source)!r} to unit {str(target)!r}: '
                'the physical dimensions match, but the logarithmic degrees differ. '
                'Unit conversion does not apply a logarithmic or exponential mapping.'
            )
        else:
            error_message = (
                f'Cannot convert quantity with dimension {source.dimension} '
                f'to unit with incompatible algebraic dimension {target.dimension}.'
            )
        raise DimensionError(error_message)
    return scale_value(value, source.factor / target.factor)
