"""Explicit integration of the two coordinate families; no shared semantic base."""

from __future__ import annotations

from typing import Any, TypeAlias, cast

from .._operands import _LinearUnitSource
from ..affineunit import AffineUnit
from ..exceptions import DimensionError
from ..factor import one
from ..logunit import LogUnit
from ..unit import Unit
from .affine import _AffineMapping
from .definition import _CoordinateDefinition

_Coordinate: TypeAlias = AffineUnit | LogUnit


def coordinate_from_definition(definition: _CoordinateDefinition) -> _Coordinate:

    cls = AffineUnit if isinstance(definition._mapping, _AffineMapping) else LogUnit
    return cls._from_definition(definition)


def as_coordinate(unit: str | Unit | _LinearUnitSource) -> _Coordinate:
    """Resolve named coordinate metadata, or select a zero-origin affine coordinate."""
    from ..quantity import Quantity

    if isinstance(unit, _Coordinate):
        return unit
    linear = Unit(unit) if isinstance(unit, str) else unit
    if not isinstance(linear, Unit):
        error_message = 'A point coordinate must be a coordinate descriptor, Unit, or spelling.'
        raise TypeError(error_message)
    if linear.scale == one:
        components = tuple(linear._monomial.components())
        if len(components) == 1:
            base, exponent = components[0]
            if (
                exponent == 1
                and not base.has_prefix()
                and isinstance(base._unit, _CoordinateDefinition)
            ):
                return coordinate_from_definition(base._unit)
    if linear.dimension != linear.dimension.physical:
        error_message = 'A logarithmic difference needs an explicit physical reference.'
        raise DimensionError(error_message)
    return AffineUnit(Quantity(0, linear))


def convert_point(value: Any, source: Unit | _Coordinate, target: Unit | _Coordinate) -> Any:
    """Convert matching point or linear operands without crossing their semantics."""
    from .._conversion import convert

    if isinstance(source, _Coordinate) and isinstance(target, _Coordinate):
        if type(source) is not type(target):
            error_message = 'Point conversion requires the same coordinate family.'
            raise DimensionError(error_message)
        return source._mapping._value_in(value, cast(Any, target._mapping))
    if isinstance(source, Unit) and isinstance(target, Unit):
        return convert(value, source, target)
    error_message = 'Point and difference conversion requires an explicit physical boundary.'
    raise TypeError(error_message)
