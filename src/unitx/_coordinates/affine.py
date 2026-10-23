"""Mathematical affine mapping, independent of names and registration."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Self, cast

from .._immutable import _Immutable
from .._numerics import Magnitude, is_numpy_object
from .._scaling import _coordinate_scalar, _finite, affine_value, scale_value
from ..dimension import Dimension
from ..exceptions import DimensionError
from ..factor import one
from ..unit import Unit

if TYPE_CHECKING:
    from ..quantity import Quantity


class _AffineMapping(_Immutable):
    __slots__ = ('_origin', '_delta_unit')
    _origin: Quantity[Fraction]
    _delta_unit: Unit

    @property
    def origin(self) -> Quantity[Fraction]:
        return self._origin

    @property
    def delta_unit(self) -> Unit:
        return self._delta_unit

    @property
    def dimension(self) -> Dimension:
        return self.origin.dimension

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _AffineMapping):
            return NotImplemented
        return self.origin == other.origin and self.delta_unit == other.delta_unit

    def __hash__(self) -> int:
        return hash((self.origin, self.delta_unit))

    def __new__(
        cls,
        reference: Quantity[Magnitude],
        /,
        *,
        at: int | float | Fraction | Decimal = 0,
        delta_unit: Unit,
    ) -> Self:
        from ..quantity import Quantity

        if not isinstance(reference, Quantity):
            error_message = 'Reference must be a physical Quantity.'
            raise TypeError(error_message)
        value = _coordinate_scalar(reference.value, 'Reference')
        reading = _coordinate_scalar(at, 'at')
        physical_unit = reference.unit
        delta = delta_unit
        if not isinstance(delta, Unit):
            error_message = 'delta_unit must be a multiplicative Unit.'
            raise TypeError(error_message)
        if physical_unit.dimension != physical_unit.dimension.physical:
            error_message = 'Reference must be a physical quantity, not a logarithmic difference.'
            raise DimensionError(error_message)
        if delta.dimension != physical_unit.dimension:
            error_message = 'Linear reference and difference must have the same dimension.'
            raise DimensionError(error_message)
        rational, irrational = (delta.factor / physical_unit.factor).decompose()
        if reading and irrational != one:
            error_message = (
                'This defining reading requires a symbolic sum for its origin; '
                'provide the physical origin directly with at=0.'
            )
            raise TypeError(error_message)
        step = reading * rational
        obj = object.__new__(cls)
        object.__setattr__(obj, '_origin', Quantity(value - step, physical_unit))
        object.__setattr__(obj, '_delta_unit', delta)
        return obj

    def _value_in(self, value: Any, target: _AffineMapping) -> Any:
        if not isinstance(target, _AffineMapping) or self.dimension != target.dimension:
            error_message = 'Affine conversion requires compatible linear coordinates.'
            raise DimensionError(error_message)
        if self == target:
            return value
        a = self.delta_unit.factor / target.delta_unit.factor
        b = cast(
            Any, scale_value(self.origin.value, self.origin.unit.factor / target.delta_unit.factor)
        ) - cast(
            Any,
            scale_value(target.origin.value, target.origin.unit.factor / target.delta_unit.factor),
        )
        return affine_value(value, a.value, b)

    def _physical_value(self, value: Any, target: Unit) -> Any:
        if target.dimension != self.dimension:
            error_message = 'Physical exit requires a compatible physical unit.'
            raise DimensionError(error_message)
        return affine_value(
            value,
            (self.delta_unit.factor / target.factor).value,
            scale_value(self.origin.value, self.origin.unit.factor / target.factor),
        )

    def _compare_values(self, a: Any, b: Any, other: _AffineMapping, op: Any) -> Any:
        from ..quantity import Quantity

        assert isinstance(other, _AffineMapping)
        if self == other:
            return op(a, b)
        if is_numpy_object(a) or is_numpy_object(b):
            return op(a, other._value_in(b, self))
        if self.origin == other.origin:
            return op(Quantity(a, self.delta_unit), Quantity(b, other.delta_unit))
        if not _finite(a) or not _finite(b):
            return op(a, b)
        return op(self._exact_physical(a), other._exact_physical(b))

    def _exact_physical(self, value: Any) -> Quantity[Any]:
        from ..quantity import Quantity

        if self.origin.value == 0:
            return Quantity(value, self.delta_unit)
        if value == 0:
            return self.origin
        rational, irrational = (self.delta_unit.factor / self.origin.unit.factor).decompose()
        if irrational != one:
            error_message = (
                'These coordinates require a symbolic sum for exact physical evaluation; '
                'convert explicitly first.'
            )
            raise TypeError(error_message)
        return Quantity(Fraction(value) * rational + Fraction(self.origin.value), self.origin.unit)
