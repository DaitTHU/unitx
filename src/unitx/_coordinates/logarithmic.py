"""Mathematical log mapping, independent of names and registration."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Self, cast

from .._immutable import _Immutable
from .._numerics import Magnitude, is_numpy_object
from .._numerics.scaling import _coordinate_scalar, affine_value, scale_value
from ..dimension import _LOG_DIMENSION, Dimension
from ..exceptions import DimensionError
from ..factor import Factor, one
from ..unit import Unit
from .logarithms import LogKind, _log_convention

if TYPE_CHECKING:
    from ..quantity import Quantity


class _LogMapping(_Immutable):
    __slots__ = ('_reference', '_at', '_delta_unit', '_kind')
    _reference: Quantity[Fraction]
    _at: Fraction
    _delta_unit: Unit
    _kind: LogKind

    @property
    def origin(self) -> Quantity[Fraction]:
        """Evaluate the physical zero reading without changing the defining anchor."""
        from ..quantity import Quantity

        value = self._physical_value(Fraction(0), self.reference.unit)
        return Quantity(_coordinate_scalar(value, 'Origin'), self.reference.unit)

    @property
    def reference(self) -> Quantity[Fraction]:
        return self._reference

    @property
    def at(self) -> Fraction:
        return self._at

    @property
    def delta_unit(self) -> Unit:
        return self._delta_unit

    @property
    def dimension(self) -> Dimension:
        return self.reference.dimension

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _LogMapping):
            return NotImplemented
        return (
            self.reference == other.reference
            and self.at == other.at
            and self.delta_unit == other.delta_unit
            and self.kind == other.kind
        )

    def __hash__(self) -> int:
        return hash((self.reference, self.at, self.delta_unit, self.kind))

    @property
    def kind(self) -> LogKind:
        return self._kind

    def __new__(
        cls,
        reference: Quantity[Magnitude],
        /,
        *,
        delta_unit: Unit,
        kind: LogKind,
        at: int | float | Fraction | Decimal = 0,
    ) -> Self:
        from ..quantity import Quantity

        if not isinstance(reference, Quantity):
            error_message = 'Logarithmic reference must be a physical Quantity.'
            raise TypeError(error_message)
        _log_convention(kind)
        value = _coordinate_scalar(reference.value, 'Reference')
        reading = _coordinate_scalar(at, 'at')
        delta = delta_unit
        if not isinstance(delta, Unit):
            error_message = 'delta_unit must be a multiplicative Unit.'
            raise TypeError(error_message)
        if delta.dimension != _LOG_DIMENSION:
            error_message = 'A logarithmic coordinate requires a pure logarithmic difference unit.'
            raise DimensionError(error_message)
        physical_unit = reference.unit
        if physical_unit.dimension != physical_unit.dimension.physical:
            error_message = 'Reference must be a physical quantity, not a logarithmic difference.'
            raise DimensionError(error_message)
        if value <= 0:
            error_message = 'Logarithmic reference must be positive.'
            raise ValueError(error_message)
        obj = object.__new__(cls)
        object.__setattr__(obj, '_reference', Quantity(value, physical_unit))
        object.__setattr__(obj, '_at', reading)
        object.__setattr__(obj, '_delta_unit', delta)
        object.__setattr__(obj, '_kind', kind)
        return obj

    def _shift_from(self, source: _LogMapping, *, exact: bool = False) -> Any:
        ratio = scale_value(
            source.reference.value / self.reference.value,
            source.reference.unit.factor / self.reference.unit.factor,
        )
        return _log_convention(self.kind).shift(
            cast(Fraction | float, ratio), self.delta_unit.factor, exact=exact
        )

    def _value_in(self, value: Any, target: _LogMapping) -> Any:
        if not isinstance(target, _LogMapping) or self.dimension != target.dimension:
            error_message = 'Logarithmic conversion requires compatible logarithmic coordinates.'
            raise DimensionError(error_message)
        if self == target:
            return value
        a = (
            self.delta_unit.factor
            / target.delta_unit.factor
            * _log_convention(target.kind).coefficient
            / _log_convention(self.kind).coefficient
        )
        offset = target._shift_from(self)
        if self.at or target.at:
            offset += target.at - a.value * self.at
        return affine_value(value, a.value, offset)

    def _physical_value(self, value: Any, target: Unit) -> Any:
        if target.dimension != self.dimension:
            error_message = 'Physical exit requires a compatible physical unit.'
            raise DimensionError(error_message)
        factor = Factor(self.reference.value) * self.reference.unit.factor / target.factor
        relative = affine_value(value, 1, -self.at)
        return scale_value(
            _log_convention(self.kind).to_ratio(relative, self.delta_unit.factor), factor
        )

    def _reading_from(self, value: Any, source: Unit) -> Any:
        if source.dimension != self.dimension:
            error_message = 'Physical input requires a compatible physical unit.'
            raise DimensionError(error_message)
        factor = source.factor / self.reference.unit.factor / Factor(self.reference.value)
        relative = _log_convention(self.kind).to_log(value, factor, self.delta_unit.factor)
        return affine_value(relative, 1, self.at)

    def _compare_values(self, a: Any, b: Any, other: _LogMapping, op: Any) -> Any:
        from ..quantity import Quantity

        assert isinstance(other, _LogMapping)
        if self == other:
            return op(a, b)
        if not is_numpy_object(a) and not is_numpy_object(b):
            try:
                a, b = Fraction(a), Fraction(b)
            except (ValueError, OverflowError):
                pass
            ratio = self.delta_unit.factor / other.delta_unit.factor
            if ratio.decompose()[1] != one:
                if self.reference == other.reference:
                    return op(
                        Quantity(affine_value(a, 1, -self.at), self.delta_unit),
                        Quantity(affine_value(b, 1, -other.at), other.delta_unit),
                    )
                error_message = (
                    'These coordinates require approximate alignment; convert explicitly first.'
                )
                raise TypeError(error_message)
            self._shift_from(other, exact=True)
        return op(a, other._value_in(b, self))
