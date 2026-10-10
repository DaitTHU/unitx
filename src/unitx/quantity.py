from __future__ import annotations

import operator
from fractions import Fraction
from typing import Generic, TypeVar

from .dimension import DIMENSIONLESS, Dimension
from .exceptions import DimensionError
from .unit import Unit
from .utils.valuetype import ValueType

T = TypeVar('T', bound=ValueType)


class Quantity(Generic[T]):

    __slots__ = ('_value', '_unit')

    def __init__(self, value: T, unit: str | Unit, /) -> None:
        self._value = value
        self._unit = unit if isinstance(unit, Unit) else Unit(unit)

    @property
    def value(self) -> T: return self._value
    @property
    def unit(self) -> Unit: return self._unit
    @property
    def dimension(self) -> Dimension: return self._unit.dimension
    @property
    def _base_value(self) -> T: return self._unit.factor(self.value)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.value!r}, {self.unit.symbol!r})'

    def __str__(self) -> str:
        if self.unit.symbol.startswith('1'):
            return f'{self.value}{self.unit.symbol[1:]}'
        return f'{self.value} {self.unit}'

    def is_dimensionless(self) -> bool: return self.dimension is DIMENSIONLESS

    def to(self, unit: str | Unit, /) -> Quantity[T]:
        if not isinstance(unit, Unit):
            unit = Unit(unit)
        if self.dimension != unit.dimension:
            raise DimensionError(f'Cannot convert quantity with dimension {self.dimension} to unit with dimension {unit.dimension}.')
        conversion_factor = self.unit.factor / unit.factor
        return Quantity(conversion_factor(self.value), unit)

    def __hash__(self) -> int: return hash((self._base_value, self.dimension))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quantity):
            if self.is_dimensionless():
                return self._base_value == other
            return NotImplemented
        return self.dimension == other.dimension and \
            self._base_value == other._base_value

    @staticmethod
    def _comparison_op(op):
        def compare(self: Quantity, other: object) -> bool:
            if not isinstance(other, Quantity):
                if self.is_dimensionless():
                    return op(self._base_value, other)
                return NotImplemented
            if self.dimension != other.dimension:
                raise DimensionError(f'Cannot compare quantities with different dimensions: {self.dimension} and {other.dimension}.')
            return op(self._base_value, other._base_value)
        return compare

    __lt__ = _comparison_op(operator.lt)
    __le__ = _comparison_op(operator.le)
    __gt__ = _comparison_op(operator.gt)
    __ge__ = _comparison_op(operator.ge)

    @staticmethod
    def _unary_op(op):
        def operate(self: Quantity) -> Quantity:
            return Quantity(op(self.value), self.unit)
        return operate

    __pos__ = _unary_op(operator.pos)
    __neg__ = _unary_op(operator.neg)

    @staticmethod
    def _add_sub(op):
        def operate(self: Quantity, other: object) -> Quantity:
            if not isinstance(other, Quantity):
                if self.is_dimensionless():
                    return Quantity(op(self._base_value, other), self.unit)
                return NotImplemented
            if self.dimension != other.dimension:
                raise DimensionError(f'Cannot operate on quantities with different dimensions: {self.dimension!r} and {other.dimension!r}.')
            conversion_factor = other.unit.factor / self.unit.factor
            return Quantity(op(self.value, conversion_factor(other.value)), self.unit)
        return operate

    __add__ = __radd__ = _add_sub(operator.add)
    __sub__ = _add_sub(operator.sub)

    def __rsub__(self, other):
        if self.is_dimensionless():
            conversion_factor = 1 / self.unit.factor
            return Quantity(conversion_factor(other) - self.value, self.unit)
        return NotImplemented

    @staticmethod
    def _mul_div(op):
        def operate(self: Quantity, other: object) -> Quantity:
            if isinstance(other, Quantity):
                return Quantity(op(self.value, other.value), op(self.unit, other.unit))
            elif isinstance(other, Unit):
                return Quantity(op(self.value, 1), op(self.unit, other))
            elif isinstance(other, (int, float)):
                return Quantity(op(self.value, other), self.unit)
            else:
                return NotImplemented
        return operate

    __mul__ = __rmul__ = _mul_div(operator.mul)
    __truediv__ = _mul_div(operator.truediv)

    def __rtruediv__(self, other):
        return Quantity(other / self.value, self.unit ** -1)

    def __pow__(self, exponent: Fraction | float) -> Quantity:
        return Quantity(self.value ** exponent, self.unit ** exponent)
    
