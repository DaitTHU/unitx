from __future__ import annotations

import operator
from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from typing import Any, Generic, TypeVar, cast, overload

from ._numerics import Fractionable, Magnitude, Number, Real, as_fraction, as_int_if_integral
from ._unicode import DOT
from .dimension import DIMENSIONLESS, Dimension
from .exceptions import DimensionError
from .factor import ONE, Factor, SymbolicConstant
from .unit import Unit

# Magnitudes are exposed read-only; operations construct new quantities.
T = TypeVar('T', bound=Magnitude, covariant=True)


class Quantity(Generic[T]):
    """A magnitude paired with a physical unit.

    Construction preserves the magnitude's type. Conversion uses Factor's
    scaling rules and can change that type. Arithmetic converts finite Decimal
    operands exactly to Fraction; other magnitudes use native arithmetic.
    Multiplication or division by a Unit, Factor, or SymbolicConstant preserves
    the magnitude unchanged. Exact factors are combined with the unit's scale,
    while ordinary numeric operands participate in magnitude arithmetic.

    Magnitudes must be scalars whose equality returns a bool. NumPy scalars,
    arrays, and NumPy operations are not supported yet; NumPy integration is
    planned.

    Addition and subtraction preserve the left quantity's unit. Plain numbers
    can be added to dimensionless quantities and represent unscaled values.
    Scalar multiplication and division accept int, float, Fraction, and Decimal.

    Type hints preserve the magnitude type at construction and when combining
    with a Unit or exact factor. Numeric conversions and arithmetic use unions
    where the result depends on runtime units or values. In particular, float
    annotations also accept int inputs, which may produce Fraction after scaling. Custom
    magnitudes retain their arithmetic protocol rather than an assumed Self.

    Equality and hashing retain the exact rational coefficient and the unit's
    remaining irrational factor. Finite floats are compared as their stored
    binary rational values, without tolerance. Floating-point conversions and
    irrational powers may round, so compare their results approximately when
    appropriate. Ordering cancels common factors before approximate evaluation
    if an irrational ratio remains. Factor's incomplete factorization limits
    still apply. Other magnitude types use native coefficient arithmetic and
    determine whether a quantity is hashable.

    Powers normalize the exponent once using as_fraction(), just as Unit does,
    and use that exponent for both the magnitude and the unit. Nonintegral
    magnitude powers use native numerical evaluation and can be approximate.
    str() displays the magnitude followed by the unit's exact product expression
    without evaluating its scale. Adjacent numeric factors use a multiplication
    dot. repr() uses a reconstructible Unit expression.

    >>> Quantity(1, 'm') + Quantity(50, 'cm')
    Quantity(Fraction(3, 2), Unit('m'))
    >>> (Quantity(1, 'km/m') + 1).to('').value
    Fraction(1001, 1)
    >>> Quantity(1, 'm') == Quantity(100, 'cm')
    True
    """

    __slots__ = ('_value', '_unit')
    _value: T
    _unit: Unit

    def __init__(self, value: T, unit: str | Unit, /) -> None:
        self._value = value
        self._unit = unit if isinstance(unit, Unit) else Unit(unit)

    @property
    def value(self) -> T:
        return self._value

    @property
    def unit(self) -> Unit:
        return self._unit

    @property
    def dimension(self) -> Dimension:
        return self._unit.dimension

    @property
    def _comparison_pair(self) -> tuple[Magnitude, Factor]:
        value = self.value
        if isinstance(self.value, Fractionable):
            try:
                # Float magnitudes retain their exact binary values, unlike exponents.
                value = Fraction(self.value)
            except (ValueError, OverflowError):
                # Positive unit scales do not change infinities or NaNs.
                return value, ONE
        rational, irrational = self.unit.factor.decompose(rationalize_denominator=True)
        value = as_int_if_integral(rational) * value
        if isinstance(value, (Fractionable, complex)) and value == 0:
            irrational = ONE
        return value, irrational

    @property
    def _arithmetic_value(self) -> Fraction | T:
        return Fraction(self.value) if isinstance(self.value, Decimal) else self.value

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.value!r}, {self.unit!r})'

    def __str__(self) -> str:
        unit = str(self.unit)
        if unit == '1':
            return str(self.value)
        if unit.startswith('1/'):
            return f'{self.value}{unit[1:]}'
        separator = DOT if unit[0].isdecimal() else ' '
        return f'{self.value}{separator}{unit}'

    def is_dimensionless(self) -> bool:
        return self.dimension is DIMENSIONLESS

    @overload
    def to(self: Quantity[Fraction], unit: str | Unit, /) -> Quantity[Fraction | float]: ...

    @overload
    def to[N: Fractionable](self: Quantity[N], unit: str | Unit, /) -> Quantity[Real]: ...

    @overload
    def to(self, unit: str | Unit, /) -> Quantity[Magnitude]: ...

    def to(self, unit: str | Unit, /) -> Quantity[Magnitude]:
        if not isinstance(unit, Unit):
            unit = Unit(unit)
        if self.dimension != unit.dimension:
            raise DimensionError(
                f'Cannot convert quantity with dimension {self.dimension} '
                f'to unit with dimension {unit.dimension}.'
            )
        conversion_factor = self.unit.factor / unit.factor
        return Quantity(conversion_factor(self.value), unit)

    def __hash__(self) -> int:
        value, irrational = self._comparison_pair
        if irrational == ONE:
            return hash(value) if self.is_dimensionless() else hash((value, self.dimension))
        return hash((value, irrational, self.dimension))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quantity):
            if self.is_dimensionless():
                value, irrational = self._comparison_pair
                if irrational == ONE:
                    return value == other
            return NotImplemented
        quantity = cast(Quantity[Magnitude], other)
        if self.dimension != quantity.dimension:
            return False
        value, irrational = self._comparison_pair
        other_value, other_irrational = quantity._comparison_pair
        return irrational == other_irrational and value == other_value

    def _compare(self, other: object, op: Callable[[Any, Any], bool]) -> bool:
        other_value: object
        if not isinstance(other, Quantity):
            if not self.is_dimensionless():
                return NotImplemented
            other_value, other_irrational = other, ONE
        else:
            quantity = cast(Quantity[Magnitude], other)
            if self.dimension != quantity.dimension:
                raise DimensionError(
                    'Cannot compare quantities with different dimensions: '
                    f'{self.dimension} and {quantity.dimension}.'
                )
            other_value, other_irrational = quantity._comparison_pair
        value, irrational = self._comparison_pair
        ratio = irrational / other_irrational
        return op(value if ratio == ONE else ratio(value), other_value)

    def __lt__(self, other: object, /) -> bool:
        return self._compare(other, operator.lt)

    def __le__(self, other: object, /) -> bool:
        return self._compare(other, operator.le)

    def __gt__(self, other: object, /) -> bool:
        return self._compare(other, operator.gt)

    def __ge__(self, other: object, /) -> bool:
        return self._compare(other, operator.ge)

    @overload
    def __pos__(self: Quantity[Decimal], /) -> Quantity[Fraction]: ...

    @overload
    def __pos__[N: (int, float, Fraction, complex)](self: Quantity[N], /) -> Quantity[N]: ...

    @overload
    def __pos__(self, /) -> Quantity[Magnitude]: ...

    def __pos__(self, /) -> Quantity[Magnitude]:
        return Quantity(+self._arithmetic_value, self.unit)

    @overload
    def __neg__(self: Quantity[Decimal], /) -> Quantity[Fraction]: ...

    @overload
    def __neg__[N: (int, float, Fraction, complex)](self: Quantity[N], /) -> Quantity[N]: ...

    @overload
    def __neg__(self, /) -> Quantity[Magnitude]: ...

    def __neg__(self, /) -> Quantity[Magnitude]:
        return Quantity(-self._arithmetic_value, self.unit)

    def _add_sub(
        self, other: object, op: Callable[[Magnitude, Magnitude], Magnitude]
    ) -> Quantity[Magnitude]:
        if not isinstance(other, Quantity):
            if self.is_dimensionless():
                other_value = (1 / self.unit.factor)(other)
                return Quantity(op(self._arithmetic_value, other_value), self.unit)
            return NotImplemented
        quantity = cast(Quantity[Magnitude], other)
        if self.dimension != quantity.dimension:
            raise DimensionError(
                'Cannot operate on quantities with different dimensions: '
                f'{self.dimension!r} and {quantity.dimension!r}.'
            )
        conversion_factor = quantity.unit.factor / self.unit.factor
        return Quantity(op(self._arithmetic_value, conversion_factor(quantity.value)), self.unit)

    @overload
    def __add__[L: Fractionable, R: Fractionable](
        self: Quantity[L], other: Quantity[R] | R, /
    ) -> Quantity[Real]: ...

    @overload
    def __add__(self, other: Quantity[Magnitude] | Magnitude, /) -> Quantity[Magnitude]: ...

    def __add__(self, other: object, /) -> Quantity[Magnitude]:
        return self._add_sub(other, operator.add)

    __radd__ = __add__

    @overload
    def __sub__[L: Fractionable, R: Fractionable](
        self: Quantity[L], other: Quantity[R] | R, /
    ) -> Quantity[Real]: ...

    @overload
    def __sub__(self, other: Quantity[Magnitude] | Magnitude, /) -> Quantity[Magnitude]: ...

    def __sub__(self, other: object, /) -> Quantity[Magnitude]:
        return self._add_sub(other, operator.sub)

    @overload
    def __rsub__[N: Fractionable](self: Quantity[N], other: Fractionable, /) -> Quantity[Real]: ...

    @overload
    def __rsub__(self, other: Magnitude, /) -> Quantity[Magnitude]: ...

    def __rsub__(self, other: object, /) -> Quantity[Magnitude]:
        if self.is_dimensionless():
            conversion_factor = 1 / self.unit.factor
            return Quantity(conversion_factor(other) - self._arithmetic_value, self.unit)
        return NotImplemented

    @overload
    def __mul__(self, other: Unit | Factor | SymbolicConstant, /) -> Quantity[T]: ...

    @overload
    def __mul__(self: Quantity[int], other: Quantity[int] | int, /) -> Quantity[int]: ...

    @overload
    def __mul__[L: Fractionable, R: Fractionable](
        self: Quantity[L], other: Quantity[R] | R, /
    ) -> Quantity[Real]: ...

    @overload
    def __mul__(self, other: Quantity[Magnitude] | Fractionable, /) -> Quantity[Magnitude]: ...

    def __mul__(self, other: object, /) -> Quantity[Magnitude]:
        if isinstance(other, Quantity):
            quantity = cast(Quantity[Magnitude], other)
            value = cast(Magnitude, quantity._arithmetic_value)
            return Quantity(self._arithmetic_value * value, self.unit * quantity.unit)
        if isinstance(other, (Unit, Factor, SymbolicConstant)):
            return Quantity(self.value, self.unit * other)
        if isinstance(other, Fractionable):
            value = cast(Magnitude, Fraction(other) if isinstance(other, Decimal) else other)
            return Quantity(self._arithmetic_value * value, self.unit)
        return NotImplemented

    __rmul__ = __mul__

    @overload
    def __truediv__(self, other: Unit | Factor | SymbolicConstant, /) -> Quantity[T]: ...

    @overload
    def __truediv__(self: Quantity[int], other: Quantity[int] | int, /) -> Quantity[float]: ...

    @overload
    def __truediv__[L: Fractionable, R: Fractionable](
        self: Quantity[L], other: Quantity[R] | R, /
    ) -> Quantity[Real]: ...

    @overload
    def __truediv__(self, other: Quantity[Magnitude] | Fractionable, /) -> Quantity[Magnitude]: ...

    def __truediv__(self, other: object, /) -> Quantity[Magnitude]:
        if isinstance(other, Quantity):
            quantity = cast(Quantity[Magnitude], other)
            value = cast(Magnitude, quantity._arithmetic_value)
            return Quantity(self._arithmetic_value / value, self.unit / quantity.unit)
        if isinstance(other, (Unit, Factor, SymbolicConstant)):
            return Quantity(self.value, self.unit / other)
        if isinstance(other, Fractionable):
            value = cast(Magnitude, Fraction(other) if isinstance(other, Decimal) else other)
            return Quantity(self._arithmetic_value / value, self.unit)
        return NotImplemented

    @overload
    def __rtruediv__(
        self: Quantity[int], other: Unit | Factor | SymbolicConstant | int, /
    ) -> Quantity[float]: ...

    @overload
    def __rtruediv__[N: Fractionable](
        self: Quantity[N], other: Unit | Factor | SymbolicConstant | Fractionable, /
    ) -> Quantity[Real]: ...

    @overload
    def __rtruediv__(
        self, other: Unit | Factor | SymbolicConstant | Fractionable, /
    ) -> Quantity[Magnitude]: ...

    def __rtruediv__(self, other: object, /) -> Quantity[Magnitude]:
        if isinstance(other, (Unit, Factor, SymbolicConstant)):
            return Quantity(1 / self._arithmetic_value, other / self.unit)
        if isinstance(other, Fractionable):
            value = Fraction(other) if isinstance(other, Decimal) else other
            return Quantity(value / self._arithmetic_value, self.unit**-1)
        return NotImplemented

    @overload
    def __pow__[N: Fractionable](
        self: Quantity[N], exponent: Fractionable, /
    ) -> Quantity[Number]: ...

    @overload
    def __pow__(self, exponent: Fractionable, /) -> Quantity[Magnitude]: ...

    def __pow__(self, exponent: object, /) -> Quantity[Magnitude]:
        if not isinstance(exponent, Fractionable):
            return NotImplemented
        power = as_fraction(exponent)
        value_power = as_int_if_integral(power)
        return Quantity(self._arithmetic_value**value_power, self.unit**power)

    def define(
        self: Quantity[Fractionable],
        symbol: str,
        /,
        name: str,
        *name_aliases: str,
        symbol_aliases: str | tuple[str, ...] = (),
        prefixable: bool = True,
    ) -> Unit:
        """Register a unit equal to this quantity, using Unit.define's name rules.

        A nonempty primary name is required; additional names are optional
        positional arguments.
        The magnitude must be a positive finite int, Fraction, Decimal, or
        float. Decimal and Fraction scales stay exact. Float scales follow
        Factor's denominator limit and can warn or round to zero and fail;
        use Decimal or Fraction for exact definitions. Registration affects
        this process and leaves this quantity unchanged.
        """
        scale = Factor(self.value)
        return (scale * self.unit).define(
            symbol,
            name,
            *name_aliases,
            symbol_aliases=symbol_aliases,
            prefixable=prefixable,
        )


def quantity(value: float, /, unit: str | Unit) -> Quantity[Decimal]:
    """Construct a quantity from a float's shortest decimal representation."""
    return Quantity(Decimal(repr(value)), unit)
