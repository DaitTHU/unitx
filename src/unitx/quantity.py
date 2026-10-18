from __future__ import annotations

import math
import operator
import sys
from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from typing import Any, Generic, TypeGuard, TypeVar, cast, overload

from ._numerics import Fractionable, Magnitude, Number, Real, as_exponent, as_int_if_integral
from ._unicode import DOT
from .dimension import DIMENSIONLESS, Dimension
from .exceptions import DimensionError
from .factor import ONE, Factor, SymbolicConstant
from .unit import Unit

# Magnitudes are exposed read-only; operations construct new quantities.
T = TypeVar('T', bound=Magnitude, covariant=True)


def _is_fractionable(value: object) -> TypeGuard[Fractionable]:
    return isinstance(value, Fractionable)


def _is_quantity(value: object) -> TypeGuard[Quantity[Magnitude]]:
    return isinstance(value, Quantity)


def _is_nonfinite_real(value: object) -> bool:
    return (isinstance(value, float) and not math.isfinite(value)) or (
        isinstance(value, Decimal) and not value.is_finite()
    )


def _scale_value(value: Magnitude, factor: Factor) -> Magnitude:
    """Apply a unit ratio in the magnitude's numerical domain."""
    if factor == ONE:
        return value
    if isinstance(value, Decimal):
        if not value.is_finite() or value.is_zero():
            return value * 1
        rational, irrational = factor.decompose()
        scaled = Fraction(value) * rational
        if irrational != ONE:
            scaled *= Fraction(Decimal(str(irrational.value)))
        # Round the final value once, avoiding intermediate Decimal range loss.
        return Decimal(scaled.numerator) / Decimal(scaled.denominator)
    return value * cast(Magnitude, factor.value)


class Quantity(Generic[T]):
    """A magnitude paired with a physical unit.

    Construction preserves the magnitude's type. Arithmetic operators act on
    the magnitudes in operand order, using their native promotion, precision,
    rounding, signals, and exceptions. Decimal arithmetic stays Decimal;
    mixing it with float or Fraction in arithmetic is unsupported, as in Python.
    Unit conversion uses native scaling for each magnitude. Rational scales
    preserve exact int/Fraction arithmetic; Decimal scaling combines the value
    and scale exactly before rounding once in the active Decimal context.
    Irrational scales use float approximations (their decimal strings for
    Decimal magnitudes). An identity conversion preserves the magnitude.
    Multiplication or division by a Unit, Factor, or SymbolicConstant preserves
    the magnitude unchanged. Exact factors are combined with the unit's scale,
    while ordinary numeric operands participate in magnitude arithmetic.

    Magnitudes must be scalars whose equality returns a bool. NumPy scalars,
    arrays, and NumPy operations are not supported yet; NumPy integration is
    planned.

    Addition and subtraction preserve the left quantity's unit. Plain numbers
    can be added to dimensionless quantities and represent unscaled values.
    Scalar operations follow the magnitude's supported operand types.

    Type hints preserve the magnitude type at construction and when combining
    with a Unit or exact factor. Numeric conversions and arithmetic use unions
    where the result depends on runtime units or values. In particular, float
    annotations also accept int inputs, which may produce Fraction after scaling. Custom
    magnitudes retain their arithmetic protocol rather than an assumed Self.

    Comparisons with the same unit factor use the magnitudes directly, including
    native !=, Decimal signals, and unsupported complex ordering. Across unit
    factors, builtin real values and complex components are aligned exactly;
    finite floats retain their stored binary rational values, without tolerance.
    Decimal signals follow the actual comparison operands: native comparisons
    preserve them, while exact cross-factor alignment uses rational coefficients.
    Equality and hashing keep residual irrational factors symbolic. Ordering cancels
    common factors and approximates only a remaining irrational ratio. Factor's
    incomplete factorization limits still apply. Custom magnitudes use native
    coefficient arithmetic and comparisons; unhashable magnitudes remain
    unhashable quantities. Truth values follow bool(magnitude).

    Powers pass the original exponent to the magnitude's native operator. Unit
    exponents use the common formal exponent rules: floats are interpreted by
    their shortest decimal text. Use Fraction for exactly one third, for example.
    Exponents must be finite real scalars other than bool. Native magnitude
    powers can round, produce complex values, or raise for incompatible types.
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

    def __bool__(self) -> bool:
        return bool(self.value)

    @overload
    def value_in(self: Quantity[Decimal], unit: str | Unit, /) -> Decimal: ...

    @overload
    def value_in(self: Quantity[Fraction], unit: str | Unit, /) -> Fraction | float: ...

    @overload
    def value_in[N: Real](self: Quantity[N], unit: str | Unit, /) -> Real: ...

    @overload
    def value_in(self, unit: str | Unit, /) -> Magnitude: ...

    def value_in(self, unit: str | Unit, /) -> Magnitude:
        """Return the magnitude in the specified unit without constructing a Quantity.

        The value and its type follow the same conversion rules as
        ``self.to(unit).value``. An identity scale preserves the magnitude;
        Decimal scaling follows the active context and returns Decimal.
        Irrational conversion factors are evaluated approximately. The original
        quantity is unchanged. Different dimensions raise DimensionError.

        >>> Quantity(1, 'cm').value_in('m')
        Fraction(1, 100)
        >>> Quantity(1, 'km/m').value_in('')
        1000
        >>> Quantity(Decimal('1.25'), 'm').value_in('m')
        Decimal('1.25')
        """
        if not isinstance(unit, Unit):
            unit = Unit(unit)
        if self.dimension != unit.dimension:
            raise DimensionError(
                f'Cannot convert quantity with dimension {self.dimension} '
                f'to unit with dimension {unit.dimension}.'
            )
        conversion_factor = self.unit.factor / unit.factor
        return _scale_value(self.value, conversion_factor)

    @overload
    def to(self: Quantity[Decimal], unit: str | Unit, /) -> Quantity[Decimal]: ...

    @overload
    def to(self: Quantity[Fraction], unit: str | Unit, /) -> Quantity[Fraction | float]: ...

    @overload
    def to[N: Real](self: Quantity[N], unit: str | Unit, /) -> Quantity[Real]: ...

    @overload
    def to(self, unit: str | Unit, /) -> Quantity[Magnitude]: ...

    def to(self, unit: str | Unit, /) -> Quantity[Magnitude]:
        if not isinstance(unit, Unit):
            unit = Unit(unit)
        return Quantity(self.value_in(unit), unit)

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
            symbol, name, *name_aliases, symbol_aliases=symbol_aliases, prefixable=prefixable
        )

    def _comparison_pair(self) -> tuple[Magnitude, Factor]:
        value = self.value
        if _is_fractionable(value):
            try:
                # Compare stored float bits exactly, without tolerance or rounding.
                value = Fraction(value)
            except (ValueError, OverflowError):
                # Positive unit scales do not change infinities or NaNs.
                return value, ONE
        rational, irrational = self.unit.factor.decompose(rationalize_denominator=True)
        if rational != 1:
            value = value * cast(Magnitude, as_int_if_integral(rational))
        if irrational != ONE and value == 0:
            irrational = ONE
        return value, irrational

    def _equality_parts(self) -> tuple[Magnitude, Magnitude, Factor]:
        if isinstance(self.value, complex):
            value = self.value
            real, real_factor = Quantity(value.real, self.unit)._comparison_pair()
            imag, imag_factor = Quantity(value.imag, self.unit)._comparison_pair()
            factor = real_factor if real_factor != ONE else imag_factor
            return real, imag, factor
        value, factor = self._comparison_pair()
        return value, 0, factor

    def _equal(self, other: object, op: Callable[[Any, Any], bool]) -> bool:
        if not _is_quantity(other):
            if not self.is_dimensionless():
                return NotImplemented
            other = Quantity(cast(Magnitude, other), Unit())
        if self.dimension != other.dimension:
            return op is operator.ne
        if self.unit.factor == other.unit.factor:
            return op(self.value, other.value)
        if _is_nonfinite_real(self.value) or _is_nonfinite_real(other.value):
            return op(self.value, other.value)
        real, imag, factor = self._equality_parts()
        other_real, other_imag, other_factor = other._equality_parts()
        if factor != other_factor:
            return op is operator.ne
        if op is operator.eq:
            return op(real, other_real) and op(imag, other_imag)
        return op(real, other_real) or op(imag, other_imag)

    def __eq__(self, other: object) -> bool:
        return self._equal(other, operator.eq)

    def __ne__(self, other: object) -> bool:
        return self._equal(other, operator.ne)

    def __hash__(self) -> int:
        native_hash = hash(self.value)
        if isinstance(self.value, complex) and (
            math.isnan(self.value.real) or math.isnan(self.value.imag)
        ):
            # Native complex NaN hashes depend on the original complex object.
            value_hash, factor = native_hash, ONE
        else:
            real, imag, factor = self._equality_parts()
            value_hash = hash(real)
            if imag != 0:
                # Python's complex hash combines the component hashes and
                # wraps to a signed Py_hash_t; retain exact rational components.
                value_hash += sys.hash_info.imag * hash(imag)
                modulus = 1 << sys.hash_info.width
                value_hash = (value_hash + modulus // 2) % modulus - modulus // 2
                if value_hash == -1:
                    value_hash = -2
        if factor == ONE:
            return value_hash if self.is_dimensionless() else hash((value_hash, self.dimension))
        return hash((value_hash, factor, self.dimension))

    def _compare(self, other: object, op: Callable[[Any, Any], bool]) -> bool:
        if not _is_quantity(other):
            if not self.is_dimensionless():
                return NotImplemented
            other = Quantity(cast(Magnitude, other), Unit())
        if self.dimension != other.dimension:
            raise DimensionError(
                'Cannot compare quantities with different dimensions: '
                f'{self.dimension} and {other.dimension}.'
            )
        if (
            self.unit.factor == other.unit.factor
            or isinstance(self.value, complex)
            or isinstance(other.value, complex)
            or _is_nonfinite_real(self.value)
            or _is_nonfinite_real(other.value)
        ):
            return op(self.value, other.value)
        value, irrational = self._comparison_pair()
        other_value, other_irrational = other._comparison_pair()
        ratio = irrational / other_irrational
        if ratio != ONE:
            if isinstance(value, Fraction) and isinstance(other_value, Fraction):
                if value != 0 and other_value != 0:
                    value *= Fraction(ratio.value)
            else:
                value = _scale_value(value, ratio)
        return op(value, other_value)

    def __lt__(self, other: object, /) -> bool:
        return self._compare(other, operator.lt)

    def __le__(self, other: object, /) -> bool:
        return self._compare(other, operator.le)

    def __gt__(self, other: object, /) -> bool:
        return self._compare(other, operator.gt)

    def __ge__(self, other: object, /) -> bool:
        return self._compare(other, operator.ge)

    def _value_for_isclose(self) -> Fraction | float:
        if not isinstance(self.value, Fractionable):
            raise TypeError('isclose() requires real scalar magnitudes.')
        value, irrational = self._comparison_pair()
        real_value = cast(Fractionable, value)
        try:
            coefficient = Fraction(real_value)
        except (ValueError, OverflowError):
            return float(real_value)
        # Approximate only the residual factor, keeping large/small coefficients exact.
        return coefficient * Fraction(irrational.value)

    def isclose(
        self: Quantity[Fractionable],
        other: Quantity[Fractionable] | Fractionable,
        /,
        *,
        rel_tol: Fractionable = 1e-9,
        abs_tol: Quantity[Fractionable] | Fractionable = 0,
    ) -> bool:
        """Compare real scalar quantities using symmetric relative/absolute tolerances.

        In coherent SI units, the test is
        ``abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)``.
        Both tolerances must be finite and nonnegative. A nonzero absolute
        tolerance for dimensional quantities must be a Quantity with the same
        dimension; plain absolute tolerances and plain other operands represent
        unscaled dimensionless values. Different dimensions raise DimensionError.

        Finite rational coefficients stay exact, including Decimal values and
        floats' stored binary values. Residual irrational unit factors use their
        numerical approximations, independently of the calling quantity's unit.
        With both tolerances zero, the result is exact equality. NaNs are never
        close; infinities are close only to an infinity of the same sign.

        >>> Quantity(0.1 + 0.2, 'm').isclose(Quantity(0.3, 'm'))
        True
        >>> a, b = Quantity(1, 'm'), Quantity(100.05, 'cm')
        >>> a.isclose(b, rel_tol=0, abs_tol=Quantity(1, 'mm'))
        True
        >>> b.isclose(a, rel_tol=0, abs_tol=Quantity(1, 'mm'))
        True
        """
        if not _is_fractionable(rel_tol):
            raise TypeError('rel_tol must be a real scalar.')
        try:
            relative = Fraction(rel_tol)
        except (ValueError, OverflowError):
            raise ValueError('rel_tol must be finite and nonnegative.') from None
        if relative < 0:
            raise ValueError('rel_tol must be finite and nonnegative.')

        if not isinstance(other, Quantity):
            if not _is_fractionable(other):
                raise TypeError('other must be a Quantity or real scalar.')
            other = Quantity(other, '')
        if self.dimension != other.dimension:
            raise DimensionError('Cannot compare quantities with different dimensions.')

        if not isinstance(abs_tol, Quantity):
            if not _is_fractionable(abs_tol):
                raise TypeError('abs_tol must be a Quantity or real scalar.')
            abs_tol = Quantity(abs_tol, self.unit if abs_tol == 0 else Unit())
        if self.dimension != abs_tol.dimension:
            raise DimensionError(
                'Absolute tolerance must have the same dimension as the quantities.'
            )
        absolute = abs_tol._value_for_isclose()
        if (isinstance(absolute, float) and not math.isfinite(absolute)) or absolute < 0:
            raise ValueError('abs_tol must be finite and nonnegative.')

        if not _is_fractionable(self.value) or not _is_fractionable(other.value):
            raise TypeError('isclose() requires real scalar magnitudes.')
        if self == other:
            return True
        if relative == 0 and absolute == 0:
            return False
        value, other_value = self._value_for_isclose(), other._value_for_isclose()
        if isinstance(value, float) or isinstance(other_value, float):
            return False
        return abs(value - other_value) <= max(
            relative * max(abs(value), abs(other_value)), absolute
        )

    @overload
    def __pos__[N: (int, float, Fraction, Decimal, complex)](
        self: Quantity[N], /
    ) -> Quantity[N]: ...

    @overload
    def __pos__(self, /) -> Quantity[Magnitude]: ...

    def __pos__(self, /) -> Quantity[Magnitude]:
        return Quantity(+self.value, self.unit)

    @overload
    def __neg__[N: (int, float, Fraction, Decimal, complex)](
        self: Quantity[N], /
    ) -> Quantity[N]: ...

    @overload
    def __neg__(self, /) -> Quantity[Magnitude]: ...

    def __neg__(self, /) -> Quantity[Magnitude]:
        return Quantity(-self.value, self.unit)

    def _add_sub(
        self,
        other: object,
        op: Callable[[Magnitude, Magnitude], Magnitude],
        *,
        reverse: bool = False,
    ) -> Quantity[Magnitude]:
        if not _is_quantity(other):
            if not self.is_dimensionless():
                return NotImplemented
            other_value = cast(Magnitude, other)
            conversion_factor = 1 / self.unit.factor
        else:
            if self.dimension != other.dimension:
                raise DimensionError(
                    'Cannot operate on quantities with different dimensions: '
                    f'{self.dimension!r} and {other.dimension!r}.'
                )
            other_value = other.value
            conversion_factor = other.unit.factor / self.unit.factor
        # Decimal accepts integers natively. Unit alignment must not turn an
        # otherwise compatible integer into an incompatible Fraction.
        if isinstance(self.value, Decimal) and isinstance(other_value, int):
            other_value = Decimal(other_value)
        other_value = _scale_value(other_value, conversion_factor)
        left, right = (other_value, self.value) if reverse else (self.value, other_value)
        return Quantity(op(left, right), self.unit)

    @overload
    def __add__(
        self: Quantity[Decimal], other: Quantity[Decimal | int] | Decimal | int, /
    ) -> Quantity[Decimal]: ...

    @overload
    def __add__(
        self: Quantity[int], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[Decimal]: ...

    @overload
    def __add__[L: Real, R: Real](
        self: Quantity[L], other: Quantity[R] | R, /
    ) -> Quantity[Real]: ...

    @overload
    def __add__(self, other: Quantity[Magnitude] | Magnitude, /) -> Quantity[Magnitude]: ...

    def __add__(self, other: object, /) -> Quantity[Magnitude]:
        return self._add_sub(other, operator.add)

    @overload
    def __radd__(self: Quantity[Decimal], other: Decimal | int, /) -> Quantity[Decimal]: ...

    @overload
    def __radd__(self: Quantity[int], other: Decimal, /) -> Quantity[Decimal]: ...

    @overload
    def __radd__[N: Real](self: Quantity[N], other: Real, /) -> Quantity[Real]: ...

    @overload
    def __radd__(self, other: Magnitude, /) -> Quantity[Magnitude]: ...

    def __radd__(self, other: object, /) -> Quantity[Magnitude]:
        return self._add_sub(other, operator.add, reverse=True)

    @overload
    def __sub__(
        self: Quantity[Decimal], other: Quantity[Decimal | int] | Decimal | int, /
    ) -> Quantity[Decimal]: ...

    @overload
    def __sub__(
        self: Quantity[int], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[Decimal]: ...

    @overload
    def __sub__[L: Real, R: Real](
        self: Quantity[L], other: Quantity[R] | R, /
    ) -> Quantity[Real]: ...

    @overload
    def __sub__(self, other: Quantity[Magnitude] | Magnitude, /) -> Quantity[Magnitude]: ...

    def __sub__(self, other: object, /) -> Quantity[Magnitude]:
        return self._add_sub(other, operator.sub)

    @overload
    def __rsub__(self: Quantity[Decimal], other: Decimal | int, /) -> Quantity[Decimal]: ...

    @overload
    def __rsub__(self: Quantity[int], other: Decimal, /) -> Quantity[Decimal]: ...

    @overload
    def __rsub__[N: Real](self: Quantity[N], other: Real, /) -> Quantity[Real]: ...

    @overload
    def __rsub__(self, other: Magnitude, /) -> Quantity[Magnitude]: ...

    def __rsub__(self, other: object, /) -> Quantity[Magnitude]:
        return self._add_sub(other, operator.sub, reverse=True)

    @overload
    def __mul__(self, other: Unit | Factor | SymbolicConstant, /) -> Quantity[T]: ...

    @overload
    def __mul__(self: Quantity[int], other: Quantity[int] | int, /) -> Quantity[int]: ...

    @overload
    def __mul__(
        self: Quantity[Decimal], other: Quantity[Decimal | int] | Decimal | int, /
    ) -> Quantity[Decimal]: ...

    @overload
    def __mul__(
        self: Quantity[int], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[Decimal]: ...

    @overload
    def __mul__[L: Real, R: Real](
        self: Quantity[L], other: Quantity[R] | R, /
    ) -> Quantity[Real]: ...

    @overload
    def __mul__(self, other: Quantity[Magnitude] | Magnitude, /) -> Quantity[Magnitude]: ...

    def __mul__(self, other: object, /) -> Quantity[Magnitude]:
        if _is_quantity(other):
            return Quantity(self.value * other.value, self.unit * other.unit)
        if isinstance(other, (Unit, Factor, SymbolicConstant)):
            return Quantity(self.value, self.unit * other)
        return Quantity(self.value * cast(Magnitude, other), self.unit)

    @overload
    def __rmul__(self, other: Unit | Factor | SymbolicConstant, /) -> Quantity[T]: ...

    @overload
    def __rmul__(self: Quantity[int], other: int, /) -> Quantity[int]: ...

    @overload
    def __rmul__(self: Quantity[Decimal], other: Decimal | int, /) -> Quantity[Decimal]: ...

    @overload
    def __rmul__(self: Quantity[int], other: Decimal, /) -> Quantity[Decimal]: ...

    @overload
    def __rmul__[N: Real](self: Quantity[N], other: Real, /) -> Quantity[Real]: ...

    @overload
    def __rmul__(self, other: Magnitude, /) -> Quantity[Magnitude]: ...

    def __rmul__(self, other: object, /) -> Quantity[Magnitude]:
        if isinstance(other, (Unit, Factor, SymbolicConstant)):
            return Quantity(self.value, other * self.unit)
        return Quantity(cast(Magnitude, other) * self.value, self.unit)

    @overload
    def __truediv__(self, other: Unit | Factor | SymbolicConstant, /) -> Quantity[T]: ...

    @overload
    def __truediv__(self: Quantity[int], other: Quantity[int] | int, /) -> Quantity[float]: ...

    @overload
    def __truediv__(
        self: Quantity[Decimal], other: Quantity[Decimal | int] | Decimal | int, /
    ) -> Quantity[Decimal]: ...

    @overload
    def __truediv__(
        self: Quantity[int], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[Decimal]: ...

    @overload
    def __truediv__[L: Real, R: Real](
        self: Quantity[L], other: Quantity[R] | R, /
    ) -> Quantity[Real]: ...

    @overload
    def __truediv__(self, other: Quantity[Magnitude] | Magnitude, /) -> Quantity[Magnitude]: ...

    def __truediv__(self, other: object, /) -> Quantity[Magnitude]:
        if _is_quantity(other):
            return Quantity(self.value / other.value, self.unit / other.unit)
        if isinstance(other, (Unit, Factor, SymbolicConstant)):
            return Quantity(self.value, self.unit / other)
        return Quantity(self.value / cast(Magnitude, other), self.unit)

    @overload
    def __rtruediv__(
        self: Quantity[int], other: Unit | Factor | SymbolicConstant | int, /
    ) -> Quantity[float]: ...

    @overload
    def __rtruediv__(
        self: Quantity[Decimal], other: Unit | Factor | SymbolicConstant | Decimal | int, /
    ) -> Quantity[Decimal]: ...

    @overload
    def __rtruediv__(self: Quantity[int], other: Decimal, /) -> Quantity[Decimal]: ...

    @overload
    def __rtruediv__[N: Real](
        self: Quantity[N], other: Unit | Factor | SymbolicConstant | Real, /
    ) -> Quantity[Real]: ...

    @overload
    def __rtruediv__(
        self, other: Unit | Factor | SymbolicConstant | Magnitude, /
    ) -> Quantity[Magnitude]: ...

    def __rtruediv__(self, other: object, /) -> Quantity[Magnitude]:
        if isinstance(other, (Unit, Factor, SymbolicConstant)):
            return Quantity(1 / self.value, other / self.unit)
        return Quantity(cast(Magnitude, other) / self.value, self.unit**-1)

    @overload
    def __pow__(self: Quantity[Decimal], exponent: int | Decimal, /) -> Quantity[Decimal]: ...

    @overload
    def __pow__(self: Quantity[int], exponent: Decimal, /) -> Quantity[Decimal]: ...

    @overload
    def __pow__[N: Real](self: Quantity[N], exponent: Real, /) -> Quantity[Number]: ...

    @overload
    def __pow__(self, exponent: Fractionable, /) -> Quantity[Magnitude]: ...

    def __pow__(self, exponent: object, /) -> Quantity[Magnitude]:
        if not isinstance(exponent, Fractionable):
            return NotImplemented
        power = as_exponent(exponent)
        return Quantity(self.value**exponent, self.unit**power)


def quantity(value: float, /, unit: str | Unit) -> Quantity[Decimal]:
    """Construct a quantity from a float's shortest decimal representation."""
    return Quantity(Decimal(repr(value)), unit)
