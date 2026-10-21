from __future__ import annotations

import math
import operator
import sys
from collections.abc import Callable, Iterator
from decimal import Decimal
from fractions import Fraction
from typing import Any, Generic, TypeGuard, TypeVar, cast, overload

from ._numerics import (
    Array,
    Fractionable,
    Magnitude,
    Number,
    Real,
    Scalar,
    as_exponent,
    as_int_if_integral,
    is_numpy_object,
)
from ._numpy_typing import NumpyArray, NumpyDType, NumpyValue, WritableArray
from ._unicode import DOT
from .dimension import DIMENSIONLESS, Dimension
from .exceptions import DimensionError
from .factor import Factor, SymbolicConstant, one
from .unit import Unit
from .unitregistry import PrefixPolicy

# Covariance is safe because Quantity never replaces its magnitude or unit.
# NumPy out= and indexed assignment write elements, preserving array type and dtype;
# array aliases were already writable through .value. No arbitrary T is stored.
T = TypeVar('T', bound=Magnitude, covariant=True)


def _is_quantity(value: object) -> TypeGuard[Quantity[Magnitude]]:
    return isinstance(value, Quantity)


def _is_nonfinite_real(value: object) -> bool:
    return (isinstance(value, float) and not math.isfinite(value)) or (
        isinstance(value, Decimal) and not value.is_finite()
    )


def _scale_value(value: Magnitude, factor: Factor) -> Magnitude:
    """Apply a unit ratio in the magnitude's numerical domain."""
    if is_numpy_object(value):
        from ._numpy import scale_magnitude

        return cast(Magnitude, scale_magnitude(value, factor))
    value = cast(Scalar, value)
    if factor == one:
        return value
    if isinstance(value, Decimal):
        if not value.is_finite() or value.is_zero():
            return value * 1
        rational, irrational = factor.decompose()
        scaled = Fraction(value) * rational
        if irrational != one:
            scaled *= Fraction(Decimal(str(irrational.value)))
        # Round the final value once, avoiding intermediate Decimal range loss.
        return Decimal(scaled.numerator) / Decimal(scaled.denominator)
    return cast(Scalar, value * cast(Scalar, factor.value))


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

    Python scalar magnitudes retain their native numerical semantics. Optional
    NumPy support accepts plain numeric ndarrays and NumPy numeric scalars,
    with one unit for the whole magnitude. See docs/numpy.md for supported
    operations, scaling, comparison, and storage rules.
    Explicit NumPy ufunc out= may write an existing ndarray magnitude without
    replacing its object, dtype, or unit. Indexed assignment uses the same fixed
    destination contract and same_kind casting. Augmented arithmetic on ndarrays
    writes back to that same buffer; scalar magnitudes create ordinary results.

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
        if is_numpy_object(value):
            from ._numpy import validate_magnitude

            validate_magnitude(value)
        elif hasattr(value, '__array_interface__') or hasattr(value, '__array_ufunc__'):
            raise TypeError('Unsupported array backend; use a plain numeric ndarray.')
        elif isinstance(value, (list, tuple)):
            raise TypeError('Use an explicit numeric ndarray instead of list/tuple magnitude.')
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

    def __format__(self, format_spec: str) -> str:
        """Format magnitude with ``;text``, ``;latex``, ``;html`` or ``;siunitx``.

        Width/alignment apply to magnitude only. No implicit unit conversion or
        float coercion is performed; magnitude controls supported numeric specs.
        """
        if not format_spec:
            return str(self)
        # A semicolon used as a Python fill character belongs to the numeric spec.
        offset = 2 if len(format_spec) > 1 and format_spec[1] in '<>=^' else 0
        split = format_spec.find(';', offset)
        if split < 0:
            numeric, style = format_spec, 'text'
        else:
            numeric, style = format_spec[:split], format_spec[split + 1 :]
        if style == 'latex':
            return self.to_latex(numeric)
        if style == 'html':
            return self.to_html(numeric)
        if style == 'siunitx':
            return self.to_siunitx(numeric)
        if style != 'text':
            raise ValueError(f'Unknown Quantity display option: {style!r}.')
        if not numeric:
            return str(self)
        from ._display import format_magnitude

        text = format_magnitude(self.value, numeric)
        unit = str(self.unit)
        if unit == '1':
            return text
        from .unit import ScaledUnit

        if isinstance(self.unit, ScaledUnit) and self.unit.scale != one:
            return f'{text} ({unit})'
        if unit.startswith('1/'):
            return f'{text}{unit[1:]}'
        return f'{text} {unit}'

    def to_latex(self, format_spec: str = '', *, fraction: bool = True) -> str:
        """Return a LaTeX fragment with slash or signed-power unit display.

        format_spec applies to magnitude; numeric scientific notation is typeset
        mathematically while custom magnitude strings remain escaped text.
        """
        from ._display import quantity_body

        return quantity_body(self.value, self.unit, format_spec, 'latex', fraction=fraction)

    def to_siunitx(self, format_spec: str = '') -> str:
        """Return a siunitx v3 quantity/number command, without math delimiters.

        Native decimal/scientific notation is parsed by siunitx. Other values
        remain escaped expressions with numerical parsing explicitly disabled.
        """
        from ._display import siunitx_quantity

        return siunitx_quantity(self.value, self.unit, format_spec)

    def to_html(self, format_spec: str = '') -> str:
        """Return escaped HTML preserving magnitude whitespace and line breaks."""
        from ._display import html_span, quantity_body

        return html_span(quantity_body(self.value, self.unit, format_spec, 'html'))

    def _repr_latex_(self) -> str:
        return '$' + self.to_latex() + '$'

    def _repr_html_(self) -> str:
        return self.to_html()

    def is_dimensionless(self) -> bool:
        return self.dimension is DIMENSIONLESS

    def __bool__(self) -> bool:
        return bool(self.value)

    @overload
    def value_in(self: Quantity[Array], unit: str | Unit, /) -> NumpyValue: ...

    @overload
    def value_in(self: Quantity[Decimal], unit: str | Unit, /) -> Decimal: ...

    @overload
    def value_in(self: Quantity[Fraction], unit: str | Unit, /) -> Fraction | float: ...

    @overload
    def value_in[N: Real](self: Quantity[N], unit: str | Unit, /) -> Real: ...

    @overload
    def value_in(self: Quantity[Scalar], unit: str | Unit, /) -> Scalar: ...

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
    def to(self: Quantity[Array], unit: str | Unit, /) -> Quantity[NumpyValue]: ...

    @overload
    def to(self: Quantity[Decimal], unit: str | Unit, /) -> Quantity[Decimal]: ...

    @overload
    def to(self: Quantity[Fraction], unit: str | Unit, /) -> Quantity[Fraction | float]: ...

    @overload
    def to[N: Real](self: Quantity[N], unit: str | Unit, /) -> Quantity[Real]: ...

    @overload
    def to(self: Quantity[Scalar], unit: str | Unit, /) -> Quantity[Scalar]: ...

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
        prefixable: PrefixPolicy = True,
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

    def _comparison_pair(self) -> tuple[Scalar, Factor]:
        value = cast(Scalar, self.value)
        if isinstance(value, Fractionable):
            try:
                # Compare stored float bits exactly, without tolerance or rounding.
                value = Fraction(value)
            except (ValueError, OverflowError):
                # Positive unit scales do not change infinities or NaNs.
                return value, one
        rational, irrational = self.unit.factor.decompose(rationalize_denominator=True)
        if rational != 1:
            value = value * cast(Scalar, as_int_if_integral(rational))
        if irrational != one and value == 0:
            irrational = one
        return cast(Scalar, value), irrational

    def _equality_parts(self) -> tuple[Scalar, Scalar, Factor]:
        if isinstance(self.value, complex):
            value = self.value
            real, real_factor = Quantity(value.real, self.unit)._comparison_pair()
            imag, imag_factor = Quantity(value.imag, self.unit)._comparison_pair()
            factor = real_factor if real_factor != one else imag_factor
            return real, imag, factor
        value, factor = self._comparison_pair()
        return value, 0, factor

    def _equal(self, other: object, op: Callable[[Any, Any], bool]) -> bool:
        if self._has_numpy_operand(other):
            from ._numpy import compare

            return compare(self, other, op)
        if not _is_quantity(other):
            if not self.is_dimensionless():
                return NotImplemented
            other = Quantity(cast(Scalar, other), Unit())
        if self.dimension != other.dimension:
            return op is operator.ne
        if self.unit.factor == other.unit.factor:
            return op(cast(Scalar, self.value), other.value)
        if _is_nonfinite_real(self.value) or _is_nonfinite_real(other.value):
            return op(cast(Scalar, self.value), other.value)
        real, imag, factor = self._equality_parts()
        other_real, other_imag, other_factor = other._equality_parts()
        if factor != other_factor:
            return op is operator.ne
        if op is operator.eq:
            return op(real, other_real) and op(imag, other_imag)
        return op(real, other_real) or op(imag, other_imag)

    # Ordered array overloads refine the open-ended object comparison fallback.
    @overload
    def __eq__(self: Quantity[Array], other: object, /) -> Array: ...

    @overload
    def __eq__(  # pyright: ignore[reportOverlappingOverload]
        self, other: Quantity[Array] | Array, /
    ) -> Array: ...

    @overload
    def __eq__(self: Quantity[Scalar], other: object, /) -> bool: ...

    @overload
    def __eq__(self, other: object, /) -> bool | Array: ...

    def __eq__(self, other: object) -> Any:
        return self._equal(other, operator.eq)

    @overload
    def __ne__(self: Quantity[Array], other: object, /) -> Array: ...

    @overload
    def __ne__(  # pyright: ignore[reportOverlappingOverload]
        self, other: Quantity[Array] | Array, /
    ) -> Array: ...

    @overload
    def __ne__(self: Quantity[Scalar], other: object, /) -> bool: ...

    @overload
    def __ne__(self, other: object, /) -> bool | Array: ...

    def __ne__(self, other: object) -> Any:
        return self._equal(other, operator.ne)

    def __hash__(self) -> int:
        if is_numpy_object(self.value):
            raise TypeError('NumPy quantities are unhashable.')
        native_hash = hash(self.value)
        if isinstance(self.value, complex) and (
            math.isnan(self.value.real) or math.isnan(self.value.imag)
        ):
            # Native complex NaN hashes depend on the original complex object.
            value_hash, factor = native_hash, one
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
        if factor == one:
            return value_hash if self.is_dimensionless() else hash((value_hash, self.dimension))
        return hash((value_hash, factor, self.dimension))

    def _compare(self, other: object, op: Callable[[Any, Any], bool]) -> bool:
        if self._has_numpy_operand(other):
            from ._numpy import compare

            return compare(self, other, op)
        if not _is_quantity(other):
            if not self.is_dimensionless():
                return NotImplemented
            other = Quantity(cast(Scalar, other), Unit())
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
            return op(cast(Scalar, self.value), other.value)
        value, irrational = self._comparison_pair()
        other_value, other_irrational = other._comparison_pair()
        ratio = irrational / other_irrational
        if ratio != one:
            if isinstance(value, Fraction) and isinstance(other_value, Fraction):
                if value != 0 and other_value != 0:
                    value *= Fraction(ratio.value)
            else:
                value = cast(Scalar, _scale_value(value, ratio))
        return op(value, other_value)

    @overload
    def __lt__(self: Quantity[Array], other: object, /) -> Array: ...

    @overload
    def __lt__(  # pyright: ignore[reportOverlappingOverload]
        self, other: Quantity[Array] | Array, /
    ) -> Array: ...

    @overload
    def __lt__(self: Quantity[Scalar], other: object, /) -> bool: ...

    @overload
    def __lt__(self, other: object, /) -> bool | Array: ...

    def __lt__(self, other: object, /) -> Any:
        return self._compare(other, operator.lt)

    @overload
    def __le__(self: Quantity[Array], other: object, /) -> Array: ...

    @overload
    def __le__(  # pyright: ignore[reportOverlappingOverload]
        self, other: Quantity[Array] | Array, /
    ) -> Array: ...

    @overload
    def __le__(self: Quantity[Scalar], other: object, /) -> bool: ...

    @overload
    def __le__(self, other: object, /) -> bool | Array: ...

    def __le__(self, other: object, /) -> Any:
        return self._compare(other, operator.le)

    @overload
    def __gt__(self: Quantity[Array], other: object, /) -> Array: ...

    @overload
    def __gt__(  # pyright: ignore[reportOverlappingOverload]
        self, other: Quantity[Array] | Array, /
    ) -> Array: ...

    @overload
    def __gt__(self: Quantity[Scalar], other: object, /) -> bool: ...

    @overload
    def __gt__(self, other: object, /) -> bool | Array: ...

    def __gt__(self, other: object, /) -> Any:
        return self._compare(other, operator.gt)

    @overload
    def __ge__(self: Quantity[Array], other: object, /) -> Array: ...

    @overload
    def __ge__(  # pyright: ignore[reportOverlappingOverload]
        self, other: Quantity[Array] | Array, /
    ) -> Array: ...

    @overload
    def __ge__(self: Quantity[Scalar], other: object, /) -> bool: ...

    @overload
    def __ge__(self, other: object, /) -> bool | Array: ...

    def __ge__(self, other: object, /) -> Any:
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

    @overload
    def isclose(
        self: Quantity[Array],
        other: object,
        /,
        *,
        rel_tol: object = 1e-9,
        abs_tol: object = 0,
    ) -> Array: ...

    @overload
    def isclose(
        self,
        other: Quantity[Array] | Array,
        /,
        *,
        rel_tol: object = 1e-9,
        abs_tol: object = 0,
    ) -> Array: ...

    @overload
    def isclose(
        self, other: object, /, *, rel_tol: Array | Quantity[Array], abs_tol: object = 0
    ) -> Array: ...

    @overload
    def isclose(
        self, other: object, /, *, rel_tol: object = 1e-9, abs_tol: Array | Quantity[Array]
    ) -> Array: ...

    @overload
    def isclose(
        self: Quantity[Fractionable],
        other: Quantity[Fractionable] | Fractionable,
        /,
        *,
        rel_tol: Fractionable = 1e-9,
        abs_tol: Quantity[Fractionable] | Fractionable = 0,
    ) -> bool: ...

    def isclose(
        self,
        other: object,
        /,
        *,
        rel_tol: object = 1e-9,
        abs_tol: object = 0,
    ) -> Any:
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
        if self._has_numpy_operand(other, rel_tol, abs_tol):
            from ._numpy import isclose

            return isclose(self, other, rel_tol, abs_tol, symmetric=True)
        if not isinstance(rel_tol, Fractionable):
            raise TypeError('rel_tol must be a real scalar.')
        try:
            relative = Fraction(rel_tol)
        except (ValueError, OverflowError):
            raise ValueError('rel_tol must be finite and nonnegative.') from None
        if relative < 0:
            raise ValueError('rel_tol must be finite and nonnegative.')

        if not _is_quantity(other):
            if not isinstance(other, Fractionable):
                raise TypeError('other must be a Quantity or real scalar.')
            other = Quantity(other, '')
        if self.dimension != other.dimension:
            raise DimensionError('Cannot compare quantities with different dimensions.')

        if not isinstance(abs_tol, Quantity):
            if not isinstance(abs_tol, Fractionable):
                raise TypeError('abs_tol must be a Quantity or real scalar.')
            abs_tol = Quantity(abs_tol, self.unit if abs_tol == 0 else Unit())
        if self.dimension != abs_tol.dimension:
            raise DimensionError(
                'Absolute tolerance must have the same dimension as the quantities.'
            )
        absolute = abs_tol._value_for_isclose()
        if (isinstance(absolute, float) and not math.isfinite(absolute)) or absolute < 0:
            raise ValueError('abs_tol must be finite and nonnegative.')

        if not isinstance(self.value, Fractionable) or not isinstance(other.value, Fractionable):
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
    def __pos__(self: Quantity[Array], /) -> Quantity[NumpyValue]: ...

    @overload
    def __pos__[N: (int, float, Fraction, Decimal, complex)](
        self: Quantity[N], /
    ) -> Quantity[N]: ...

    @overload
    def __pos__(self: Quantity[Scalar], /) -> Quantity[Scalar]: ...

    @overload
    def __pos__(self, /) -> Quantity[Magnitude]: ...

    def __pos__(self, /) -> Quantity[Magnitude]:
        if is_numpy_object(self.value):
            from ._numpy import apply_ufunc

            return apply_ufunc('positive', self)
        return Quantity(cast(Scalar, +cast(Scalar, self.value)), self.unit)

    @overload
    def __neg__(self: Quantity[Array], /) -> Quantity[NumpyValue]: ...

    @overload
    def __neg__[N: (int, float, Fraction, Decimal, complex)](
        self: Quantity[N], /
    ) -> Quantity[N]: ...

    @overload
    def __neg__(self: Quantity[Scalar], /) -> Quantity[Scalar]: ...

    @overload
    def __neg__(self, /) -> Quantity[Magnitude]: ...

    def __neg__(self, /) -> Quantity[Magnitude]:
        if is_numpy_object(self.value):
            from ._numpy import apply_ufunc

            return apply_ufunc('negative', self)
        return Quantity(cast(Scalar, -cast(Scalar, self.value)), self.unit)

    def _add_sub(
        self,
        other: object,
        op: Callable[[Scalar, Scalar], object],
        *,
        reverse: bool = False,
    ) -> Quantity[Scalar]:
        if not _is_quantity(other):
            if not self.is_dimensionless():
                return NotImplemented
            other_value = cast(Scalar, other)
            conversion_factor = 1 / self.unit.factor
        else:
            if self.dimension != other.dimension:
                raise DimensionError(
                    'Cannot operate on quantities with different dimensions: '
                    f'{self.dimension!r} and {other.dimension!r}.'
                )
            other_value = cast(Scalar, other.value)
            conversion_factor = other.unit.factor / self.unit.factor
        # Decimal accepts integers natively. Unit alignment must not turn an
        # otherwise compatible integer into an incompatible Fraction.
        if isinstance(self.value, Decimal) and isinstance(other_value, int):
            other_value = Decimal(other_value)
        other_value = cast(Scalar, _scale_value(other_value, conversion_factor))
        value = cast(Scalar, self.value)
        left, right = (other_value, value) if reverse else (value, other_value)
        return Quantity(cast(Scalar, op(left, right)), self.unit)

    @overload
    def __add__(
        self: Quantity[Array], other: Quantity[Magnitude] | Magnitude, /
    ) -> Quantity[NumpyValue]: ...

    @overload
    def __add__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...

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
    def __add__(
        self: Quantity[Scalar], other: Quantity[Scalar] | Scalar, /
    ) -> Quantity[Scalar]: ...

    @overload
    def __add__(self, other: Quantity[Magnitude] | Magnitude, /) -> Quantity[Magnitude]: ...

    def __add__(self, other: object, /) -> Quantity[Magnitude]:
        if self._has_numpy_operand(other) and not isinstance(
            other, (Unit, Factor, SymbolicConstant)
        ):
            from ._numpy import apply_ufunc

            return apply_ufunc('add', self, other)
        return self._add_sub(other, operator.add)

    @overload
    def __radd__(
        self: Quantity[Array], other: Quantity[Magnitude] | Magnitude, /
    ) -> Quantity[NumpyValue]: ...

    @overload
    def __radd__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...

    @overload
    def __radd__(self: Quantity[Decimal], other: Decimal | int, /) -> Quantity[Decimal]: ...

    @overload
    def __radd__(self: Quantity[int], other: Decimal, /) -> Quantity[Decimal]: ...

    @overload
    def __radd__[N: Real](self: Quantity[N], other: Real, /) -> Quantity[Real]: ...

    @overload
    def __radd__(self: Quantity[Scalar], other: Scalar, /) -> Quantity[Scalar]: ...

    @overload
    def __radd__(self, other: Magnitude, /) -> Quantity[Magnitude]: ...

    def __radd__(self, other: object, /) -> Quantity[Magnitude]:
        if self._has_numpy_operand(other) and not isinstance(
            other, (Unit, Factor, SymbolicConstant)
        ):
            from ._numpy import apply_ufunc

            return apply_ufunc('add', other, self)
        return self._add_sub(other, operator.add, reverse=True)

    @overload
    def __sub__(
        self: Quantity[Array], other: Quantity[Magnitude] | Magnitude, /
    ) -> Quantity[NumpyValue]: ...

    @overload
    def __sub__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...

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
    def __sub__(
        self: Quantity[Scalar], other: Quantity[Scalar] | Scalar, /
    ) -> Quantity[Scalar]: ...

    @overload
    def __sub__(self, other: Quantity[Magnitude] | Magnitude, /) -> Quantity[Magnitude]: ...

    def __sub__(self, other: object, /) -> Quantity[Magnitude]:
        if self._has_numpy_operand(other) and not isinstance(
            other, (Unit, Factor, SymbolicConstant)
        ):
            from ._numpy import apply_ufunc

            return apply_ufunc('subtract', self, other)
        return self._add_sub(other, operator.sub)

    @overload
    def __rsub__(
        self: Quantity[Array], other: Quantity[Magnitude] | Magnitude, /
    ) -> Quantity[NumpyValue]: ...

    @overload
    def __rsub__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...

    @overload
    def __rsub__(self: Quantity[Decimal], other: Decimal | int, /) -> Quantity[Decimal]: ...

    @overload
    def __rsub__(self: Quantity[int], other: Decimal, /) -> Quantity[Decimal]: ...

    @overload
    def __rsub__[N: Real](self: Quantity[N], other: Real, /) -> Quantity[Real]: ...

    @overload
    def __rsub__(self: Quantity[Scalar], other: Scalar, /) -> Quantity[Scalar]: ...

    @overload
    def __rsub__(self, other: Magnitude, /) -> Quantity[Magnitude]: ...

    def __rsub__(self, other: object, /) -> Quantity[Magnitude]:
        if self._has_numpy_operand(other) and not isinstance(
            other, (Unit, Factor, SymbolicConstant)
        ):
            from ._numpy import apply_ufunc

            return apply_ufunc('subtract', other, self)
        return self._add_sub(other, operator.sub, reverse=True)

    @overload
    def __mul__(self, other: Unit | Factor | SymbolicConstant, /) -> Quantity[T]: ...

    @overload
    def __mul__(
        self: Quantity[Array], other: Quantity[Magnitude] | Magnitude, /
    ) -> Quantity[NumpyValue]: ...

    @overload
    def __mul__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...

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
    def __mul__(
        self: Quantity[Scalar], other: Quantity[Scalar] | Scalar, /
    ) -> Quantity[Scalar]: ...

    @overload
    def __mul__(self, other: Quantity[Magnitude] | Magnitude, /) -> Quantity[Magnitude]: ...

    def __mul__(self, other: object, /) -> Quantity[Magnitude]:
        if self._has_numpy_operand(other) and not isinstance(
            other, (Unit, Factor, SymbolicConstant)
        ):
            from ._numpy import apply_ufunc

            return apply_ufunc('multiply', self, other)
        if _is_quantity(other):
            return Quantity(
                cast(Scalar, cast(Scalar, self.value) * cast(Scalar, other.value)),
                self.unit * other.unit,
            )
        if isinstance(other, (Unit, Factor, SymbolicConstant)):
            return Quantity(self.value, self.unit * other)
        return Quantity(cast(Scalar, cast(Scalar, self.value) * cast(Scalar, other)), self.unit)

    @overload
    def __rmul__(self, other: Unit | Factor | SymbolicConstant, /) -> Quantity[T]: ...

    @overload
    def __rmul__(
        self: Quantity[Array], other: Quantity[Magnitude] | Magnitude, /
    ) -> Quantity[NumpyValue]: ...

    @overload
    def __rmul__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...

    @overload
    def __rmul__(self: Quantity[int], other: int, /) -> Quantity[int]: ...

    @overload
    def __rmul__(self: Quantity[Decimal], other: Decimal | int, /) -> Quantity[Decimal]: ...

    @overload
    def __rmul__(self: Quantity[int], other: Decimal, /) -> Quantity[Decimal]: ...

    @overload
    def __rmul__[N: Real](self: Quantity[N], other: Real, /) -> Quantity[Real]: ...

    @overload
    def __rmul__(self: Quantity[Scalar], other: Scalar, /) -> Quantity[Scalar]: ...

    @overload
    def __rmul__(self, other: Magnitude, /) -> Quantity[Magnitude]: ...

    def __rmul__(self, other: object, /) -> Quantity[Magnitude]:
        if self._has_numpy_operand(other) and not isinstance(
            other, (Unit, Factor, SymbolicConstant)
        ):
            from ._numpy import apply_ufunc

            return apply_ufunc('multiply', other, self)
        if isinstance(other, (Unit, Factor, SymbolicConstant)):
            return Quantity(self.value, other * self.unit)
        return Quantity(cast(Scalar, cast(Scalar, other) * cast(Scalar, self.value)), self.unit)

    @overload
    def __truediv__(self, other: Unit | Factor | SymbolicConstant, /) -> Quantity[T]: ...

    @overload
    def __truediv__(
        self: Quantity[Array], other: Quantity[Magnitude] | Magnitude, /
    ) -> Quantity[NumpyValue]: ...

    @overload
    def __truediv__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...

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
    def __truediv__(
        self: Quantity[Scalar], other: Quantity[Scalar] | Scalar, /
    ) -> Quantity[Scalar]: ...

    @overload
    def __truediv__(self, other: Quantity[Magnitude] | Magnitude, /) -> Quantity[Magnitude]: ...

    def __truediv__(self, other: object, /) -> Quantity[Magnitude]:
        if self._has_numpy_operand(other) and not isinstance(
            other, (Unit, Factor, SymbolicConstant)
        ):
            from ._numpy import apply_ufunc

            return apply_ufunc('true_divide', self, other)
        if _is_quantity(other):
            return Quantity(
                cast(Scalar, cast(Scalar, self.value) / cast(Scalar, other.value)),
                self.unit / other.unit,
            )
        if isinstance(other, (Unit, Factor, SymbolicConstant)):
            return Quantity(self.value, self.unit / other)
        return Quantity(cast(Scalar, cast(Scalar, self.value) / cast(Scalar, other)), self.unit)

    @overload
    def __rtruediv__(
        self: Quantity[Array],
        other: Unit | Factor | SymbolicConstant | Quantity[Magnitude] | Magnitude,
        /,
    ) -> Quantity[NumpyValue]: ...

    @overload
    def __rtruediv__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...

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
        self: Quantity[Scalar], other: Unit | Factor | SymbolicConstant | Scalar, /
    ) -> Quantity[Scalar]: ...

    @overload
    def __rtruediv__(
        self, other: Unit | Factor | SymbolicConstant | Magnitude, /
    ) -> Quantity[Magnitude]: ...

    def __rtruediv__(self, other: object, /) -> Quantity[Magnitude]:
        if self._has_numpy_operand(other) and not isinstance(
            other, (Unit, Factor, SymbolicConstant)
        ):
            from ._numpy import apply_ufunc

            return apply_ufunc('true_divide', other, self)
        if isinstance(other, (Unit, Factor, SymbolicConstant)):
            return Quantity(cast(Scalar, 1 / cast(Scalar, self.value)), other / self.unit)
        return Quantity(cast(Scalar, cast(Scalar, other) / cast(Scalar, self.value)), self.unit**-1)

    @overload
    def __pow__(self: Quantity[Array], exponent: object, /) -> Quantity[NumpyValue]: ...

    @overload
    def __pow__(self, exponent: Array | Quantity[Array], /) -> Quantity[NumpyValue]: ...

    @overload
    def __pow__(self: Quantity[Decimal], exponent: int | Decimal, /) -> Quantity[Decimal]: ...

    @overload
    def __pow__(self: Quantity[int], exponent: Decimal, /) -> Quantity[Decimal]: ...

    @overload
    def __pow__[N: Real](self: Quantity[N], exponent: Real, /) -> Quantity[Number]: ...

    @overload
    def __pow__(self: Quantity[Scalar], exponent: Fractionable, /) -> Quantity[Scalar]: ...

    @overload
    def __pow__(self, exponent: Fractionable, /) -> Quantity[Magnitude]: ...

    def __pow__(self, exponent: object, /) -> Quantity[Magnitude]:
        if self._has_numpy_operand(exponent):
            from ._numpy import power

            return power(self, exponent)
        if not isinstance(exponent, Fractionable):
            return NotImplemented
        power = as_exponent(exponent)
        return Quantity(cast(Scalar, cast(Scalar, self.value) ** exponent), self.unit**power)

    @overload
    def __iadd__[A: WritableArray](self: Quantity[A], other: object, /) -> Quantity[A]: ...

    @overload
    def __iadd__(
        self: Quantity[Array], other: Quantity[Magnitude] | Magnitude, /
    ) -> Quantity[NumpyValue]: ...

    @overload
    def __iadd__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...

    @overload
    def __iadd__(
        self: Quantity[Decimal], other: Quantity[Decimal | int] | Decimal | int, /
    ) -> Quantity[Decimal]: ...

    @overload
    def __iadd__(
        self: Quantity[int], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[Decimal]: ...

    @overload
    def __iadd__[L: Real, R: Real](
        self: Quantity[L], other: Quantity[R] | R, /
    ) -> Quantity[Real]: ...

    @overload
    def __iadd__(
        self: Quantity[Scalar], other: Quantity[Scalar] | Scalar, /
    ) -> Quantity[Scalar]: ...

    @overload
    def __iadd__(self, other: Quantity[Magnitude] | Magnitude, /) -> Quantity[Magnitude]: ...

    def __iadd__(self, other: object, /) -> Quantity[Magnitude]:
        return self._inplace_result(self.__add__(cast(Any, other)))

    @overload
    def __isub__[A: WritableArray](self: Quantity[A], other: object, /) -> Quantity[A]: ...

    @overload
    def __isub__(
        self: Quantity[Array], other: Quantity[Magnitude] | Magnitude, /
    ) -> Quantity[NumpyValue]: ...

    @overload
    def __isub__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...

    @overload
    def __isub__(
        self: Quantity[Decimal], other: Quantity[Decimal | int] | Decimal | int, /
    ) -> Quantity[Decimal]: ...

    @overload
    def __isub__(
        self: Quantity[int], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[Decimal]: ...

    @overload
    def __isub__[L: Real, R: Real](
        self: Quantity[L], other: Quantity[R] | R, /
    ) -> Quantity[Real]: ...

    @overload
    def __isub__(
        self: Quantity[Scalar], other: Quantity[Scalar] | Scalar, /
    ) -> Quantity[Scalar]: ...

    @overload
    def __isub__(self, other: Quantity[Magnitude] | Magnitude, /) -> Quantity[Magnitude]: ...

    def __isub__(self, other: object, /) -> Quantity[Magnitude]:
        return self._inplace_result(self.__sub__(cast(Any, other)))

    @overload
    def __imul__[A: WritableArray](self: Quantity[A], other: object, /) -> Quantity[A]: ...

    @overload
    def __imul__(self, other: Unit | Factor | SymbolicConstant, /) -> Quantity[T]: ...

    @overload
    def __imul__(
        self: Quantity[Array], other: Quantity[Magnitude] | Magnitude, /
    ) -> Quantity[NumpyValue]: ...

    @overload
    def __imul__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...

    @overload
    def __imul__(self: Quantity[int], other: Quantity[int] | int, /) -> Quantity[int]: ...

    @overload
    def __imul__(
        self: Quantity[Decimal], other: Quantity[Decimal | int] | Decimal | int, /
    ) -> Quantity[Decimal]: ...

    @overload
    def __imul__(
        self: Quantity[int], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[Decimal]: ...

    @overload
    def __imul__[L: Real, R: Real](
        self: Quantity[L], other: Quantity[R] | R, /
    ) -> Quantity[Real]: ...

    @overload
    def __imul__(
        self: Quantity[Scalar], other: Quantity[Scalar] | Scalar, /
    ) -> Quantity[Scalar]: ...

    @overload
    def __imul__(self, other: Quantity[Magnitude] | Magnitude, /) -> Quantity[Magnitude]: ...

    def __imul__(self, other: object, /) -> Quantity[Magnitude]:
        return self._inplace_result(cast('Quantity[Magnitude]', self.__mul__(cast(Any, other))))

    @overload
    def __itruediv__[A: WritableArray](self: Quantity[A], other: object, /) -> Quantity[A]: ...

    @overload
    def __itruediv__(self, other: Unit | Factor | SymbolicConstant, /) -> Quantity[T]: ...

    @overload
    def __itruediv__(
        self: Quantity[Array], other: Quantity[Magnitude] | Magnitude, /
    ) -> Quantity[NumpyValue]: ...

    @overload
    def __itruediv__(self, other: Quantity[Array] | Array, /) -> Quantity[NumpyValue]: ...

    @overload
    def __itruediv__(self: Quantity[int], other: Quantity[int] | int, /) -> Quantity[float]: ...

    @overload
    def __itruediv__(
        self: Quantity[Decimal], other: Quantity[Decimal | int] | Decimal | int, /
    ) -> Quantity[Decimal]: ...

    @overload
    def __itruediv__(
        self: Quantity[int], other: Quantity[Decimal] | Decimal, /
    ) -> Quantity[Decimal]: ...

    @overload
    def __itruediv__[L: Real, R: Real](
        self: Quantity[L], other: Quantity[R] | R, /
    ) -> Quantity[Real]: ...

    @overload
    def __itruediv__(
        self: Quantity[Scalar], other: Quantity[Scalar] | Scalar, /
    ) -> Quantity[Scalar]: ...

    @overload
    def __itruediv__(self, other: Quantity[Magnitude] | Magnitude, /) -> Quantity[Magnitude]: ...

    def __itruediv__(self, other: object, /) -> Quantity[Magnitude]:
        return self._inplace_result(cast('Quantity[Magnitude]', self.__truediv__(cast(Any, other))))

    @overload
    def __ipow__[A: WritableArray](  # pyright: ignore[reportOverlappingOverload]
        self: Quantity[A], exponent: object, /
    ) -> Quantity[A]: ...

    @overload
    def __ipow__(self: Quantity[Array], exponent: object, /) -> Quantity[NumpyValue]: ...

    @overload
    def __ipow__(self, exponent: Array | Quantity[Array], /) -> Quantity[NumpyValue]: ...

    @overload
    def __ipow__(self: Quantity[Decimal], exponent: int | Decimal, /) -> Quantity[Decimal]: ...

    @overload
    def __ipow__(self: Quantity[int], exponent: Decimal, /) -> Quantity[Decimal]: ...

    @overload
    def __ipow__[N: Real](self: Quantity[N], exponent: Real, /) -> Quantity[Number]: ...

    @overload
    def __ipow__(self: Quantity[Scalar], exponent: Fractionable, /) -> Quantity[Scalar]: ...

    @overload
    def __ipow__(self, exponent: Fractionable, /) -> Quantity[Magnitude]: ...

    def __ipow__(self, exponent: object, /) -> Quantity[Magnitude]:
        return self._inplace_result(self.__pow__(cast(Any, exponent)))

    @overload
    def __imatmul__[A: WritableArray](self: Quantity[A], other: object, /) -> Quantity[A]: ...

    @overload
    def __imatmul__(self, other: object, /) -> Quantity[NumpyValue]: ...

    def __imatmul__(self, other: object, /) -> Quantity[Magnitude]:
        return self._inplace_result(self.__matmul__(other))

    def _inplace_result(self, result: Quantity[Magnitude]) -> Quantity[Magnitude]:
        if result is NotImplemented:
            return NotImplemented
        if is_numpy_object(self.value):
            from ._numpy import write_inplace

            return write_inplace(self, result)
        return result

    def _has_numpy_operand(self, *others: object) -> bool:
        return is_numpy_object(self.value) or any(
            is_numpy_object(x.value if _is_quantity(x) else x) for x in others
        )

    def __array__(self, dtype: object = None, copy: object = None) -> Any:
        raise TypeError('Implicit unit removal is forbidden; use q.value_in(unit) or q.value.')

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        from ._numpy import dispatch_ufunc

        return dispatch_ufunc(ufunc, method, inputs, kwargs)

    def __array_function__(
        self, func: Any, types: tuple[type, ...], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        from ._numpy import dispatch_function

        return dispatch_function(func, types, args, kwargs)

    def _array_value(self) -> Any:
        if not is_numpy_object(self.value):
            raise AttributeError('Array attributes require a NumPy magnitude.')
        return self.value

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array_value().shape

    @property
    def ndim(self) -> int:
        return self._array_value().ndim

    @property
    def size(self) -> int:
        return self._array_value().size

    @property
    def dtype(self: Quantity[Array]) -> NumpyDType:
        return self._array_value().dtype

    @property
    def T[A: Array](self: Quantity[A]) -> Quantity[A]:
        return Quantity(self._array_value().T, self.unit)

    def __getitem__(self: Quantity[Array], key: Any) -> Quantity[NumpyValue]:
        return Quantity(self._array_value()[key], self.unit)

    def __setitem__(
        self: Quantity[Array], key: Any, value: Quantity[Magnitude] | Magnitude
    ) -> None:
        """Assign compatible values to an ndarray, keeping its unit and dtype.

        Bare values are dimensionless, including zero. Convert and broadcast
        into temporary storage with same_kind casting before the indexed write.
        NumPy scalar and Python scalar magnitudes do not support assignment.
        """
        if not is_numpy_object(self.value):
            raise TypeError('Indexed assignment requires a plain numeric ndarray magnitude.')
        from ._numpy import assign_item

        assign_item(self, key, value)

    def __len__(self: Quantity[Array]) -> int:
        return len(self._array_value())

    def __iter__(self: Quantity[Array]) -> Iterator[Quantity[NumpyValue]]:
        return (Quantity(value, self.unit) for value in self._array_value())

    def flatten(self: Quantity[Array], order: str = 'C') -> Quantity[NumpyArray]:
        return Quantity(self._array_value().flatten(order), self.unit)

    @overload
    def __abs__(self: Quantity[Array]) -> Quantity[NumpyValue]: ...

    @overload
    def __abs__(self) -> Quantity[Magnitude]: ...

    def __abs__(self) -> Quantity[Magnitude]:
        if is_numpy_object(self.value):
            from ._numpy import apply_ufunc

            return apply_ufunc('absolute', self)
        return Quantity(abs(cast(Any, self.value)), self.unit)

    def __matmul__(self, other: object) -> Quantity[NumpyValue]:
        if not self._has_numpy_operand(other):
            return NotImplemented
        from ._numpy import apply_ufunc

        return apply_ufunc('matmul', self, other)

    def __rmatmul__(self, other: object) -> Quantity[NumpyValue]:
        if not self._has_numpy_operand(other):
            return NotImplemented
        from ._numpy import apply_ufunc

        return apply_ufunc('matmul', other, self)


def quantity(value: float, /, unit: str | Unit) -> Quantity[Decimal]:
    """Construct a quantity from a float's shortest decimal representation."""
    return Quantity(Decimal(repr(value)), unit)
