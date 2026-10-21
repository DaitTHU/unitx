"""Absolute coordinate readings, distinct from linear Quantity differences."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Iterator
from decimal import Decimal
from fractions import Fraction
from typing import Any, Generic, Never, TypeVar, cast, overload

from ._conversion import _affine, _log_shift, convert
from ._numerics import Array, Magnitude, Scalar, is_numpy_object
from ._numpy_typing import NumpyArray, NumpyDType, NumpyValue, WritableArray
from .affineunit import AffineUnit, _as_affine_unit
from .dimension import Dimension
from .exceptions import DimensionError
from .factor import one
from .quantity import Quantity
from .unit import Unit

T = TypeVar('T', bound=Magnitude, covariant=True)


class AffineQuantity(Generic[T]):
    """An absolute reading in a coordinate with an origin.

    Strings are resolved as coordinates: K has zero origin and degC has its
    Celsius origin. Conversion always preserves point identity, including a
    conversion to K or W. Use to_quantity(unit) to explicitly obtain an ordinary
    physical Quantity. Point minus point returns a Quantity difference; adding
    or subtracting a Quantity difference returns an AffineQuantity.
    """

    __slots__ = ('_value', '_unit')
    _value: T
    _unit: AffineUnit

    def __init__(self, value: T, unit: str | Unit | AffineUnit, /) -> None:
        if is_numpy_object(value):
            from ._numpy import validate_magnitude

            validate_magnitude(value)
            if getattr(value, 'dtype').kind == 'c':
                raise TypeError('Affine coordinates require real magnitudes.')
        elif isinstance(value, bool) or not isinstance(value, (int, float, Fraction, Decimal)):
            raise TypeError('Affine coordinates require real numeric magnitudes.')
        self._value = value
        self._unit = _as_affine_unit(unit)

    @property
    def value(self) -> T:
        return self._value

    @property
    def unit(self) -> AffineUnit:
        return self._unit

    @property
    def dimension(self) -> Dimension:
        return self.unit.dimension

    def __repr__(self) -> str:
        try:
            unit = repr(self.unit.expression())
        except TypeError:
            unit = repr(self.unit)
        return f'AffineQuantity({self.value!r}, {unit})'

    def __str__(self) -> str:
        from ._display import text_quantity

        return text_quantity(self.value, self.unit)

    def __format__(self, format_spec: str) -> str:
        from ._display import format_quantity

        return format_quantity(self.value, self.unit, format_spec)

    def to_latex(self, format_spec: str = '', *, fraction: bool = True) -> str:
        from ._display import quantity_body

        return quantity_body(self.value, self.unit, format_spec, 'latex', fraction=fraction)

    def to_html(self, format_spec: str = '') -> str:
        from ._display import html_span, quantity_body

        return html_span(quantity_body(self.value, self.unit, format_spec, 'html'))

    def to_siunitx(self, format_spec: str = '') -> str:
        from ._display import siunitx_quantity

        return siunitx_quantity(self.value, self.unit, format_spec)

    def _repr_latex_(self) -> str:
        return '$' + self.to_latex() + '$'

    def _repr_html_(self) -> str:
        return self.to_html()

    def is_dimensionless(self) -> bool:
        return self.unit.is_dimensionless()

    def __bool__(self) -> bool:
        return bool(self.value)

    @overload
    def value_in(self: AffineQuantity[Array], unit: str | Unit | AffineUnit, /) -> NumpyValue: ...

    @overload
    def value_in(self: AffineQuantity[Decimal], unit: str | Unit | AffineUnit, /) -> Decimal: ...

    @overload
    def value_in(self, unit: str | Unit | AffineUnit, /) -> Magnitude: ...

    def value_in(self, unit: str | Unit | AffineUnit, /) -> Magnitude:
        from ._conversion import convert

        return cast(Magnitude, convert(self.value, self.unit, _as_affine_unit(unit)))

    @overload
    def to(
        self: AffineQuantity[Array], unit: str | Unit | AffineUnit, /
    ) -> AffineQuantity[NumpyValue]: ...

    @overload
    def to(
        self: AffineQuantity[Decimal], unit: str | Unit | AffineUnit, /
    ) -> AffineQuantity[Decimal]: ...

    @overload
    def to(self, unit: str | Unit | AffineUnit, /) -> AffineQuantity[Magnitude]: ...

    def to(self, unit: str | Unit | AffineUnit, /) -> AffineQuantity[Magnitude]:
        target = _as_affine_unit(unit)
        return AffineQuantity(self.value_in(target), target)

    @overload
    def to_quantity(
        self: AffineQuantity[Array], unit: str | Unit | AffineUnit, /
    ) -> Quantity[NumpyValue]: ...

    @overload
    def to_quantity(
        self: AffineQuantity[Decimal], unit: str | Unit | AffineUnit, /
    ) -> Quantity[Decimal]: ...

    @overload
    def to_quantity(self, unit: str | Unit | AffineUnit, /) -> Quantity[Magnitude]: ...

    def to_quantity(self, unit: str | Unit | AffineUnit, /) -> Quantity[Magnitude]:
        """Evaluate the physical quantity in a linear unit, explicitly exiting point rules."""
        from ._conversion import convert, linear_unit

        target = linear_unit(unit)
        if target.log_degree:
            raise TypeError(
                'A physical exit requires an ordinary physical unit, not a log difference.'
            )
        return Quantity(cast(Magnitude, convert(self.value, self.unit, target)), target)

    __hash__ = None  # type: ignore[assignment]

    def _compare(self, other: object, op: Callable[[Any, Any], Any]) -> Any:
        return _compare(self, other, op)

    @overload
    def __eq__(self: AffineQuantity[Array], other: object, /) -> Array: ...

    @overload
    def __eq__(self: AffineQuantity[Scalar], other: AffineQuantity[Array], /) -> Array: ...

    @overload
    def __eq__(self: AffineQuantity[Scalar], other: object, /) -> bool: ...

    @overload
    def __eq__(self, other: object, /) -> bool | Array: ...

    def __eq__(self, other: object, /) -> Any:
        return self._compare(other, operator.eq)

    @overload
    def __ne__(self: AffineQuantity[Array], other: object, /) -> Array: ...

    @overload
    def __ne__(self: AffineQuantity[Scalar], other: AffineQuantity[Array], /) -> Array: ...

    @overload
    def __ne__(self: AffineQuantity[Scalar], other: object, /) -> bool: ...

    @overload
    def __ne__(self, other: object, /) -> bool | Array: ...

    def __ne__(self, other: object, /) -> Any:
        return self._compare(other, operator.ne)

    @overload
    def __lt__(self: AffineQuantity[Array], other: object, /) -> Array: ...

    @overload
    def __lt__(self: AffineQuantity[Scalar], other: AffineQuantity[Array], /) -> Array: ...

    @overload
    def __lt__(self: AffineQuantity[Scalar], other: object, /) -> bool: ...

    @overload
    def __lt__(self, other: object, /) -> bool | Array: ...

    def __lt__(self, other: object, /) -> Any:
        return self._compare(other, operator.lt)

    @overload
    def __le__(self: AffineQuantity[Array], other: object, /) -> Array: ...

    @overload
    def __le__(self: AffineQuantity[Scalar], other: AffineQuantity[Array], /) -> Array: ...

    @overload
    def __le__(self: AffineQuantity[Scalar], other: object, /) -> bool: ...

    @overload
    def __le__(self, other: object, /) -> bool | Array: ...

    def __le__(self, other: object, /) -> Any:
        return self._compare(other, operator.le)

    @overload
    def __gt__(self: AffineQuantity[Array], other: object, /) -> Array: ...

    @overload
    def __gt__(self: AffineQuantity[Scalar], other: AffineQuantity[Array], /) -> Array: ...

    @overload
    def __gt__(self: AffineQuantity[Scalar], other: object, /) -> bool: ...

    @overload
    def __gt__(self, other: object, /) -> bool | Array: ...

    def __gt__(self, other: object, /) -> Any:
        return self._compare(other, operator.gt)

    @overload
    def __ge__(self: AffineQuantity[Array], other: object, /) -> Array: ...

    @overload
    def __ge__(self: AffineQuantity[Scalar], other: AffineQuantity[Array], /) -> Array: ...

    @overload
    def __ge__(self: AffineQuantity[Scalar], other: object, /) -> bool: ...

    @overload
    def __ge__(self, other: object, /) -> bool | Array: ...

    def __ge__(self, other: object, /) -> Any:
        return self._compare(other, operator.ge)

    @overload
    def isclose(
        self: AffineQuantity[Array],
        other: AffineQuantity[Magnitude],
        *,
        rel_tol: Any = 0,
        abs_tol: Any = 0,
    ) -> Array: ...

    @overload
    def isclose(
        self: AffineQuantity[Scalar],
        other: AffineQuantity[Array],
        *,
        rel_tol: Any = 0,
        abs_tol: Any = 0,
    ) -> Array: ...

    @overload
    def isclose(
        self: AffineQuantity[Scalar],
        other: AffineQuantity[Scalar],
        *,
        rel_tol: Any = 0,
        abs_tol: Any = 0,
    ) -> bool | Array: ...

    @overload
    def isclose(
        self, other: AffineQuantity[Magnitude], *, rel_tol: Any = 0, abs_tol: Any = 0
    ) -> Any: ...

    def isclose(
        self, other: AffineQuantity[Magnitude], *, rel_tol: Any = 0, abs_tol: Any = 0
    ) -> Any:
        """Compare points with a difference tolerance; relative tolerance must be zero."""
        return _isclose(self, other, rel_tol, abs_tol)

    def __pos__(self) -> AffineQuantity[Magnitude]:
        return AffineQuantity(cast(Magnitude, +cast(Any, self.value)), self.unit)

    def __neg__(self) -> Never:
        raise TypeError('Negation is undefined for an affine point.')

    def __abs__(self) -> Never:
        raise TypeError('Absolute value is undefined for an affine point.')

    def __add__(self, other: Quantity[Magnitude], /) -> AffineQuantity[Magnitude]:
        return _add_sub(self, other, subtract=False)

    def __radd__(self, other: Quantity[Magnitude], /) -> AffineQuantity[Magnitude]:
        return self.__add__(other)

    @overload
    def __sub__(self, other: AffineQuantity[Magnitude], /) -> Quantity[Magnitude]: ...

    @overload
    def __sub__(self, other: Quantity[Magnitude], /) -> AffineQuantity[Magnitude]: ...

    def __sub__(self, other: object, /) -> Quantity[Magnitude] | AffineQuantity[Magnitude]:
        return _add_sub(self, other, subtract=True)

    def __rsub__(self, other: object, /) -> Never:
        raise TypeError('A difference cannot subtract an affine point.')

    def __mul__(self, other: object, /) -> Never:
        raise TypeError('Convert an affine point to_quantity(unit) before multiplication.')

    def __rmul__(self, other: object, /) -> Never:
        raise TypeError('Convert an affine point to_quantity(unit) before multiplication.')

    def __truediv__(self, other: object, /) -> Never:
        raise TypeError('Convert an affine point to_quantity(unit) before division.')

    def __rtruediv__(self, other: object, /) -> Never:
        raise TypeError('Convert an affine point to_quantity(unit) before division.')

    def __pow__(self, exponent: object, /) -> Never:
        raise TypeError('Convert an affine point to_quantity(unit) before powers.')

    def __matmul__(self, other: object, /) -> Never:
        raise TypeError('Matrix multiplication is undefined for affine points.')

    def __rmatmul__(self, other: object, /) -> Never:
        raise TypeError('Matrix multiplication is undefined for affine points.')

    @overload
    def __iadd__[A: WritableArray](
        self: AffineQuantity[A], other: Quantity[Magnitude], /
    ) -> AffineQuantity[A]: ...

    @overload
    def __iadd__(self, other: Quantity[Magnitude], /) -> AffineQuantity[Magnitude]: ...

    def __iadd__(self, other: Quantity[Magnitude], /) -> AffineQuantity[Magnitude]:
        result = self + other
        if is_numpy_object(self.value):
            from ._numpy import write_inplace

            return write_inplace(self, result)
        return result

    @overload
    def __isub__[A: WritableArray](
        self: AffineQuantity[A], other: object, /
    ) -> AffineQuantity[A]: ...

    @overload
    def __isub__(self, other: AffineQuantity[Magnitude], /) -> Quantity[Magnitude]: ...

    @overload
    def __isub__(self, other: Quantity[Magnitude], /) -> AffineQuantity[Magnitude]: ...

    def __isub__(self, other: Any, /) -> Quantity[Magnitude] | AffineQuantity[Magnitude]:
        result = self - other
        if is_numpy_object(self.value):
            from ._numpy import write_inplace

            return write_inplace(self, result)
        return result

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
    def dtype(self: AffineQuantity[Array]) -> NumpyDType:
        return self._array_value().dtype

    @property
    def T[A: Array](self: AffineQuantity[A]) -> AffineQuantity[A]:
        return AffineQuantity(self._array_value().T, self.unit)

    def __getitem__(self: AffineQuantity[Array], key: Any) -> AffineQuantity[NumpyValue]:
        return AffineQuantity(self._array_value()[key], self.unit)

    def __setitem__(
        self: AffineQuantity[Array], key: Any, value: AffineQuantity[Magnitude]
    ) -> None:
        from ._numpy import assign_item

        assign_item(self, key, value)

    def __len__(self: AffineQuantity[Array]) -> int:
        return len(self._array_value())

    def __iter__(self: AffineQuantity[Array]) -> Iterator[AffineQuantity[NumpyValue]]:
        return (AffineQuantity(value, self.unit) for value in self._array_value())

    def flatten(self: AffineQuantity[Array], order: str = 'C') -> AffineQuantity[NumpyArray]:
        return AffineQuantity(self._array_value().flatten(order), self.unit)


def _addition_parts(left: Any, right: Any, *, subtract: bool) -> tuple[Any, Any, Unit | AffineUnit]:

    if not isinstance(left, AffineQuantity):
        if not subtract and isinstance(left, Quantity) and isinstance(right, AffineQuantity):
            b, a, unit = _addition_parts(right, left, subtract=False)
            return a, b, unit
        raise TypeError('A difference cannot subtract an affine point.')
    if isinstance(right, AffineQuantity):
        if not subtract:
            raise TypeError('Two affine points cannot be added.')
        if left.unit.kind != right.unit.kind:
            raise DimensionError('Point subtraction requires the same coordinate convention.')
        raw = (
            Decimal(right.value)
            if isinstance(left.value, Decimal) and isinstance(right.value, int)
            else right.value
        )
        return left.value, convert(raw, right.unit, left.unit), left.unit.delta_unit
    if not isinstance(right, Quantity):
        raise TypeError('Point arithmetic requires an explicit Quantity difference.')
    delta = convert(right.value, right.unit, left.unit.delta_unit)
    if isinstance(left.value, Decimal) and isinstance(delta, (int, Fraction)):
        delta = _affine(Decimal(0), 1, delta)
    return left.value, delta, left.unit


def _add_sub(left: Any, right: Any, *, subtract: bool) -> Any:

    a, b, unit = _addition_parts(left, right, subtract=subtract)
    # Avoid rounding the aligned coordinate before cancellation in a Decimal difference.
    if (
        isinstance(a, Decimal)
        and _finite(a)
        and isinstance(right.value, (Decimal, int))
        and _finite(right.value)
    ):
        target = left.unit if isinstance(right, AffineQuantity) else left.unit.delta_unit
        exact = (operator.sub if subtract else operator.add)(
            Fraction(a), convert(Fraction(right.value), right.unit, target)
        )
        value = (
            Decimal(exact.numerator) / Decimal(exact.denominator)
            if isinstance(exact, Fraction)
            else Decimal(str(exact))
        )
    else:
        value = (operator.sub if subtract else operator.add)(a, b)
    return AffineQuantity(value, unit) if isinstance(unit, AffineUnit) else Quantity(value, unit)


def _compare(left: Any, right: Any, op: Any) -> Any:
    equal = op in (operator.eq, operator.ne)
    mismatch = not isinstance(right, AffineQuantity)
    if not mismatch:
        mismatch = left.dimension != right.dimension or left.unit.kind != right.unit.kind
    if mismatch:
        if equal:
            raw = getattr(right, 'value', right)
            if is_numpy_object(left.value) or is_numpy_object(raw):
                from ._numpy import comparison_constant

                return comparison_constant(left.value, raw, op is operator.ne)
            return op is operator.ne
        raise DimensionError('Point comparison requires compatible AffineQuantity objects.')
    u, v = left.unit, right.unit
    arrays = is_numpy_object(left.value) or is_numpy_object(right.value)
    a, b = left.value, right.value
    if u == v:
        return op(a, b)
    if not arrays and u.kind is None:
        if u.origin == v.origin:
            return op(Quantity(a, u.delta_unit), Quantity(b, v.delta_unit))
        if not _finite(a) or not _finite(b):
            return op(a, b)
        return op(_physical_point(left), _physical_point(right))
    if u.kind is not None and not arrays:
        ratio = u.delta_unit.factor / v.delta_unit.factor
        if ratio.decompose()[1] != one:
            if u.origin == v.origin:
                return op(Quantity(left.value, u.delta_unit), Quantity(right.value, v.delta_unit))
            raise TypeError(
                'These coordinates require approximate alignment; convert explicitly first.'
            )
        _log_shift(v, u, exact=True)
    if not arrays:
        try:
            a, b = Fraction(a), Fraction(b)
        except (ValueError, OverflowError):
            pass
    return op(a, convert(b, v, u))


def _physical_point(point: AffineQuantity[Any]) -> Quantity[Any]:
    """Collapse a finite linear point exactly, without approximating symbolic sums."""
    unit = point.unit
    if unit.origin.value == 0:
        return Quantity(point.value, unit.delta_unit)
    if point.value == 0:
        return unit.origin
    rational, irrational = (unit.delta_unit.factor / unit.origin.unit.factor).decompose()
    if irrational != one:
        raise TypeError(
            'These coordinates require a symbolic sum for exact comparison; '
            'use isclose with a difference tolerance or convert explicitly first.'
        )
    return Quantity(
        Fraction(point.value) * rational
        + Fraction(cast(int | float | Fraction | Decimal, unit.origin.value)),
        unit.origin.unit,
    )


def _finite(value: Any) -> bool:
    if isinstance(value, Decimal):
        return value.is_finite()
    if isinstance(value, (int, Fraction)):
        return True
    return math.isfinite(value)


def _isclose(left: Any, right: Any, rel_tol: Any, abs_tol: Any, *, equal_nan: bool = False) -> Any:

    if not isinstance(left, AffineQuantity) or not isinstance(right, AffineQuantity):
        raise TypeError('Affine isclose requires two AffineQuantity points.')
    unit = left.unit
    if left.dimension != right.dimension or left.unit.kind != right.unit.kind:
        raise DimensionError('Point comparison requires compatible coordinates.')
    if isinstance(rel_tol, Quantity):
        rel_tol = rel_tol.value_in(Unit())
    if is_numpy_object(rel_tol):
        from ._numpy import zero_tolerance

        zero_tolerance(rel_tol)
    elif not isinstance(rel_tol, (int, float, Fraction, Decimal)):
        raise TypeError('Relative tolerance must be real.')
    elif rel_tol != 0:
        raise ValueError(
            'Affine isclose requires rel_tol=0; use a difference abs_tol '
            'or convert to physical units.'
        )
    if isinstance(abs_tol, Quantity):
        tolerance = convert(abs_tol.value, abs_tol.unit, unit.delta_unit)
    elif is_numpy_object(abs_tol):
        from ._numpy import zero_tolerance

        zero_tolerance(abs_tol)
        tolerance = abs_tol
    elif isinstance(abs_tol, (int, float, Fraction, Decimal)) and abs_tol == 0:
        tolerance = 0
    else:
        raise TypeError('Absolute tolerance must be a Quantity difference or real zero.')
    if not is_numpy_object(tolerance) and not isinstance(
        tolerance, (int, float, Fraction, Decimal)
    ):
        raise TypeError('Absolute tolerance must be real.')
    if not is_numpy_object(tolerance) and tolerance == 0:
        if not any(is_numpy_object(x) for x in (left.value, right.value, rel_tol)):
            return _compare(left, right, operator.eq)
    a = convert(left.value, left.unit, unit)
    b = convert(right.value, right.unit, unit)
    if any(is_numpy_object(x) for x in (a, b, tolerance, rel_tol)):
        from ._numpy import affine_close_magnitude

        return affine_close_magnitude(a, b, tolerance, rel_tol, equal_nan=equal_nan)
    if not _finite(tolerance) or tolerance < 0:
        raise ValueError('Absolute tolerance must be finite and nonnegative.')
    if a == b:
        return True
    if not _finite(a) or not _finite(b):
        return equal_nan and math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tolerance
