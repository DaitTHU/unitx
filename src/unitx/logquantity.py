"""Log readings with concrete result construction and point/difference rules."""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterator
from decimal import Decimal
from fractions import Fraction
from typing import Any, Generic, Literal, Never, SupportsIndex, TypeVar, cast, overload

from ._coordinates.points import _add_sub, _compare, _isclose
from ._numerics import Array, Magnitude, Scalar, is_numpy_object
from ._numerics.numpy_typing import (
    _UNSET,
    NumpyArray,
    NumpyDType,
    NumpyValue,
    WritableArray,
    _Axis,
    _NumpyArrayOf,
    _NumpyScalar,
    _Shape,
    _TypedArray,
)
from ._operands import _Point, linear_unit
from .dimension import Dimension
from .logunit import LogUnit, _as_log_unit
from .quantity import Quantity
from .unit import Unit

__all__ = ['LogQuantity']
T = TypeVar('T', bound=Magnitude, covariant=True)


class LogQuantity(_Point['LogQuantity[Magnitude]'], Generic[T]):
    """A logarithmic reading; `.to()` stays in its coordinate family."""

    __slots__ = ('_value', '_unit')

    _value: T

    _unit: LogUnit

    def __init__(self, value: T, unit: str | Unit | LogUnit, /) -> None:
        if is_numpy_object(value):
            from ._numerics.numpy import validate_magnitude

            validate_magnitude(value)
            if getattr(value, 'dtype').kind == 'c':
                error_message = 'Logarithmic coordinates require real magnitudes.'
                raise TypeError(error_message)
        elif isinstance(value, bool) or not isinstance(value, (int, float, Fraction, Decimal)):
            error_message = 'Logarithmic coordinates require real numeric magnitudes.'
            raise TypeError(error_message)
        self._value = value
        self._unit = self._resolve_coordinate(unit)

    @staticmethod
    def _resolve_coordinate(unit: str | Unit | LogUnit) -> LogUnit:
        return _as_log_unit(unit)

    @property
    def value(self) -> T:
        return self._value

    @property
    def unit(self) -> LogUnit:
        return self._unit

    def as_pair(self) -> tuple[T, LogUnit]:
        """Return the stored reading and coordinate without conversion or copy."""
        return self.value, self.unit

    @property
    def dimension(self) -> Dimension:
        return self.unit.dimension

    def __repr__(self) -> str:
        expression = self.unit._default_expression()
        unit = repr(expression) if expression is not None else repr(self.unit)
        return f'{type(self).__name__}({self.value!r}, {unit})'

    def __str__(self) -> str:
        from ._formatting.render import text_quantity

        return text_quantity(self.value, self.unit)

    def __format__(self, format_spec: str) -> str:
        from ._formatting.render import format_quantity

        return format_quantity(self.value, self.unit, format_spec)

    def to_latex(self, format_spec: str = '', *, fraction: bool = True) -> str:
        from ._formatting.render import quantity_body

        return quantity_body(self.value, self.unit, format_spec, 'latex', fraction=fraction)

    def to_html(self, format_spec: str = '') -> str:
        from ._formatting.render import html_span, quantity_body

        return html_span(quantity_body(self.value, self.unit, format_spec, 'html'))

    def to_siunitx(self, format_spec: str = '') -> str:
        from ._formatting.render import siunitx_quantity

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
    def value_in(self: LogQuantity[Array], unit: str | Unit | LogUnit, /) -> NumpyValue: ...

    @overload
    def value_in(self: LogQuantity[Decimal], unit: str | Unit | LogUnit, /) -> Decimal: ...

    @overload
    def value_in(self, unit: str | Unit | LogUnit, /) -> Magnitude: ...

    def value_in(self, unit: str | Unit | LogUnit, /) -> Magnitude:
        return cast(
            Magnitude,
            self.unit._mapping._value_in(self.value, self._resolve_coordinate(unit)._mapping),
        )

    @overload
    def to(self: LogQuantity[Array], unit: str | Unit | LogUnit, /) -> LogQuantity[NumpyValue]: ...

    @overload
    def to(self: LogQuantity[Decimal], unit: str | Unit | LogUnit, /) -> LogQuantity[Decimal]: ...

    @overload
    def to(self: LogQuantity[T], unit: str | Unit | LogUnit, /) -> LogQuantity[Magnitude]: ...

    def to(self, unit: str | Unit | LogUnit, /) -> LogQuantity[Magnitude]:
        target = self._resolve_coordinate(unit)
        return LogQuantity(self.unit._mapping._value_in(self.value, target._mapping), target)

    @overload
    def to_quantity(
        self: LogQuantity[Array], unit: str | Unit | LogUnit, /
    ) -> Quantity[NumpyValue]: ...

    @overload
    def to_quantity(
        self: LogQuantity[Decimal], unit: str | Unit | LogUnit, /
    ) -> Quantity[Decimal]: ...

    @overload
    def to_quantity(self, unit: str | Unit | LogUnit, /) -> Quantity[Magnitude]: ...

    def to_quantity(self, unit: str | Unit | LogUnit, /) -> Quantity[Magnitude]:
        """Evaluate the physical quantity in a linear unit, explicitly exiting point rules."""
        target = linear_unit(unit)
        if target.dimension != target.dimension.physical:
            error_message = (
                'A physical exit requires an ordinary physical unit, not a logarithmic difference.'
            )
            raise TypeError(error_message)
        return Quantity(
            cast(Magnitude, self.unit._mapping._physical_value(self.value, target)), target
        )

    __hash__ = None  # type: ignore[assignment]

    def _compare(self, other: object, op: Callable[[Any, Any], Any]) -> Any:
        return _compare(self, other, op)

    @overload
    def __eq__(self: LogQuantity[Array], other: object, /) -> Array: ...

    @overload
    def __eq__(self: LogQuantity[Scalar], other: LogQuantity[Array], /) -> Array: ...

    @overload
    def __eq__(self: LogQuantity[Scalar], other: object, /) -> bool: ...

    @overload
    def __eq__(self, other: object, /) -> bool | Array: ...

    def __eq__(self, other: object, /) -> Any:
        return self._compare(other, operator.eq)

    @overload
    def __ne__(self: LogQuantity[Array], other: object, /) -> Array: ...

    @overload
    def __ne__(self: LogQuantity[Scalar], other: LogQuantity[Array], /) -> Array: ...

    @overload
    def __ne__(self: LogQuantity[Scalar], other: object, /) -> bool: ...

    @overload
    def __ne__(self, other: object, /) -> bool | Array: ...

    def __ne__(self, other: object, /) -> Any:
        return self._compare(other, operator.ne)

    @overload
    def __lt__(self: LogQuantity[Array], other: object, /) -> Array: ...

    @overload
    def __lt__(self: LogQuantity[Scalar], other: LogQuantity[Array], /) -> Array: ...

    @overload
    def __lt__(self: LogQuantity[Scalar], other: object, /) -> bool: ...

    @overload
    def __lt__(self, other: object, /) -> bool | Array: ...

    def __lt__(self, other: object, /) -> Any:
        return self._compare(other, operator.lt)

    @overload
    def __le__(self: LogQuantity[Array], other: object, /) -> Array: ...

    @overload
    def __le__(self: LogQuantity[Scalar], other: LogQuantity[Array], /) -> Array: ...

    @overload
    def __le__(self: LogQuantity[Scalar], other: object, /) -> bool: ...

    @overload
    def __le__(self, other: object, /) -> bool | Array: ...

    def __le__(self, other: object, /) -> Any:
        return self._compare(other, operator.le)

    @overload
    def __gt__(self: LogQuantity[Array], other: object, /) -> Array: ...

    @overload
    def __gt__(self: LogQuantity[Scalar], other: LogQuantity[Array], /) -> Array: ...

    @overload
    def __gt__(self: LogQuantity[Scalar], other: object, /) -> bool: ...

    @overload
    def __gt__(self, other: object, /) -> bool | Array: ...

    def __gt__(self, other: object, /) -> Any:
        return self._compare(other, operator.gt)

    @overload
    def __ge__(self: LogQuantity[Array], other: object, /) -> Array: ...

    @overload
    def __ge__(self: LogQuantity[Scalar], other: LogQuantity[Array], /) -> Array: ...

    @overload
    def __ge__(self: LogQuantity[Scalar], other: object, /) -> bool: ...

    @overload
    def __ge__(self, other: object, /) -> bool | Array: ...

    def __ge__(self, other: object, /) -> Any:
        return self._compare(other, operator.ge)

    @overload
    def isclose(
        self: LogQuantity[Array],
        other: LogQuantity[Magnitude],
        *,
        rel_tol: Any = 0,
        abs_tol: Any = 0,
    ) -> Array: ...

    @overload
    def isclose(
        self: LogQuantity[Scalar],
        other: LogQuantity[Array],
        *,
        rel_tol: Any = 0,
        abs_tol: Any = 0,
    ) -> Array: ...

    @overload
    def isclose(
        self: LogQuantity[Scalar],
        other: LogQuantity[Scalar],
        *,
        rel_tol: Any = 0,
        abs_tol: Any = 0,
    ) -> bool | Array: ...

    @overload
    def isclose(
        self, other: LogQuantity[Magnitude], *, rel_tol: Any = 0, abs_tol: Any = 0
    ) -> Any: ...

    def isclose(self, other: LogQuantity[Magnitude], *, rel_tol: Any = 0, abs_tol: Any = 0) -> Any:
        """Compare points with a difference tolerance; relative tolerance must be zero."""
        return _isclose(self, other, rel_tol, abs_tol)

    def __pos__(self) -> LogQuantity[Magnitude]:
        return LogQuantity(cast(Magnitude, +cast(Any, self.value)), self.unit)

    def __neg__(self) -> Never:
        error_message = 'Negation is undefined for an logarithmic reading.'
        raise TypeError(error_message)

    def __abs__(self) -> Never:
        error_message = 'Absolute value is undefined for an logarithmic reading.'
        raise TypeError(error_message)

    def __add__(self, other: Quantity[Magnitude], /) -> LogQuantity[Magnitude]:
        value, _ = _add_sub(self, other, subtract=False)
        return LogQuantity(value, self.unit)

    def __radd__(self, other: Quantity[Magnitude], /) -> LogQuantity[Magnitude]:
        value, _ = _add_sub(self, other, subtract=False)
        return LogQuantity(value, self.unit)

    @overload
    def __sub__(self: LogQuantity[T], other: LogQuantity[Magnitude], /) -> Quantity[Magnitude]: ...

    @overload
    def __sub__(self: LogQuantity[T], other: Quantity[Magnitude], /) -> LogQuantity[Magnitude]: ...

    @overload
    def __sub__(
        self: LogQuantity[T], other: object, /
    ) -> Quantity[Magnitude] | LogQuantity[Magnitude]: ...

    def __sub__(self, other: object, /) -> Quantity[Magnitude] | LogQuantity[Magnitude]:
        value, unit = _add_sub(self, other, subtract=True)
        return Quantity(value, unit) if isinstance(unit, Unit) else LogQuantity(value, self.unit)

    def __rsub__(self, other: object, /) -> Never:
        error_message = 'A difference cannot subtract an logarithmic reading.'
        raise TypeError(error_message)

    def __mul__(self, other: object, /) -> Never:
        error_message = 'Convert an logarithmic reading to_quantity(unit) before multiplication.'
        raise TypeError(error_message)

    def __rmul__(self, other: object, /) -> Never:
        error_message = 'Convert an logarithmic reading to_quantity(unit) before multiplication.'
        raise TypeError(error_message)

    def __truediv__(self, other: object, /) -> Never:
        error_message = 'Convert an logarithmic reading to_quantity(unit) before division.'
        raise TypeError(error_message)

    def __rtruediv__(self, other: object, /) -> Never:
        error_message = 'Convert an logarithmic reading to_quantity(unit) before division.'
        raise TypeError(error_message)

    def __pow__(self, exponent: object, /) -> Never:
        error_message = 'Convert an logarithmic reading to_quantity(unit) before powers.'
        raise TypeError(error_message)

    def __matmul__(self, other: object, /) -> Never:
        error_message = 'Matrix multiplication is undefined for logarithmic readings.'
        raise TypeError(error_message)

    def __rmatmul__(self, other: object, /) -> Never:
        error_message = 'Matrix multiplication is undefined for logarithmic readings.'
        raise TypeError(error_message)

    @overload
    def __iadd__[A: WritableArray](
        self: LogQuantity[A], other: Quantity[Magnitude], /
    ) -> LogQuantity[A]: ...

    @overload
    def __iadd__(self: LogQuantity[T], other: Quantity[Magnitude], /) -> LogQuantity[Magnitude]: ...

    def __iadd__(self, other: Quantity[Magnitude], /) -> LogQuantity[Magnitude]:
        result = self + other
        if is_numpy_object(self.value):
            from ._numerics.numpy import write_inplace

            return write_inplace(self, result)
        return result

    @overload
    def __isub__[A: WritableArray](self: LogQuantity[A], other: object, /) -> LogQuantity[A]: ...

    @overload
    def __isub__(self: LogQuantity[T], other: LogQuantity[Magnitude], /) -> Quantity[Magnitude]: ...

    @overload
    def __isub__(self: LogQuantity[T], other: Quantity[Magnitude], /) -> LogQuantity[Magnitude]: ...

    @overload
    def __isub__(
        self: LogQuantity[T], other: Any, /
    ) -> Quantity[Magnitude] | LogQuantity[Magnitude]: ...

    def __isub__(self, other: Any, /) -> Quantity[Magnitude] | LogQuantity[Magnitude]:
        result = self - other
        if is_numpy_object(self.value):
            from ._numerics.numpy import write_inplace

            return write_inplace(self, result)
        return result

    def __array__(self, dtype: object = None, copy: object = None) -> Any:
        error_message = 'Implicit unit removal is forbidden; use q.value_in(unit) or q.value.'
        raise TypeError(error_message)

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        from ._numerics.numpy import dispatch_ufunc

        return dispatch_ufunc(ufunc, method, inputs, kwargs)

    def __array_function__(
        self, func: Any, types: tuple[type, ...], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        from ._numerics.numpy import dispatch_function

        return dispatch_function(func, types, args, kwargs)

    def _array_value(self) -> Any:
        if not is_numpy_object(self.value):
            error_message = 'Array attributes require a NumPy magnitude.'
            raise AttributeError(error_message)
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
    def dtype(self: LogQuantity[Array]) -> NumpyDType:
        return self._array_value().dtype

    @property
    def T[A: Array](self: LogQuantity[A]) -> LogQuantity[A]:
        return LogQuantity(self._array_value().T, self.unit)

    def __getitem__(self: LogQuantity[Array], key: Any) -> LogQuantity[NumpyValue]:
        return LogQuantity(self._array_value()[key], self.unit)

    def __setitem__(self: LogQuantity[Array], key: Any, value: LogQuantity[Magnitude]) -> None:
        from ._numerics.numpy import assign_item

        assign_item(self, key, value)

    def __len__(self: LogQuantity[Array]) -> int:
        if not is_numpy_object(self.value):
            error_message = 'Scalar quantities have no len().'
            raise TypeError(error_message)
        return len(self._array_value())

    def __iter__(self: LogQuantity[Array]) -> Iterator[LogQuantity[NumpyValue]]:
        if not is_numpy_object(self.value):
            error_message = 'Scalar quantities are not iterable.'
            raise TypeError(error_message)
        return (LogQuantity(value, self.unit) for value in self._array_value())

    @overload
    def reshape[S: _NumpyScalar](
        self: LogQuantity[_TypedArray[S]],
        shape: _Shape,
        /,
        *dimensions: SupportsIndex,
        order: Literal['C', 'F', 'A'] = 'C',
        copy: bool | None = None,
    ) -> LogQuantity[_NumpyArrayOf[S]]: ...
    @overload
    def reshape(
        self: LogQuantity[Array],
        shape: _Shape,
        /,
        *dimensions: SupportsIndex,
        order: Literal['C', 'F', 'A'] = 'C',
        copy: bool | None = None,
    ) -> LogQuantity[NumpyValue]: ...
    def reshape(
        self: LogQuantity[Array],
        shape: _Shape,
        /,
        *dimensions: SupportsIndex,
        order: Literal['C', 'F', 'A'] = 'C',
        copy: bool | None = None,
    ) -> LogQuantity[Array]:
        """Reshape with a tuple or positional dimensions and NumPy's copy/view rules."""
        self._array_value()
        from ._numerics.numpy import _reshape

        new_shape = (shape, *dimensions) if dimensions else shape
        return _reshape(self, new_shape, order=order, copy=copy)

    @overload
    def ravel[S: _NumpyScalar](
        self: LogQuantity[_TypedArray[S]],
        order: Literal['C', 'F', 'A', 'K'] = 'C',
    ) -> LogQuantity[_NumpyArrayOf[S]]: ...
    @overload
    def ravel(
        self: LogQuantity[Array],
        order: Literal['C', 'F', 'A', 'K'] = 'C',
    ) -> LogQuantity[NumpyArray]: ...
    def ravel(
        self: LogQuantity[Array],
        order: Literal['C', 'F', 'A', 'K'] = 'C',
    ) -> LogQuantity[Array]:
        """Return a one-dimensional array, sharing storage when NumPy permits."""
        self._array_value()
        from ._numerics.numpy import _ravel

        return _ravel(self, order=order)

    @overload
    def copy[S: _NumpyScalar](
        self: LogQuantity[_TypedArray[S]],
        order: Literal['C', 'F', 'A', 'K'] = 'C',
    ) -> LogQuantity[_NumpyArrayOf[S]]: ...
    @overload
    def copy(
        self: LogQuantity[Array],
        order: Literal['C', 'F', 'A', 'K'] = 'C',
    ) -> LogQuantity[NumpyArray]: ...
    def copy(
        self: LogQuantity[Array],
        order: Literal['C', 'F', 'A', 'K'] = 'C',
    ) -> LogQuantity[Array]:
        """Copy into independent ndarray storage; the default order is `C`."""
        self._array_value()
        from ._numerics.numpy import _array_transform, np

        return _array_transform(np.copy, self, order=order)

    @overload
    def mean(
        self: LogQuantity[Array],
        axis: _Axis = None,
        dtype: object = None,
        out: None = None,
        keepdims: bool = False,
        *,
        where: object = True,
    ) -> LogQuantity[NumpyValue]: ...
    @overload
    def mean[O: WritableArray](
        self: LogQuantity[Array],
        axis: _Axis,
        dtype: object,
        out: LogQuantity[O],
        keepdims: bool = False,
        *,
        where: object = True,
    ) -> LogQuantity[O]: ...
    @overload
    def mean[O: WritableArray](
        self: LogQuantity[Array],
        axis: _Axis = None,
        dtype: object = None,
        *,
        out: LogQuantity[O],
        keepdims: bool = False,
        where: object = True,
    ) -> LogQuantity[O]: ...
    def mean(
        self: LogQuantity[Array],
        axis: _Axis = None,
        dtype: object = None,
        out: LogQuantity[Array] | None = None,
        keepdims: bool = False,
        *,
        where: object = True,
    ) -> LogQuantity[Array]:
        """Average stored readings, using the existing NumPy unit and output rules."""
        self._array_value()
        from ._numerics.numpy import _mean, np

        return _mean(np.mean, self, axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where)

    @overload
    def std(
        self: LogQuantity[Array],
        axis: _Axis = None,
        dtype: object = None,
        out: None = None,
        ddof: object = ...,
        keepdims: bool = False,
        *,
        where: object = True,
        mean: object = ...,
        correction: object = ...,
    ) -> Quantity[NumpyValue]: ...
    @overload
    def std[O: WritableArray](
        self: LogQuantity[Array],
        axis: _Axis,
        dtype: object,
        out: Quantity[O],
        ddof: object = ...,
        keepdims: bool = False,
        *,
        where: object = True,
        mean: object = ...,
        correction: object = ...,
    ) -> Quantity[O]: ...
    @overload
    def std[O: WritableArray](
        self: LogQuantity[Array],
        axis: _Axis = None,
        dtype: object = None,
        *,
        out: Quantity[O],
        ddof: object = ...,
        keepdims: bool = False,
        where: object = True,
        mean: object = ...,
        correction: object = ...,
    ) -> Quantity[O]: ...
    def std(
        self: LogQuantity[Array],
        axis: _Axis = None,
        dtype: object = None,
        out: Quantity[Array] | None = None,
        ddof: object = _UNSET,
        keepdims: bool = False,
        *,
        where: object = True,
        mean: object = _UNSET,
        correction: object = _UNSET,
    ) -> Quantity[Array]:
        """Return a difference standard deviation using NumPy unit/output rules."""
        self._array_value()
        from ._numerics.numpy import _std_var, np

        return _std_var(
            np.std,
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            where=where,
            ddof=ddof,
            mean=mean,
            correction=correction,
        )

    def flatten(self: LogQuantity[Array], order: str = 'C') -> LogQuantity[NumpyArray]:
        return LogQuantity(self._array_value().flatten(order), self.unit)

    @classmethod
    @overload
    def from_quantity(
        cls, quantity: Quantity[Array], unit: str | Unit | LogUnit, /
    ) -> LogQuantity[Array]: ...

    @classmethod
    @overload
    def from_quantity(
        cls, quantity: Quantity[Decimal], unit: str | Unit | LogUnit, /
    ) -> LogQuantity[Decimal]: ...

    @classmethod
    @overload
    def from_quantity(
        cls, quantity: Quantity[Magnitude], unit: str | Unit | LogUnit, /
    ) -> LogQuantity[Magnitude]: ...

    @classmethod
    def from_quantity(
        cls, quantity: Quantity[Magnitude], unit: str | Unit | LogUnit, /
    ) -> LogQuantity[Magnitude]:
        """Explicitly map a physical Quantity to a logarithmic reading."""
        if not isinstance(quantity, Quantity):
            error_message = 'Logarithmic input must be a physical Quantity.'
            raise TypeError(error_message)
        coordinate = _as_log_unit(unit)
        return cls(coordinate._mapping._reading_from(quantity.value, quantity.unit), coordinate)
