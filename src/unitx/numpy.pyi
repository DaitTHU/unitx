from decimal import Decimal
from fractions import Fraction
from typing import Any, Literal, Protocol, SupportsIndex, TypeVar, overload

import numpy as np
from numpy.typing import DTypeLike, NDArray

from ._numerics import Array, Fractionable, Magnitude
from ._numerics.numpy_typing import _DTypeValue
from .affinequantity import AffineQuantity
from .logquantity import LogQuantity
from .quantity import Quantity
from .typing import NumpyValue

type _Numeric = Array | complex | Fractionable
type _Wrapped[T: Magnitude] = Quantity[T] | AffineQuantity[T] | LogQuantity[T]
type _Fill[Q] = _Numeric | Q | list[Any] | tuple[Any, ...]
type _Order = Literal['K', 'A', 'C', 'F']
type _Shape = SupportsIndex | tuple[SupportsIndex, ...] | None
type _Device = Literal['cpu'] | None

class _Like(Protocol):
    @overload
    def __call__[S: np.number[Any]](
        self,
        a: Quantity[_DTypeValue[S]],
        dtype: None = None,
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> Quantity[NDArray[S]]: ...
    @overload
    def __call__[S: np.number[Any]](
        self,
        a: Quantity[_Numeric],
        dtype: type[S] | np.dtype[S],
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> Quantity[NDArray[S]]: ...
    @overload
    def __call__(
        self,
        a: Quantity[_Numeric],
        dtype: type[int],
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> Quantity[NDArray[np.intp]]: ...
    @overload
    def __call__(
        self,
        a: Quantity[_Numeric],
        dtype: type[float],
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> Quantity[NDArray[np.float64]]: ...
    @overload
    def __call__(
        self,
        a: Quantity[_Numeric],
        dtype: type[complex],
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> Quantity[NDArray[np.complex128]]: ...
    @overload
    def __call__(
        self,
        a: Quantity[_Numeric],
        dtype: DTypeLike,
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> Quantity[NDArray[Any]]: ...
    @overload
    def __call__(
        self,
        a: Quantity[Array | complex],
        dtype: None = None,
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> Quantity[NDArray[Any]]: ...
    @overload
    def __call__[S: np.number[Any]](
        self,
        a: AffineQuantity[_DTypeValue[S]],
        dtype: None = None,
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> AffineQuantity[NDArray[S]]: ...
    @overload
    def __call__[S: np.number[Any]](
        self,
        a: AffineQuantity[Array | Fractionable],
        dtype: type[S] | np.dtype[S],
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> AffineQuantity[NDArray[S]]: ...
    @overload
    def __call__(
        self,
        a: AffineQuantity[Array | Fractionable],
        dtype: type[int],
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> AffineQuantity[NDArray[np.intp]]: ...
    @overload
    def __call__(
        self,
        a: AffineQuantity[Array | Fractionable],
        dtype: type[float],
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> AffineQuantity[NDArray[np.float64]]: ...
    @overload
    def __call__(
        self,
        a: AffineQuantity[Array | Fractionable],
        dtype: DTypeLike,
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> AffineQuantity[NDArray[Any]]: ...
    @overload
    def __call__(
        self,
        a: AffineQuantity[Array | float],
        dtype: None = None,
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> AffineQuantity[NDArray[Any]]: ...
    @overload
    def __call__[S: np.number[Any]](
        self,
        a: LogQuantity[_DTypeValue[S]],
        dtype: None = None,
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> LogQuantity[NDArray[S]]: ...
    @overload
    def __call__[S: np.number[Any]](
        self,
        a: LogQuantity[Array | Fractionable],
        dtype: type[S] | np.dtype[S],
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> LogQuantity[NDArray[S]]: ...
    @overload
    def __call__(
        self,
        a: LogQuantity[Array | Fractionable],
        dtype: type[int],
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> LogQuantity[NDArray[np.intp]]: ...
    @overload
    def __call__(
        self,
        a: LogQuantity[Array | Fractionable],
        dtype: type[float],
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> LogQuantity[NDArray[np.float64]]: ...
    @overload
    def __call__(
        self,
        a: LogQuantity[Array | Fractionable],
        dtype: DTypeLike,
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> LogQuantity[NDArray[Any]]: ...
    @overload
    def __call__(
        self,
        a: LogQuantity[Array | float],
        dtype: None = None,
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> LogQuantity[NDArray[Any]]: ...

class _FullLike(Protocol):
    @overload
    def __call__[S: np.number[Any]](
        self,
        a: Quantity[_DTypeValue[S]],
        fill_value: _Fill[Quantity[_Numeric]],
        dtype: None = None,
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> Quantity[NDArray[S]]: ...
    @overload
    def __call__[S: np.number[Any]](
        self,
        a: Quantity[_Numeric],
        fill_value: _Fill[Quantity[_Numeric]],
        dtype: type[S] | np.dtype[S],
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> Quantity[NDArray[S]]: ...
    @overload
    def __call__(
        self,
        a: Quantity[_Numeric],
        fill_value: _Fill[Quantity[_Numeric]],
        dtype: type[int],
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> Quantity[NDArray[np.intp]]: ...
    @overload
    def __call__(
        self,
        a: Quantity[_Numeric],
        fill_value: _Fill[Quantity[_Numeric]],
        dtype: type[float],
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> Quantity[NDArray[np.float64]]: ...
    @overload
    def __call__(
        self,
        a: Quantity[_Numeric],
        fill_value: _Fill[Quantity[_Numeric]],
        dtype: type[complex],
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> Quantity[NDArray[np.complex128]]: ...
    @overload
    def __call__(
        self,
        a: Quantity[_Numeric],
        fill_value: _Fill[Quantity[_Numeric]],
        dtype: DTypeLike | None = None,
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> Quantity[NDArray[Any]]: ...
    @overload
    def __call__[S: np.number[Any]](
        self,
        a: AffineQuantity[_DTypeValue[S]],
        fill_value: _Fill[AffineQuantity[Array | Fractionable]],
        dtype: None = None,
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> AffineQuantity[NDArray[S]]: ...
    @overload
    def __call__[S: np.number[Any]](
        self,
        a: AffineQuantity[Array | Fractionable],
        fill_value: _Fill[AffineQuantity[Array | Fractionable]],
        dtype: type[S] | np.dtype[S],
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> AffineQuantity[NDArray[S]]: ...
    @overload
    def __call__(
        self,
        a: AffineQuantity[Array | Fractionable],
        fill_value: _Fill[AffineQuantity[Array | Fractionable]],
        dtype: type[int],
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> AffineQuantity[NDArray[np.intp]]: ...
    @overload
    def __call__(
        self,
        a: AffineQuantity[Array | Fractionable],
        fill_value: _Fill[AffineQuantity[Array | Fractionable]],
        dtype: type[float],
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> AffineQuantity[NDArray[np.float64]]: ...
    @overload
    def __call__(
        self,
        a: AffineQuantity[Array | Fractionable],
        fill_value: _Fill[AffineQuantity[Array | Fractionable]],
        dtype: DTypeLike | None = None,
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> AffineQuantity[NDArray[Any]]: ...
    @overload
    def __call__[S: np.number[Any]](
        self,
        a: LogQuantity[_DTypeValue[S]],
        fill_value: _Fill[LogQuantity[Array | Fractionable]],
        dtype: None = None,
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> LogQuantity[NDArray[S]]: ...
    @overload
    def __call__[S: np.number[Any]](
        self,
        a: LogQuantity[Array | Fractionable],
        fill_value: _Fill[LogQuantity[Array | Fractionable]],
        dtype: type[S] | np.dtype[S],
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> LogQuantity[NDArray[S]]: ...
    @overload
    def __call__(
        self,
        a: LogQuantity[Array | Fractionable],
        fill_value: _Fill[LogQuantity[Array | Fractionable]],
        dtype: type[int],
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> LogQuantity[NDArray[np.intp]]: ...
    @overload
    def __call__(
        self,
        a: LogQuantity[Array | Fractionable],
        fill_value: _Fill[LogQuantity[Array | Fractionable]],
        dtype: type[float],
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> LogQuantity[NDArray[np.float64]]: ...
    @overload
    def __call__(
        self,
        a: LogQuantity[Array | Fractionable],
        fill_value: _Fill[LogQuantity[Array | Fractionable]],
        dtype: DTypeLike | None = None,
        order: _Order = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: _Device = None,
    ) -> LogQuantity[NDArray[Any]]: ...

zeros_like: _Like
ones_like: _Like
empty_like: _Like
full_like: _FullLike

# NumPy treats a Python float as weak and a NumPy float64 as strong.
# A parameter annotated float can contain either, so an explicit float step
# may widen a low-precision sample even when a default step preserves it.
type _Step = (
    Fractionable | complex | np.number[Any] | np.ndarray[tuple[()], np.dtype[np.number[Any]]]
)
type _LinearStep = _Step | Quantity[_Step]
type _Coordinates = Array | Quantity[Array]
type _PointCoordinates = np.ndarray[tuple[int], np.dtype[np.number[Any]]]
type _Spacing = (
    _Numeric
    | Quantity[_Numeric]
    | AffineQuantity[_PointCoordinates]
    | LogQuantity[_PointCoordinates]
)
_Inexact = TypeVar(
    '_Inexact',
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
)

@overload
def trapezoid[S: np.inexact[Any]](
    y: Quantity[_DTypeValue[S]],
    x: None = None,
    dx: int | Decimal | Fraction | Quantity[int | Decimal | Fraction] = ...,
    axis: SupportsIndex = -1,
) -> Quantity[NDArray[S] | S]: ...
@overload
def trapezoid(
    y: Quantity[_DTypeValue[_Inexact]],
    x: _DTypeValue[_Inexact] | Quantity[_DTypeValue[_Inexact]],
    dx: object = 1.0,
    axis: SupportsIndex = -1,
) -> Quantity[NDArray[_Inexact] | _Inexact]: ...
@overload
def trapezoid(
    y: Quantity[_DTypeValue[np.float16 | np.float32 | np.float64]],
    x: None = None,
    dx: np.float64 | Quantity[np.float64] = ...,
    axis: SupportsIndex = -1,
) -> Quantity[NDArray[np.float64] | np.float64]: ...
@overload
def trapezoid(
    y: Quantity[_DTypeValue[np.complex64 | np.complex128]],
    x: None = None,
    dx: np.float64 | Quantity[np.float64] = ...,
    axis: SupportsIndex = -1,
) -> Quantity[NDArray[np.complex128] | np.complex128]: ...
@overload
def trapezoid(
    y: Quantity[_DTypeValue[_Inexact]],
    x: None = None,
    dx: _Inexact | Quantity[_Inexact] = ...,
    axis: SupportsIndex = -1,
) -> Quantity[NDArray[_Inexact] | _Inexact]: ...
@overload
def trapezoid[S: np.floating[Any]](
    y: Quantity[_DTypeValue[S]],
    x: None = None,
    dx: float | Quantity[float] = ...,
    axis: SupportsIndex = -1,
) -> Quantity[NDArray[S | np.float64] | S | np.float64]: ...
@overload
def trapezoid[S: np.complexfloating[Any, Any]](
    y: Quantity[_DTypeValue[S]],
    x: None = None,
    dx: float | Quantity[float] = ...,
    axis: SupportsIndex = -1,
) -> Quantity[NDArray[S | np.complex128] | S | np.complex128]: ...
@overload
def trapezoid(
    y: Quantity[Array], x: _Coordinates, dx: object = 1.0, axis: SupportsIndex = -1
) -> Quantity[NumpyValue]: ...
@overload
def trapezoid(
    y: Quantity[Array], x: None = None, dx: _LinearStep = 1.0, axis: SupportsIndex = -1
) -> Quantity[NumpyValue]: ...

type _SingleAxis = SupportsIndex | tuple[SupportsIndex]
type _MultipleAxes = tuple[SupportsIndex, SupportsIndex, *tuple[SupportsIndex, ...]]
type _Gradient[S: np.number[Any]] = Quantity[NDArray[S]]

@overload
def gradient(
    f: _Wrapped[Fractionable | complex | np.number[Any]],
    *varargs: _Spacing,
    axis: None = None,
    edge_order: Literal[1, 2] = 1,
) -> tuple[()]: ...
@overload
def gradient(
    f: _Wrapped[_Numeric], *varargs: _Spacing, axis: tuple[()], edge_order: Literal[1, 2] = 1
) -> tuple[()]: ...
@overload
def gradient[S: np.inexact[Any]](
    f: _Wrapped[_DTypeValue[S]],
    *varargs: _Spacing,
    axis: _SingleAxis,
    edge_order: Literal[1, 2] = 1,
) -> _Gradient[S]: ...
@overload
def gradient(
    f: _Wrapped[_DTypeValue[np.integer[Any]]],
    *varargs: _Spacing,
    axis: _SingleAxis,
    edge_order: Literal[1, 2] = 1,
) -> _Gradient[np.float64]: ...
@overload
def gradient(
    f: _Wrapped[Array],
    *varargs: _Spacing,
    axis: _SingleAxis,
    edge_order: Literal[1, 2] = 1,
) -> _Gradient[Any]: ...
@overload
def gradient[S: np.inexact[Any]](
    f: _Wrapped[_DTypeValue[S]],
    *varargs: _Spacing,
    axis: _MultipleAxes,
    edge_order: Literal[1, 2] = 1,
) -> tuple[_Gradient[S], ...]: ...
@overload
def gradient(
    f: _Wrapped[_DTypeValue[np.integer[Any]]],
    *varargs: _Spacing,
    axis: _MultipleAxes,
    edge_order: Literal[1, 2] = 1,
) -> tuple[_Gradient[np.float64], ...]: ...
@overload
def gradient(
    f: _Wrapped[Array],
    *varargs: _Spacing,
    axis: _MultipleAxes,
    edge_order: Literal[1, 2] = 1,
) -> tuple[_Gradient[Any], ...]: ...
@overload
def gradient[S: np.inexact[Any]](
    f: _Wrapped[_DTypeValue[S]],
    *varargs: _Spacing,
    axis: tuple[SupportsIndex, ...] | None = None,
    edge_order: Literal[1, 2] = 1,
) -> _Gradient[S] | tuple[_Gradient[S], ...]: ...
@overload
def gradient(
    f: _Wrapped[_DTypeValue[np.integer[Any]]],
    *varargs: _Spacing,
    axis: tuple[SupportsIndex, ...] | None = None,
    edge_order: Literal[1, 2] = 1,
) -> _Gradient[np.float64] | tuple[_Gradient[np.float64], ...]: ...
@overload
def gradient(
    f: _Wrapped[Array],
    *varargs: _Spacing,
    axis: tuple[SupportsIndex, ...] | None = None,
    edge_order: Literal[1, 2] = 1,
) -> _Gradient[Any] | tuple[_Gradient[Any], ...]: ...
