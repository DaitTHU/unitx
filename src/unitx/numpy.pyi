from typing import Any, Literal, Protocol, SupportsIndex, overload

import numpy as np
from numpy.typing import DTypeLike, NDArray

from ._numerics import Array
from ._numerics.numpy_typing import _DTypeValue
from .affinequantity import AffineQuantity
from .logquantity import LogQuantity
from .quantity import Quantity
from .typing import NumpyValue

type _Spacing = Array | int | float | Quantity[Array | int | float]
type _Shape = SupportsIndex | tuple[SupportsIndex, ...] | None

class _Like(Protocol):
    @overload
    def __call__[S: np.number[Any]](
        self,
        a: Quantity[_DTypeValue[S]],
        dtype: None = None,
        order: Literal['K', 'A', 'C', 'F'] = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: Literal['cpu'] | None = None,
    ) -> Quantity[NDArray[S]]: ...
    @overload
    def __call__(
        self,
        a: Quantity[Array],
        dtype: DTypeLike | None = None,
        order: Literal['K', 'A', 'C', 'F'] = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: Literal['cpu'] | None = None,
    ) -> Quantity[NDArray[Any]]: ...
    @overload
    def __call__[S: np.number[Any]](
        self,
        a: AffineQuantity[_DTypeValue[S]],
        dtype: None = None,
        order: Literal['K', 'A', 'C', 'F'] = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: Literal['cpu'] | None = None,
    ) -> AffineQuantity[NDArray[S]]: ...
    @overload
    def __call__(
        self,
        a: AffineQuantity[Array],
        dtype: DTypeLike | None = None,
        order: Literal['K', 'A', 'C', 'F'] = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: Literal['cpu'] | None = None,
    ) -> AffineQuantity[NDArray[Any]]: ...
    @overload
    def __call__[S: np.number[Any]](
        self,
        a: LogQuantity[_DTypeValue[S]],
        dtype: None = None,
        order: Literal['K', 'A', 'C', 'F'] = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: Literal['cpu'] | None = None,
    ) -> LogQuantity[NDArray[S]]: ...
    @overload
    def __call__(
        self,
        a: LogQuantity[Array],
        dtype: DTypeLike | None = None,
        order: Literal['K', 'A', 'C', 'F'] = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: Literal['cpu'] | None = None,
    ) -> LogQuantity[NDArray[Any]]: ...

class _FullLike(Protocol):
    @overload
    def __call__[S: np.number[Any]](
        self,
        a: Quantity[_DTypeValue[S]],
        fill_value: object,
        dtype: None = None,
        order: Literal['K', 'A', 'C', 'F'] = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: Literal['cpu'] | None = None,
    ) -> Quantity[NDArray[S]]: ...
    @overload
    def __call__(
        self,
        a: Quantity[Array],
        fill_value: object,
        dtype: DTypeLike | None = None,
        order: Literal['K', 'A', 'C', 'F'] = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: Literal['cpu'] | None = None,
    ) -> Quantity[NDArray[Any]]: ...
    @overload
    def __call__[S: np.number[Any]](
        self,
        a: AffineQuantity[_DTypeValue[S]],
        fill_value: object,
        dtype: None = None,
        order: Literal['K', 'A', 'C', 'F'] = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: Literal['cpu'] | None = None,
    ) -> AffineQuantity[NDArray[S]]: ...
    @overload
    def __call__(
        self,
        a: AffineQuantity[Array],
        fill_value: object,
        dtype: DTypeLike | None = None,
        order: Literal['K', 'A', 'C', 'F'] = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: Literal['cpu'] | None = None,
    ) -> AffineQuantity[NDArray[Any]]: ...
    @overload
    def __call__[S: np.number[Any]](
        self,
        a: LogQuantity[_DTypeValue[S]],
        fill_value: object,
        dtype: None = None,
        order: Literal['K', 'A', 'C', 'F'] = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: Literal['cpu'] | None = None,
    ) -> LogQuantity[NDArray[S]]: ...
    @overload
    def __call__(
        self,
        a: LogQuantity[Array],
        fill_value: object,
        dtype: DTypeLike | None = None,
        order: Literal['K', 'A', 'C', 'F'] = 'K',
        subok: bool = True,
        shape: _Shape = None,
        *,
        device: Literal['cpu'] | None = None,
    ) -> LogQuantity[NDArray[Any]]: ...

zeros_like: _Like
ones_like: _Like
empty_like: _Like
full_like: _FullLike

@overload
def trapezoid[S: np.inexact[Any]](
    y: Quantity[_DTypeValue[S]],
    x: _DTypeValue[S] | Quantity[_DTypeValue[S]] | None = None,
    dx: float = 1.0,
    axis: SupportsIndex = -1,
) -> Quantity[NDArray[S] | S]: ...
@overload
def trapezoid(
    y: Quantity[Array],
    x: _Spacing | None = None,
    dx: _Spacing = 1.0,
    axis: SupportsIndex = -1,
) -> Quantity[NumpyValue]: ...
@overload
def gradient[S: np.inexact[Any]](
    f: Quantity[_DTypeValue[S]] | AffineQuantity[_DTypeValue[S]] | LogQuantity[_DTypeValue[S]],
    *varargs: _Spacing,
    axis: SupportsIndex,
    edge_order: Literal[1, 2] = 1,
) -> Quantity[NDArray[S]]: ...
@overload
def gradient(
    f: Quantity[Array] | AffineQuantity[Array] | LogQuantity[Array],
    *varargs: _Spacing,
    axis: SupportsIndex,
    edge_order: Literal[1, 2] = 1,
) -> Quantity[NDArray[Any]]: ...
@overload
def gradient(
    f: Quantity[Array] | AffineQuantity[Array] | LogQuantity[Array],
    *varargs: _Spacing,
    axis: tuple[SupportsIndex, ...] | None = None,
    edge_order: Literal[1, 2] = 1,
) -> Quantity[NDArray[Any]] | tuple[Quantity[NDArray[Any]], ...]: ...
