"""Concrete NumPy result types without a runtime NumPy import.

Inputs still use the NumPy-independent Array protocol. Missing optional NumPy
types must never turn scalar overload inputs into Any or Unknown.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, SupportsIndex, TypeAlias, TypeVar

from . import Array

# Shared method defaults distinguish omission from explicitly invalid values.
_UNSET = object()
_Axis: TypeAlias = SupportsIndex | tuple[SupportsIndex, ...] | None
_Shape: TypeAlias = SupportsIndex | tuple[SupportsIndex, ...]

Component = TypeVar('Component', bound=Array, covariant=True)
_ScalarType = TypeVar('_ScalarType', bound='_NumpyScalar', covariant=True)


class HasReal(Array, Protocol[Component]):
    @property
    def real(self) -> Component: ...


class HasImag(Array, Protocol[Component]):
    @property
    def imag(self) -> Component: ...


class WritableArray(Array, Protocol):
    """Distinguish ndarray storage from NumPy scalars without optional imports.

    Retain the concrete magnitude type in shape/copy, out and augmented annotations;
    the runtime adapter still validates plain numeric ndarrays.
    """

    def __setitem__(self, key: Any, value: Any, /) -> None: ...


class _TypedArray(WritableArray, Protocol[_ScalarType]):
    """Capture dtype without retaining a source ndarray's shape parameter."""

    @property
    def dtype(self) -> _NumpyDTypeOf[_ScalarType]: ...


if TYPE_CHECKING:
    import numpy as np  # pyright: ignore[reportMissingImports]
    from numpy.typing import NDArray  # pyright: ignore[reportMissingImports]

    _NumpyScalar: TypeAlias = np.generic
    _NumpyArrayOf: TypeAlias = NDArray[_ScalarType]
    _NumpyDTypeOf: TypeAlias = np.dtype[_ScalarType]
    NumpyArray: TypeAlias = NDArray[Any]
    NumpyValue: TypeAlias = NumpyArray | np.number[Any]
    NumpyDType: TypeAlias = np.dtype[Any]
else:
    # Keep annotations resolvable without importing an optional dependency.
    type _NumpyArrayOf[S] = Array
    type _NumpyDTypeOf[S] = object
    _NumpyScalar = object
    NumpyArray = NumpyValue = Array
    NumpyDType = object
