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


class _DTypeValue(Array, Protocol[_ScalarType]):
    """Capture a NumPy array/scalar dtype independently of its shape."""

    @property
    def dtype(self) -> _NumpyDTypeOf[_ScalarType]: ...


class _TypedArray(WritableArray, _DTypeValue[_ScalarType], Protocol[_ScalarType]):
    """Capture ndarray dtype without retaining the source shape parameter."""


if TYPE_CHECKING:
    import numpy as np  # pyright: ignore[reportMissingImports]
    from numpy.typing import NDArray  # pyright: ignore[reportMissingImports]

    _NumpyScalar: TypeAlias = np.generic
    _NumpyNumber: TypeAlias = np.number[Any]
    _InexactScalar = TypeVar('_InexactScalar', bound=np.inexact[Any])
    _FloatingScalar = TypeVar('_FloatingScalar', bound=np.floating[Any])
    _ComplexScalar = TypeVar('_ComplexScalar', bound=np.complexfloating[Any, Any])
    _NumpyArrayOf: TypeAlias = NDArray[_ScalarType]
    _NumpyValueOf: TypeAlias = NDArray[_ScalarType] | _ScalarType
    _NumpyDTypeOf: TypeAlias = np.dtype[_ScalarType]
    NumpyArray: TypeAlias = NDArray[Any]
    NumpyValue: TypeAlias = NumpyArray | np.number[Any]
    NumpyDType: TypeAlias = np.dtype[Any]
else:
    # Keep annotations resolvable without importing an optional dependency.
    type _NumpyArrayOf[S] = Array
    type _NumpyValueOf[S] = Array
    type _NumpyDTypeOf[S] = object
    _NumpyScalar = object
    _NumpyNumber = object
    _InexactScalar = TypeVar('_InexactScalar')
    _FloatingScalar = TypeVar('_FloatingScalar')
    _ComplexScalar = TypeVar('_ComplexScalar')
    NumpyArray = NumpyValue = Array
    NumpyDType = object
