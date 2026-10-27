"""Concrete NumPy result types without a runtime NumPy import.

Inputs still use the NumPy-independent Array protocol. Missing optional NumPy
types must never turn scalar overload inputs into Any or Unknown.
"""

from __future__ import annotations

from types import EllipsisType
from typing import TYPE_CHECKING, Any, Literal, Protocol, SupportsIndex, TypeAlias, TypeVar

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


class _TypedScalar(_DTypeValue[_ScalarType], Protocol[_ScalarType]):
    """Capture NumPy scalar dtype without matching an ndarray's unknown shape."""

    @property
    def ndim(self) -> Literal[0]: ...

    @property
    def shape(self) -> tuple[()]: ...


if TYPE_CHECKING:
    import numpy as np  # pyright: ignore[reportMissingImports]
    from numpy.typing import NDArray  # pyright: ignore[reportMissingImports]

    _NumpyScalar: TypeAlias = np.generic
    _NumpyNumber: TypeAlias = np.number[Any]
    _IntegerScalar: TypeAlias = np.integer[Any]
    _Float32: TypeAlias = np.float32
    _Float64: TypeAlias = np.float64
    _Complex64: TypeAlias = np.complex64
    _Complex128: TypeAlias = np.complex128
    # A successful selection with one of these indices always returns an array.
    # Integer arrays are excluded: a zero-dimensional integer index can scalarize.
    _ArrayIndex: TypeAlias = slice | EllipsisType | None | bool | _DTypeValue[np.bool_]
    _ArrayLeadingIndex: TypeAlias = _ArrayIndex | tuple[_ArrayIndex, *tuple[Any, ...]]
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
    _IntegerScalar = object
    _Float32 = _Float64 = _Complex64 = _Complex128 = object
    _ArrayIndex = _ArrayLeadingIndex = object
    _InexactScalar = TypeVar('_InexactScalar')
    _FloatingScalar = TypeVar('_FloatingScalar')
    _ComplexScalar = TypeVar('_ComplexScalar')
    NumpyArray = NumpyValue = Array
    NumpyDType = object
