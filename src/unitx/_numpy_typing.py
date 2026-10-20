"""Concrete NumPy result types without a runtime NumPy import.

Inputs still use the NumPy-independent Array protocol. Missing optional NumPy
types must never turn scalar overload inputs into Any or Unknown.
"""

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

from ._numerics import Array


class WritableArray(Array, Protocol):
    """Distinguish ndarray storage from NumPy scalars without optional imports.

    Only used to retain the concrete magnitude type in augmented annotations;
    the runtime adapter still validates plain numeric ndarrays.
    """

    def __setitem__(self, key: Any, value: Any, /) -> None: ...


if TYPE_CHECKING:
    import numpy as np  # pyright: ignore[reportMissingImports]
    from numpy.typing import NDArray  # pyright: ignore[reportMissingImports]

    NumpyArray: TypeAlias = NDArray[Any]
    NumpyValue: TypeAlias = NumpyArray | np.number[Any]
    NumpyDType: TypeAlias = np.dtype[Any]
else:
    # Keep annotations resolvable without importing an optional dependency.
    NumpyArray = NumpyValue = Array
    NumpyDType = object
