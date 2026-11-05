"""Finite optional SciPy adapters; dependencies load only on use."""

from typing import Callable, Protocol, TypeAlias, cast, overload

from ._numerics import Array, Fractionable
from ._numerics.numpy_typing import (
    _Complex128,
    _DTypeValue,
    _Float64,
    _NumpyValueOf,
)
from ._operands import _Point
from .quantity import Quantity

__all__ = ['jv', 'simpson']

_Samples: TypeAlias = Array | Fractionable | complex | Quantity[Array | Fractionable | complex]
_Coordinates: TypeAlias = Array | Fractionable | Quantity[Array | Fractionable]


class _BesselFunction(Protocol):
    @overload
    def __call__(
        self,
        v: _Coordinates,
        z: _DTypeValue[_Float64] | Quantity[_DTypeValue[_Float64]],
        /,
    ) -> Quantity[_NumpyValueOf[_Float64]]: ...

    @overload
    def __call__(
        self,
        v: _Coordinates,
        z: _DTypeValue[_Complex128] | Quantity[_DTypeValue[_Complex128]],
        /,
    ) -> Quantity[_NumpyValueOf[_Complex128]]: ...

    @overload
    def __call__(self, v: _Coordinates, z: _Samples, /) -> Quantity[Array]: ...


def _typed_bessel(function: Callable[[_Coordinates, _Samples], Quantity[Array]]) -> _BesselFunction:
    return cast(_BesselFunction, function)


@_typed_bessel
def jv(v: _Coordinates, z: _Samples, /) -> Quantity[Array]:
    """Return dimensionless Bessel values with typed Quantity dispatch.

    Both inputs must be dimensionless linear values. Bare numeric inputs are
    coherent dimensionless values. Broadcasting and numerical domains follow
    SciPy. Use `scipy.special.jv` directly for ufunc options such as `out`.
    NumPy and SciPy load only when called.
    """
    from scipy.special import jv as bessel  # pyright: ignore[reportMissingImports]

    from ._numerics.numpy_operands import _is_supported_operand
    from .si import si

    if (
        isinstance(v, _Point)
        or isinstance(z, _Point)
        or not _is_supported_operand(v)
        or not _is_supported_operand(z)
    ):
        error_message = 'jv requires numeric values or linear quantities.'
        raise TypeError(error_message)
    # Wrapping the argument also supplies Quantity dispatch for bare-only calls.
    argument = z if isinstance(z, Quantity) else Quantity(z, si.one)
    return cast(Quantity[Array], bessel(v, argument))


def simpson(
    y: _Samples,
    *,
    x: _Coordinates | None = None,
    dx: _Coordinates = 1.0,
    axis: int = -1,
) -> Quantity[Array]:
    """Integrate linear samples with SciPy; return the sample unit times spacing.

    `x` and `dx` are real numeric coordinates, bare or wrapped in Quantity.
    Bare values are dimensionless. `dx` must be scalar and is ignored when
    `x` is supplied. Shape, axis, sampling and numerical behavior follow SciPy.
    Points, object arrays and implicit list/tuple magnitudes are unsupported.
    SciPy and NumPy load on first call; scalar core imports need neither.
    """
    import numpy as np  # pyright: ignore[reportMissingImports]
    from scipy.integrate import simpson as integrate  # pyright: ignore[reportMissingImports]

    from ._numerics.numpy_operands import _is_supported_operand, _linear_parts

    if not _is_supported_operand(y):
        error_message = 'simpson requires numeric samples or a linear Quantity.'
        raise TypeError(error_message)
    values, unit = _linear_parts(y)
    coordinate = dx if x is None else x
    if not _is_supported_operand(coordinate):
        error_message = 'Integration coordinates must be real numeric values or linear quantities.'
        raise TypeError(error_message)
    spacing, spacing_unit = _linear_parts(coordinate)
    if np.iscomplexobj(spacing):
        error_message = 'Integration coordinates must be real.'
        raise TypeError(error_message)
    if x is None:
        if np.ndim(spacing) != 0:
            error_message = 'dx must be scalar.'
            raise TypeError(error_message)
        result = integrate(values, dx=spacing, axis=axis)
    else:
        result = integrate(values, x=spacing, axis=axis)
    return Quantity(cast(Array, result), unit * spacing_unit)
