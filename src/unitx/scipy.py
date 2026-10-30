"""Finite optional SciPy integration adapters; dependencies load only on use."""

from typing import TypeAlias, cast

from ._numerics import Array, Fractionable
from .quantity import Quantity

__all__ = ['simpson']

_Samples: TypeAlias = Array | Fractionable | complex | Quantity[Array | Fractionable | complex]
_Coordinates: TypeAlias = Array | Fractionable | Quantity[Array | Fractionable]


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
