"""Finite typed NumPy entry points for wrapped arrays.

Import this optional module explicitly; importing it requires NumPy. Calls
require wrapped inputs and retain ordinary NumPy dispatch, complete units,
coordinate families and native numerical behavior. Public overloads live in
`numpy.pyi`; this module owns runtime validation and forwarding.
"""

from typing import Any, Literal

import numpy as np

from ._operands import _Point
from .quantity import Quantity

__all__ = ['empty_like', 'full_like', 'gradient', 'ones_like', 'trapezoid', 'zeros_like']


def _require_quantity(value: Any) -> None:
    if not isinstance(value, (Quantity, _Point)):
        error_message = 'This entry point requires a Quantity or point template.'
        raise TypeError(error_message)


def zeros_like(
    a: Any,
    dtype: Any = None,
    order: Literal['K', 'A', 'C', 'F'] = 'K',
    subok: bool = True,
    shape: Any = None,
    *,
    device: Any = None,
) -> Any:
    """Allocate zero readings in the template's complete unit or coordinate."""
    _require_quantity(a)
    return np.zeros_like(a, dtype, order, subok, shape, device=device)


def ones_like(
    a: Any,
    dtype: Any = None,
    order: Literal['K', 'A', 'C', 'F'] = 'K',
    subok: bool = True,
    shape: Any = None,
    *,
    device: Any = None,
) -> Any:
    """Allocate one reading per element, retaining the template's complete unit."""
    _require_quantity(a)
    return np.ones_like(a, dtype, order, subok, shape, device=device)


def empty_like(
    a: Any,
    dtype: Any = None,
    order: Literal['K', 'A', 'C', 'F'] = 'K',
    subok: bool = True,
    shape: Any = None,
    *,
    device: Any = None,
) -> Any:
    """Allocate uninitialized storage in the template's unit or coordinate."""
    _require_quantity(a)
    return np.empty_like(a, dtype, order, subok, shape, device=device)


def full_like(
    a: Any,
    fill_value: Any,
    dtype: Any = None,
    order: Literal['K', 'A', 'C', 'F'] = 'K',
    subok: bool = True,
    shape: Any = None,
    *,
    device: Any = None,
) -> Any:
    """Fill readings in the template's unit or coordinate, retaining its family.

    Bare fills are stored readings, not coherent dimensionless values. Quantity
    fills convert to the template unit; point fills require compatible points.
    Dtype, shape, order and casting follow the existing NumPy allocation rules.
    """
    _require_quantity(a)
    return np.full_like(a, fill_value, dtype, order, subok, shape, device=device)


def trapezoid(y: Any, x: Any = None, dx: Any = 1.0, axis: int = -1) -> Any:
    """Integrate linear Quantity samples, returning sample unit times coordinate unit.

    With `x`, its unit supplies the coordinate unit and `dx` is ignored.
    Otherwise `dx` must be scalar; bare spacing is dimensionless. Numerical
    shapes, axis, orientation and sampling follow `numpy.trapezoid`. Points
    are unsupported. See `docs/numpy/calculus.md` for the runtime contract.
    """
    if not isinstance(y, Quantity):
        error_message = 'trapezoid requires linear Quantity samples.'
        raise TypeError(error_message)
    return np.trapezoid(y, x=x, dx=dx, axis=axis)


def gradient(f: Any, *varargs: Any, axis: Any = None, edge_order: Literal[1, 2] = 1) -> Any:
    """Differentiate wrapped values, using difference units for point coordinates.

    Each result has the input difference unit divided by its spacing unit.
    Bare spacing is dimensionless. A single selected axis returns one Quantity;
    multiple axes return a tuple. NumPy controls shapes and numerical kernels.
    See `docs/numpy/calculus.md` for coordinate and boundary rules.
    """
    _require_quantity(f)
    return np.gradient(f, *varargs, axis=axis, edge_order=edge_order)
