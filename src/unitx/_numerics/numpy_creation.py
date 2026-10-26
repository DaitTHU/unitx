"""Unit-aware allocation, sample grids, padding and window views."""

from typing import Any, Literal, cast

import numpy as np

from .._operands import _Point
from ..dimension import DIMENSIONLESS
from ..exceptions import DimensionError
from ..quantity import Quantity
from ..unit import Unit
from .numpy import (
    _delta_unit,
    _is_supported_operand,
    _linear_parts,
    _operand_parts,
    _selection_parts,
    _validate_dtype,
    _wrap,
    scale_magnitude,
)


def _readings(value: Any, unit: Any) -> Any:
    """Bare fill values are readings; wrapped values align to the template."""
    if isinstance(value, (Quantity, _Point)):
        _, raw, _ = _selection_parts(_wrap(0, unit), value)
        return raw
    if isinstance(value, (tuple, list)):
        return type(value)(_readings(item, unit) for item in value)
    if not _is_supported_operand(value):
        error_message = 'Fill values must be numeric readings or compatible quantities/points.'
        raise TypeError(error_message)
    return _operand_parts(value)[0]


def _full_like(
    a: Any,
    fill_value: Any,
    dtype: Any = None,
    order: Literal['K', 'A', 'C', 'F'] = 'K',
    subok: bool = True,
    shape: Any = None,
    *,
    device: Any = None,
) -> Any:
    _validate_dtype(dtype)
    if not _is_supported_operand(a):
        error_message = 'full_like requires a numeric template.'
        raise TypeError(error_message)
    values, unit = _operand_parts(a)
    raw = _readings(fill_value, unit)
    return _wrap(np.full_like(values, raw, dtype, order, subok, shape, device=device), unit)


def _linspace(
    start: Any,
    stop: Any,
    num: int = 50,
    endpoint: bool = True,
    retstep: bool = False,
    dtype: Any = None,
    axis: int = 0,
    *,
    device: Any = None,
) -> Any:
    _validate_dtype(dtype)
    a, b, unit = _selection_parts(start, stop)
    result = np.linspace(a, b, num, endpoint, cast(Any, retstep), dtype, axis, device=device)
    if retstep:
        values, step = result
        return _wrap(values, unit), Quantity(step, _delta_unit(unit))
    return _wrap(result, unit)


def _geomspace(
    start: Any,
    stop: Any,
    num: int = 50,
    endpoint: bool = True,
    dtype: Any = None,
    axis: int = 0,
) -> Any:
    _validate_dtype(dtype)
    _linear_parts(start)
    _linear_parts(stop)
    a, b, unit = _selection_parts(start, stop)
    return _wrap(np.geomspace(a, b, num, endpoint, dtype, axis), unit)


def _dimensionless(value: Any) -> Any:
    if not _is_supported_operand(value):
        error_message = 'logspace requires numeric dimensionless inputs.'
        raise TypeError(error_message)
    raw, unit = _linear_parts(value)
    if unit.dimension != DIMENSIONLESS:
        error_message = 'logspace exponents and base must be dimensionless.'
        raise DimensionError(error_message)
    return scale_magnitude(raw, unit.factor)


def _logspace(
    start: Any,
    stop: Any,
    num: int = 50,
    endpoint: bool = True,
    base: Any = 10.0,
    dtype: Any = None,
    axis: int = 0,
) -> Any:
    _validate_dtype(dtype)
    return Quantity(
        np.logspace(
            _dimensionless(start),
            _dimensionless(stop),
            num,
            endpoint,
            _dimensionless(base),
            dtype,
            axis,
        ),
        Unit(),
    )


def _meshgrid(
    *xi: Any, copy: bool = True, sparse: bool = False, indexing: Literal['ij', 'xy'] = 'xy'
) -> Any:
    if not all(_is_supported_operand(x) for x in xi):
        error_message = 'meshgrid requires numeric arrays, quantities or points.'
        raise TypeError(error_message)
    values = [_operand_parts(x)[0] for x in xi]
    grids = np.meshgrid(*values, copy=copy, sparse=sparse, indexing=indexing)
    return tuple(
        _wrap(grid, x.unit) if isinstance(x, (Quantity, _Point)) else grid
        for grid, x in zip(grids, xi)
    )


def _pad(array: Any, pad_width: Any, mode: str = 'constant', **kwargs: Any) -> Any:
    if mode not in (
        'constant',
        'edge',
        'linear_ramp',
        'maximum',
        'mean',
        'median',
        'minimum',
        'reflect',
        'symmetric',
        'wrap',
        'empty',
    ):
        error_message = 'Quantity pad supports NumPy built-in modes, not callbacks.'
        raise TypeError(error_message)
    values, unit = _operand_parts(array)
    for key in ('constant_values', 'end_values'):
        if key in kwargs:
            kwargs[key] = _readings(kwargs[key], unit)
    return _wrap(np.pad(values, pad_width, mode, **kwargs), unit)


def _sliding_window_view(
    x: Any,
    window_shape: Any,
    axis: Any = None,
    *,
    subok: bool = False,
    writeable: bool = False,
) -> Any:
    values, unit = _operand_parts(x)
    return _wrap(
        np.lib.stride_tricks.sliding_window_view(
            values,
            window_shape,
            axis,
            subok=subok,
            writeable=writeable,
        ),
        unit,
    )


FUNCTIONS = {
    np.full_like: _full_like,
    np.linspace: _linspace,
    np.geomspace: _geomspace,
    np.logspace: _logspace,
    np.meshgrid: _meshgrid,
    np.pad: _pad,
    np.lib.stride_tricks.sliding_window_view: _sliding_window_view,
}
