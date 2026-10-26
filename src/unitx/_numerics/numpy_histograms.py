"""NumPy histogram coordinates, edges, weights and density units."""

from typing import Any, cast

import numpy as np  # pyright: ignore[reportMissingImports]

from .._operands import _Point
from ..quantity import Quantity
from ..unit import Unit
from .numpy import (
    _call_with_warnings,
    _delta_unit,
    _is_supported_operand,
    _linear_parts,
    _operand_parts,
    _selection_parts,
    _wrap,
)


def _sample(value: Any) -> tuple[Any, Any]:
    if not _is_supported_operand(value):
        error_message = 'Histogram samples require plain numeric arrays or quantities/points.'
        raise TypeError(error_message)
    return _operand_parts(value)


def _aligned(value: Any, unit: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return np.asarray([_aligned(item, unit) for item in value])
    return _selection_parts(_wrap(0, unit), value)[1]


def _bins(value: Any, unit: Any) -> Any:
    if isinstance(value, (str, int, np.integer)):
        return value
    return _aligned(value, unit)


def _weights(value: Any) -> tuple[Any, Unit | None]:
    if value is None:
        return None, None
    if isinstance(value, (list, tuple)):
        value = np.asarray(value)
    if not _is_supported_operand(value):
        error_message = 'Histogram weights require numeric arrays or linear quantities.'
        raise TypeError(error_message)
    raw, unit = _linear_parts(value)
    return raw, unit if isinstance(value, Quantity) else None


def _histogram(
    a: Any, bins: Any = 10, range: Any = None, density: Any = None, weights: Any = None
) -> Any:
    values, unit = _sample(a)
    weight_values, weight_unit = _weights(weights)
    hist, edges = _call_with_warnings(
        np.histogram,
        values,
        bins=_bins(bins, unit),
        range=None if range is None else _aligned(range, unit),
        density=density,
        weights=weight_values,
    )
    result_unit = _delta_unit(unit) ** -1 if density else weight_unit
    return (hist if result_unit is None else Quantity(hist, result_unit)), _wrap(edges, unit)


def _histogram_bin_edges(a: Any, bins: Any = 10, range: Any = None, weights: Any = None) -> Any:
    values, unit = _sample(a)
    weight_values, _ = _weights(weights)
    edges = _call_with_warnings(
        np.histogram_bin_edges,
        values,
        bins=_bins(bins, unit),
        range=None if range is None else _aligned(range, unit),
        weights=weight_values,
    )
    return _wrap(edges, unit)


def _histogramdd(
    sample: Any, bins: Any = 10, range: Any = None, density: Any = None, weights: Any = None
) -> Any:
    if isinstance(sample, (Quantity, _Point)) or type(sample) is np.ndarray:
        values, unit = _sample(sample)
        if np.ndim(values) != 2:
            error_message = 'Histogram matrix samples must have shape (N, D).'
            raise ValueError(error_message)
        units = [unit] * values.shape[1]
    else:
        parts = [_sample(column) for column in sample]
        values, units = [p[0] for p in parts], [p[1] for p in parts]
    dimensions = len(units)
    if isinstance(bins, (int, np.integer)):
        raw_bins = bins
    else:
        specs = list(bins)
        if len(specs) != dimensions:
            error_message = 'Histogram bins must match the number of coordinates.'
            raise ValueError(error_message)
        raw_bins = [_bins(spec, unit) for spec, unit in zip(specs, units)]
    raw_range = None
    if range is not None:
        ranges = list(range)
        if len(ranges) != dimensions:
            error_message = 'Histogram range must match the number of coordinates.'
            raise ValueError(error_message)
        raw_range = [
            None if bounds is None else _aligned(bounds, unit)
            for bounds, unit in zip(ranges, units)
        ]
    weight_values, weight_unit = _weights(weights)
    hist, edges = _call_with_warnings(
        np.histogramdd,
        values,
        bins=raw_bins,
        range=cast(Any, raw_range),
        density=density,
        weights=weight_values,
    )
    result_unit = weight_unit
    if density:
        result_unit = Unit()
        for unit in units:
            result_unit /= _delta_unit(unit)
    return (hist if result_unit is None else Quantity(hist, result_unit)), [
        _wrap(edge, unit) for edge, unit in zip(edges, units)
    ]


def _histogram2d(
    x: Any, y: Any, bins: Any = 10, range: Any = None, density: Any = None, weights: Any = None
) -> Any:
    # NumPy interprets a flat edge vector of length > 2 as shared edges.
    try:
        count = len(bins)
    except TypeError:
        count = 1
    if count not in (1, 2):
        bins = [bins, bins]
    hist, edges = _histogramdd([x, y], bins, range, density, weights)
    return hist, edges[0], edges[1]


def _digitize(x: Any, bins: Any, right: bool = False) -> Any:
    _, unit = _sample(x)
    return _call_with_warnings(np.digitize, _aligned(x, unit), _aligned(bins, unit), right=right)


def _bincount(x: Any, weights: Any = None, minlength: int = 0) -> Any:
    if isinstance(x, (Quantity, _Point)):
        error_message = 'bincount indices must be bare integers.'
        raise TypeError(error_message)
    values, unit = _weights(weights)
    result = _call_with_warnings(np.bincount, x, weights=values, minlength=minlength)
    return result if unit is None else Quantity(result, unit)


_FUNCTIONS = {
    np.histogram: _histogram,
    np.histogram_bin_edges: _histogram_bin_edges,
    np.histogram2d: _histogram2d,
    np.histogramdd: _histogramdd,
    np.digitize: _digitize,
    np.bincount: _bincount,
}


def dispatch(func: Any, *args: Any, **kwargs: Any) -> Any:
    return _FUNCTIONS[func](*args, **kwargs)
