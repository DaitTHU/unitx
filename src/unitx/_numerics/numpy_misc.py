"""Remaining array preparation, predicate and cumulative adapters."""

from typing import Any

import numpy as np

from .._operands import _Point
from ..dimension import DIMENSIONLESS
from ..exceptions import DimensionError
from ..quantity import Quantity
from ..unit import Unit
from .numpy import (
    _array_transform,
    _boolean_condition,
    _delta_unit,
    _join_values,
    _linear_parts,
    _operand_parts,
    _selection_parts,
    _snapshot_index,
    _validate_dtype,
    _validate_where,
    _wrap,
    _write_output,
    scale_magnitude,
)
from .numpy_creation import _readings


def _block(arrays: Any) -> Any:
    leaves = []

    def collect(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                collect(item)
        else:
            leaves.append(node)

    collect(arrays)
    values, unit = _join_values(leaves)
    items = iter(values)

    def rebuild(node: Any) -> Any:
        return [rebuild(item) for item in node] if isinstance(node, list) else next(items)

    return _wrap(np.block(rebuild(arrays)), unit)


def _nan_to_num(
    x: Any,
    copy: bool = True,
    nan: Any = 0.0,
    posinf: Any = None,
    neginf: Any = None,
) -> Any:
    values, unit = _operand_parts(x)
    result = _wrap(
        np.nan_to_num(
            values,
            copy=True,
            nan=_readings(nan, unit),
            posinf=None if posinf is None else _readings(posinf, unit),
            neginf=None if neginf is None else _readings(neginf, unit),
        ),
        unit,
    )
    if not copy and type(values) is np.ndarray:
        return _write_output(result, x)
    return result


def _ediff1d(ary: Any, to_end: Any = None, to_begin: Any = None) -> Any:
    values, unit = _operand_parts(ary)
    delta = _delta_unit(unit)
    options = {}
    for key, value in (('to_end', to_end), ('to_begin', to_begin)):
        if value is not None:
            _, raw, _ = _selection_parts(Quantity(0, delta), value)
            options[key] = raw
    return Quantity(np.ediff1d(values, **options), delta)


def _cumulative(
    func: Any,
    x: Any,
    /,
    *,
    axis: Any = None,
    dtype: Any = None,
    out: Any = None,
    include_initial: bool = False,
) -> Any:
    _validate_dtype(dtype)
    values, unit = _linear_parts(x)
    if func is np.cumulative_prod:
        if unit.dimension != DIMENSIONLESS:
            error_message = 'Cumulative products require dimensionless input.'
            raise DimensionError(error_message)
        values, unit = scale_magnitude(values, unit.factor), Unit()
    result = Quantity(func(values, axis=axis, dtype=dtype, include_initial=include_initial), unit)
    return _write_output(result, out)


def _array_equality(func: Any, a1: Any, a2: Any, **kwargs: Any) -> Any:
    if isinstance(a1, _Point) != isinstance(a2, _Point):
        return False
    try:
        a, b, _ = _selection_parts(a1, a2)
    except DimensionError:
        return False
    return func(a, b, **kwargs)


def _array_equal(a1: Any, a2: Any, equal_nan: bool = False) -> Any:
    return _array_equality(np.array_equal, a1, a2, equal_nan=equal_nan)


def _predicate(func: Any, x: Any) -> Any:
    values, _ = _operand_parts(x)
    return func(values)


def _memory(func: Any, a: Any, b: Any, /, max_work: Any = None) -> Any:
    a, _ = _operand_parts(a)
    b, _ = _operand_parts(b)
    return func(a, b) if max_work is None else func(a, b, max_work=max_work)


def _truth(
    func: Any,
    a: Any,
    axis: Any = None,
    out: Any = None,
    keepdims: bool = False,
    *,
    where: Any = True,
) -> Any:
    _validate_where({'where': where})
    values, _ = _operand_parts(a)
    result = func(values, axis=axis, keepdims=keepdims, where=where)
    return _write_output(result, out)


def _choose(a: Any, choices: Any, out: Any = None, mode: Any = 'raise') -> Any:
    values, unit = _join_values(choices)
    result = _wrap(np.choose(_snapshot_index(a), values, mode=mode), unit)
    return _write_output(result, out)


def _select(condlist: Any, choicelist: Any, default: Any = 0) -> Any:
    values, unit = _join_values(choicelist)
    conditions = [_boolean_condition(condition) for condition in condlist]
    return _wrap(np.select(conditions, values, default=_readings(default, unit)), unit)


def _unstack(x: Any, /, *, axis: Any = 0) -> Any:
    values, unit = _operand_parts(x)
    return tuple(_wrap(view, unit) for view in np.unstack(values, axis=axis))


def _i0(x: Any) -> Any:
    values, unit = _linear_parts(x)
    if unit.dimension != DIMENSIONLESS:
        error_message = 'i0 requires dimensionless input.'
        raise DimensionError(error_message)
    return Quantity(np.i0(scale_magnitude(values, unit.factor)), Unit())


def _infinity_predicate(func: Any, x: Any, out: Any = None) -> Any:
    values, _ = _operand_parts(x)
    return _write_output(func(values), out)


def dispatch(func: Any, *args: Any, **kwargs: Any) -> Any:
    if func is np.choose:
        return _choose(*args, **kwargs)
    if func is np.select:
        return _select(*args, **kwargs)
    if func is np.unstack:
        return _unstack(*args, **kwargs)
    if func is np.i0:
        return _i0(*args, **kwargs)
    if func in (np.isposinf, np.isneginf):
        return _infinity_predicate(func, *args, **kwargs)
    if func is np.block:
        return _block(*args, **kwargs)
    if func is np.nan_to_num:
        return _nan_to_num(*args, **kwargs)
    if func is np.ediff1d:
        return _ediff1d(*args, **kwargs)
    if func in (np.cumulative_sum, np.cumulative_prod):
        return _cumulative(func, *args, **kwargs)
    if func is np.array_equal:
        return _array_equal(*args, **kwargs)
    if func is np.array_equiv:
        return _array_equality(func, *args, **kwargs)
    if func in (np.isreal, np.iscomplex, np.isrealobj, np.iscomplexobj):
        return _predicate(func, *args, **kwargs)
    if func in (np.shares_memory, np.may_share_memory):
        return _memory(func, *args, **kwargs)
    if func in (np.all, np.any):
        return _truth(func, *args, **kwargs)
    return _array_transform(func, *args, **kwargs)
