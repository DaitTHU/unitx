"""Indexed assignment, staged fixed-unit outputs and batch writes.

Import operation dispatch only when a ufunc method needs a numerical kernel.
Buffer preparation and final writes do not depend on operation routing.
"""

from typing import Any, Literal

import numpy as np

from .._coordinates.resolve import _Coordinate
from .._coordinates.resolve import convert_point as convert
from .._operands import _Point
from ..dimension import DIMENSIONLESS
from ..exceptions import DimensionError
from ..quantity import Quantity
from ..unit import Unit
from .numpy_operands import (
    _align_operands,
    _boolean_condition,
    _is_supported_operand,
    _linear_parts,
    _numeric_value,
    _operand_parts,
    _selection_parts,
    _snapshot_index,
    _validate_dtype,
    _validate_where,
    _wrap,
    scale_magnitude,
)

_EXTREMA = (np.minimum, np.maximum, np.fmin, np.fmax)
_ADDITIVE = (np.add, np.subtract, np.remainder, np.fmod)


def assign_item(target: Quantity[Any] | _Point[Any], key: Any, value: Any) -> None:
    """Convert and stage selected values before modifying caller storage."""
    buffer = target.value
    if type(buffer) is not np.ndarray:
        error_message = 'Indexed assignment requires a plain numeric ndarray magnitude.'
        raise TypeError(error_message)
    if not buffer.flags.writeable:
        error_message = 'Assignment destination is read-only.'
        raise ValueError(error_message)
    if not _is_supported_operand(value):
        error_message = 'Assignment requires a numeric scalar, ndarray, or Quantity.'
        raise TypeError(error_message)
    raw, unit = _operand_parts(value)
    if unit.dimension != target.dimension:
        error_message = 'Assigned value has an incompatible dimension.'
        raise DimensionError(error_message)
    key = _snapshot_index(key)
    if isinstance(unit, _Coordinate) != isinstance(target.unit, _Coordinate):
        error_message = (
            'Assignment requires matching point/difference semantics; convert explicitly.'
        )
        raise DimensionError(error_message)
    raw = _numeric_value(convert(raw, unit, target.unit))
    staged = np.empty(np.shape(raw), dtype=buffer.dtype)
    # Resolve every cast before entering NumPy indexed assignment, which leaves
    # its target unchanged on index/shape errors but not on casting errors.
    # Preserve RHS shape so native scalar/boolean assignment rules still apply.
    np.copyto(staged, raw, casting='same_kind')
    buffer[key] = staged


def write_inplace(target: Quantity[Any] | _Point[Any], result: Quantity[Any] | _Point[Any]) -> Any:
    # NumPy scalars stay immutable and follow ordinary result/rebinding rules.
    if type(target.value) is not np.ndarray:
        return result
    return _write_output(result, target, broadcast=True)


def _write_output(
    result: Any,
    target: Any,
    *,
    where: Any = True,
    casting: Any = 'same_kind',
    broadcast: bool = False,
) -> Any:
    """Stage a single output without replacing its magnitude, dtype, or unit."""
    if target is None:
        return result
    buffer, staged, mask = _stage_output(
        result, target, where=where, casting=casting, broadcast=broadcast
    )
    np.copyto(buffer, staged, where=mask, casting='no')
    return target


def _write_outputs(
    result: tuple[Any, ...],
    outputs: tuple[Any, ...],
    *,
    where: Any = True,
    casting: Any = 'same_kind',
) -> tuple[Any, ...]:
    """Stage both outputs of a supported ufunc before writing either buffer."""
    buffers = [
        x.value if isinstance(x, (Quantity, _Point)) else x for x in outputs if x is not None
    ]
    if any(type(buffer) is not np.ndarray for buffer in buffers):
        error_message = 'Quantity out requires a plain numeric ndarray buffer.'
        raise TypeError(error_message)
    if len(buffers) == 2 and np.shares_memory(*buffers):
        error_message = 'Multiple output buffers must not overlap.'
        raise ValueError(error_message)
    # All outputs share the ufunc broadcast shape, including newly allocated
    # outputs. Do not broadcast one provided output into another's larger shape.
    shape = np.broadcast_shapes(np.shape(result[0].value), *(x.shape for x in buffers))
    if any(buffer.shape != shape for buffer in buffers):
        error_message = 'All output buffers must have the ufunc broadcast shape.'
        raise ValueError(error_message)
    if type(where) is np.ndarray:
        # A mask can alias either output; retain it across both final writes.
        where = where.copy()
    returned = []
    staged = []
    for value, target in zip(result, outputs):
        raw = value.value if isinstance(value, (Quantity, _Point)) else value
        if np.shape(raw) != shape:
            raw = np.broadcast_to(raw, shape).copy()
            value = _wrap(raw, value.unit) if isinstance(value, (Quantity, _Point)) else raw
        if target is not None:
            staged.append(_stage_output(value, target, where=where, casting=casting))
        returned.append(value if target is None else target)
    # No caller storage changes until every output has converted and cast.
    for buffer, temporary, mask in staged:
        np.copyto(buffer, temporary, where=mask, casting='no')
    return tuple(returned)


def _stage_output(
    result: Any,
    target: Any,
    *,
    where: Any = True,
    casting: Any = 'same_kind',
    broadcast: bool = False,
) -> tuple[Any, Any, Any]:
    if isinstance(result, (Quantity, _Point)):
        if not isinstance(target, (Quantity, _Point)) or type(target.value) is not np.ndarray:
            error_message = 'Quantity out requires a plain numeric ndarray buffer.'
            raise TypeError(error_message)
        if result.dimension != target.dimension:
            error_message = 'out has an incompatible dimension.'
            raise DimensionError(error_message)
        if isinstance(result.unit, _Coordinate) != isinstance(target.unit, _Coordinate):
            error_message = 'out must preserve point/difference semantics.'
            raise DimensionError(error_message)
        buffer = target.value
    else:
        kinds = 'b' if np.asarray(result).dtype.kind == 'b' else 'iu'
        if type(target) is not np.ndarray or target.dtype.kind not in kinds:
            error_message = 'Unitless out requires a plain ndarray with the matching result kind.'
            raise TypeError(error_message)
        buffer = target
    if not buffer.flags.writeable:
        error_message = 'out buffer is read-only.'
        raise ValueError(error_message)
    value = result.value if isinstance(result, (Quantity, _Point)) else result
    if not broadcast and np.shape(value) != buffer.shape:
        error_message = 'out must have exactly the result shape.'
        raise ValueError(error_message)
    if isinstance(result, (Quantity, _Point)) and isinstance(target, (Quantity, _Point)):
        if isinstance(result.unit, _Coordinate):
            if where is not True:
                value = np.where(where, value, 0)
            value = convert(value, result.unit, target.unit)
        else:
            value = scale_magnitude(value, result.unit.factor / target.unit.factor, where=where)
    # Validate broadcasting and casting before touching shared storage. Casting
    # also runs in temporary storage so e.g. float narrowing errors cannot
    # partially overwrite the caller's buffer.
    value = np.broadcast_to(value, buffer.shape)
    mask = np.broadcast_to(where, buffer.shape)
    if np.may_share_memory(mask, buffer):
        mask = mask.copy()
    # Kernels and nonidentity conversions usually already own a temporary.
    # Reuse it when no cast or separation from the destination is needed.
    # can_cast still validates an explicit casting option, even for equal dtypes.
    if (
        value.dtype == buffer.dtype
        and np.can_cast(value.dtype, buffer.dtype, casting=casting)
        and not np.may_share_memory(value, buffer)
    ):
        return buffer, value, mask
    staged = np.empty(buffer.shape, dtype=buffer.dtype)
    np.copyto(staged, value, where=mask, casting=casting)
    return buffer, staged, mask


def _buffer(target: Any) -> Any:
    if not isinstance(target, (Quantity, _Point)) or type(target.value) is not np.ndarray:
        error_message = 'Writes require a Quantity or point with a plain ndarray magnitude.'
        raise TypeError(error_message)
    if not target.value.flags.writeable:
        error_message = 'Write destination is read-only.'
        raise ValueError(error_message)
    return target.value


def _copyto(dst: Any, src: Any, casting: str = 'same_kind', where: Any = True) -> None:
    _buffer(dst)
    _validate_where(where)
    if not _is_supported_operand(src):
        error_message = 'copyto requires a numeric scalar, ndarray, quantity or point source.'
        raise TypeError(error_message)
    raw, unit = _operand_parts(src)
    _write_output(_wrap(raw, unit), dst, casting=casting, where=where, broadcast=True)


def _write_values(target: Any, values: Any) -> Any:
    _, raw, _ = _selection_parts(target, values)
    staged = np.empty(np.shape(raw), dtype=target.value.dtype)
    np.copyto(staged, raw, casting='same_kind')
    return staged


def _put(a: Any, ind: Any, v: Any, mode: Literal['raise', 'wrap', 'clip'] = 'raise') -> None:
    buffer = _buffer(a)
    indices = _snapshot_index(ind)
    values = _write_values(a, v)
    staged = buffer.copy(order='K')
    np.put(staged, indices, values, mode=mode)
    np.copyto(buffer, staged, casting='no')


def _put_along_axis(arr: Any, indices: Any, values: Any, axis: Any) -> None:
    buffer = _buffer(arr)
    indices = _snapshot_index(indices)
    values = _write_values(arr, values)
    staged = buffer.copy(order='K')
    np.put_along_axis(staged, indices, values, axis=axis)
    np.copyto(buffer, staged, casting='no')


def _masked_write(func: Any, a: Any, mask: Any, values: Any) -> None:
    buffer = _buffer(a)
    mask = np.array(_boolean_condition(mask), copy=True)
    values = _write_values(a, values)
    staged = buffer.copy(order='K')
    func(staged, mask, values)
    np.copyto(buffer, staged, casting='no')


def _fill_diagonal(a: Any, val: Any, wrap: bool = False) -> None:
    buffer = _buffer(a)
    values = _write_values(a, val)
    staged = buffer.copy(order='K')
    np.fill_diagonal(staged, values, wrap=wrap)
    np.copyto(buffer, staged, casting='no')


def _at(ufunc: Any, inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if kwargs or len(inputs) != ufunc.nin + 1 or ufunc.nout != 1:
        return NotImplemented
    target, indices, *operands = inputs
    if not isinstance(target, (Quantity, _Point)) or not all(
        _is_supported_operand(rhs) for rhs in operands
    ):
        return NotImplemented
    from .numpy import dispatch_ufunc

    buffer = _buffer(target)
    indices = _snapshot_index(indices)
    raw = None
    if operands:
        rhs = operands[0]
        if ufunc in _EXTREMA:
            _, raw, _ = _selection_parts(target, rhs)
        elif ufunc in (np.add, np.subtract) and isinstance(target, _Point):
            from .._coordinates.points import _addition_parts

            _, raw, result_unit = _addition_parts(target, rhs, subtract=ufunc is np.subtract)
            if result_unit != target.unit:
                error_message = 'ufunc.at must preserve the destination coordinate.'
                raise DimensionError(error_message)
        elif ufunc in _ADDITIVE:
            _, raw, _ = _align_operands(target, rhs)
        elif ufunc in (np.multiply, np.divide, np.true_divide):
            _linear_parts(target)
            raw, unit = _linear_parts(rhs)
            if unit.dimension != DIMENSIONLESS:
                error_message = 'Multiplicative ufunc.at requires a dimensionless multiplier.'
                raise DimensionError(error_message)
            raw = scale_magnitude(raw, unit.factor)
        elif ufunc is np.copysign:
            _linear_parts(target)
            raw, _ = _linear_parts(rhs)
        else:
            return NotImplemented
    elif ufunc not in (np.positive, np.floor, np.ceil, np.trunc, np.rint):
        if ufunc not in (np.negative, np.absolute, np.fabs, np.conjugate):
            return NotImplemented
        _linear_parts(target)
    if raw is not None:
        raw = np.array(raw, copy=True)
        # at uses an unsafe inner cast; enforce the fixed-buffer policy first.
        selected = _wrap(buffer[indices], target.unit)
        if operands and ufunc in (np.multiply, np.divide, np.true_divide):
            operand = Quantity(raw, Unit())
        elif ufunc is np.copysign:
            operand = Quantity(raw, Unit())
        else:
            operand = _wrap(
                raw,
                target.unit.delta_unit
                if isinstance(target, _Point) and ufunc in (np.add, np.subtract)
                else target.unit,
            )
        preview = dispatch_ufunc(ufunc, '__call__', (selected, operand), {})
    else:
        preview = dispatch_ufunc(ufunc, '__call__', (_wrap(buffer[indices], target.unit),), {})
    # Validate calculation dtype without replacing the kernel's RHS dtype.
    check = np.empty(np.shape(preview.value), dtype=buffer.dtype)
    np.copyto(check, preview.value, casting='same_kind')
    staged = buffer.copy(order='K')
    ufunc.at(staged, indices, *(() if not operands else (raw,)))
    np.copyto(buffer, staged, casting='no')
    return None


def dispatch_method(
    ufunc: Any, method: str, inputs: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    if method == 'at':
        return _at(ufunc, inputs, kwargs)
    from .numpy import _prod, _sum, dispatch_ufunc

    if method == 'outer':
        if ufunc.nin != 2 or ufunc.nout != 1 or ufunc.signature is not None:
            return NotImplemented
        if not all(_is_supported_operand(x) for x in inputs):
            return NotImplemented
        a, b = inputs
        av, bv = (
            np.asarray(_numeric_value(a.value if hasattr(a, 'unit') else a)),
            np.asarray(_numeric_value(b.value if hasattr(b, 'unit') else b)),
        )
        av = av.reshape(av.shape + (1,) * bv.ndim)
        bv = bv.reshape((1,) * np.ndim(a.value if hasattr(a, 'unit') else a) + bv.shape)
        return dispatch_ufunc(
            ufunc,
            '__call__',
            (
                _wrap(av, a.unit) if isinstance(a, (Quantity, _Point)) else av,
                _wrap(bv, b.unit) if isinstance(b, (Quantity, _Point)) else bv,
            ),
            kwargs,
        )
    if method not in ('reduce', 'accumulate', 'reduceat') or ufunc.nout != 1:
        return NotImplemented
    if len(inputs) != (2 if method == 'reduceat' else 1):
        return NotImplemented
    q = inputs[0]
    if not isinstance(q, (Quantity, _Point)):
        return NotImplemented
    options = kwargs.copy()
    outputs = options.pop('out', (None,))
    if len(outputs) != 1:
        return NotImplemented
    out = outputs[0]
    if out is not None and not isinstance(out, (Quantity, _Point)):
        return NotImplemented
    if method == 'reduce' and ufunc in (np.add, np.multiply):
        handler = _sum if ufunc is np.add else _prod
        return handler(ufunc.reduce, q, out=out, **{'axis': 0, **options})
    allowed = {'axis', 'dtype'}
    if method == 'reduce':
        allowed |= {'keepdims', 'initial', 'where'}
    if options.keys() - allowed:
        return NotImplemented
    _validate_dtype(options.get('dtype'))
    _validate_where(options.get('where', True))
    values, unit = _operand_parts(q)
    if ufunc in _EXTREMA:
        if options.get('initial') is not None:
            _, options['initial'], _ = _selection_parts(q, options['initial'])
    elif ufunc in (np.add, np.subtract, np.remainder, np.fmod):
        _linear_parts(q)
        if options.get('initial') is not None:
            _, options['initial'], _ = _align_operands(q, options['initial'])
    elif ufunc in (
        np.multiply,
        np.divide,
        np.true_divide,
        np.floor_divide,
        np.logaddexp,
        np.logaddexp2,
    ):
        values, unit = _linear_parts(q)
        if unit.dimension != DIMENSIONLESS:
            error_message = 'This ufunc method requires dimensionless input.'
            raise DimensionError(error_message)
        values, unit = scale_magnitude(values, unit.factor), Unit()
        if options.get('initial') is not None:
            initial, initial_unit = _linear_parts(options['initial'])
            if initial_unit.dimension != DIMENSIONLESS:
                error_message = 'initial must be dimensionless.'
                raise DimensionError(error_message)
            options['initial'] = scale_magnitude(initial, initial_unit.factor)
    else:
        return NotImplemented
    extra = (_snapshot_index(inputs[1]),) if method == 'reduceat' else ()
    result = getattr(ufunc, method)(values, *extra, **options)
    return _write_output(_wrap(result, unit), out)


FUNCTIONS = {
    np.copyto: _copyto,
    np.put: _put,
    np.put_along_axis: _put_along_axis,
    np.putmask: lambda a, mask, values: _masked_write(np.putmask, a, mask, values),
    np.place: lambda arr, mask, vals: _masked_write(np.place, arr, mask, vals),
    np.fill_diagonal: _fill_diagonal,
}
