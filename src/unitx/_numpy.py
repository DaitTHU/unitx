"""Optional NumPy kernels. Imported only for NumPy magnitudes or dispatch.

An identity scale retains storage. Integral scales use a Python int multiplier;
nonintegral rational scales use one Python float multiplier, and irrational
scales use Factor's approximation. NumPy then controls promotion and errors.
No Fraction/Decimal object arrays are produced. Explicit outputs write
only after computing, converting, and casting a temporary result.
"""

import math
import operator
from decimal import Decimal
from fractions import Fraction
from functools import partial
from typing import Any, Literal

import numpy as np  # pyright: ignore[reportMissingImports]

from ._numerics import as_exponent
from .exceptions import DimensionError
from .factor import ONE, Factor
from .quantity import Quantity
from .unit import Unit

_COMPARISON_UFUNCS = {
    operator.eq: np.equal,
    operator.ne: np.not_equal,
    operator.lt: np.less,
    operator.le: np.less_equal,
    operator.gt: np.greater,
    operator.ge: np.greater_equal,
}


def _is_numpy_magnitude(value: Any) -> bool:
    return (
        type(value) is np.ndarray
        or isinstance(value, np.number)
        and type(value).__module__ == 'numpy'
    ) and value.dtype.kind in 'iufc'


def validate_magnitude(value: Any) -> None:
    if not _is_numpy_magnitude(value):
        raise TypeError('Only plain numeric ndarrays and NumPy numeric scalars are supported.')


def _is_supported_operand(value: Any) -> bool:
    if isinstance(value, Quantity):
        value = value.value
    return type(value) in (int, float, complex, Decimal, Fraction) or _is_numpy_magnitude(value)


def _numeric_value(value: Any) -> Any:
    # NumPy has no native Decimal/Fraction dtype. Enter its floating domain at
    # this boundary, leaving scalar Quantity arithmetic and constants intact.
    return float(value) if isinstance(value, (Decimal, Fraction)) else value


def scale_magnitude(value: Any, factor: Factor, *, where: Any = True) -> Any:
    if not _is_supported_operand(value):
        raise TypeError('NumPy operations require numeric arrays or supported numeric scalars.')
    value = _numeric_value(value)
    if factor == ONE:
        return value
    rational, irrational = factor.decompose()
    multiplier = (
        (rational.numerator if rational.denominator == 1 else float(rational))
        if irrational == ONE
        else float(factor.value)
    )
    # Do not pass a Fraction to NumPy (it would create object dtype).
    return np.multiply(value, multiplier, where=where, out=None)


def _operand_parts(value: Any) -> tuple[Any, Unit]:
    if isinstance(value, Quantity):
        return _numeric_value(value.value), value.unit
    if isinstance(value, Unit):
        return 1, value
    return _numeric_value(value), Unit()


def _align_operands(left: Any, right: Any) -> tuple[Any, Any, Unit]:
    a, u = _operand_parts(left)
    b, v = _operand_parts(right)
    if u.dimension != v.dimension:
        raise DimensionError('Cannot operate on quantities with different dimensions.')
    # Addition retains the first Quantity's unit, even with a bare left input.
    target = u if isinstance(left, Quantity) else v
    return (
        scale_magnitude(a, u.factor / target.factor),
        scale_magnitude(b, v.factor / target.factor),
        target,
    )


def _validate_where(kwargs: dict[str, Any]) -> None:
    mask = kwargs.get('where', True)
    if (
        type(mask) is not bool
        and not (type(mask) is np.ndarray and mask.dtype == np.dtype(bool))
        and type(mask) is not np.bool_
    ):
        raise TypeError('where must be a unitless boolean mask.')


def dispatch_ufunc(ufunc: Any, method: str, inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if ufunc in (np.modf, np.frexp) and method == '__call__':
        return _dispatch_multi_output(ufunc, inputs, kwargs)
    outputs = kwargs.get('out', ())
    if any(x is not None for x in outputs):
        if (
            method not in ('__call__', 'reduce', 'accumulate')
            or ufunc.nout != 1
            or len(outputs) != 1
        ):
            return NotImplemented
        target = outputs[0]
        if ufunc in _COMPARISON_UFUNCS.values() or ufunc in (
            np.isfinite,
            np.isnan,
            np.isinf,
            np.signbit,
        ):
            if type(target) is not np.ndarray or target.dtype.kind != 'b':
                return NotImplemented
        elif not isinstance(target, Quantity):
            return NotImplemented
        options = {k: v for k, v in kwargs.items() if k != 'out'}
        result = _dispatch_ufunc(ufunc, method, inputs, options)
        if result is NotImplemented:
            return NotImplemented
        return _write_output(
            result,
            target,
            where=options.get('where', True) if method == '__call__' else True,
            casting=options.get('casting', 'same_kind'),
            broadcast=method == '__call__',
        )
    return _dispatch_ufunc(ufunc, method, inputs, kwargs)


def _dispatch_multi_output(ufunc: Any, inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    outputs = kwargs.get('out', (None, None))
    if len(outputs) != 2:
        return NotImplemented
    for index, target in enumerate(outputs):
        if target is None:
            continue
        if ufunc is np.frexp and index == 1:
            if type(target) is not np.ndarray or target.dtype.kind not in 'iu':
                return NotImplemented
        elif not isinstance(target, Quantity):
            return NotImplemented
    options = {k: v for k, v in kwargs.items() if k != 'out'}
    result = _dispatch_ufunc(ufunc, '__call__', inputs, options)
    if result is NotImplemented:
        return NotImplemented
    buffers = [x.value if isinstance(x, Quantity) else x for x in outputs if x is not None]
    if any(type(buffer) is not np.ndarray for buffer in buffers):
        raise TypeError('Quantity out requires a plain numeric ndarray buffer.')
    if len(buffers) == 2 and np.shares_memory(*buffers):
        raise ValueError('Multiple output buffers must not overlap.')
    # All outputs share the ufunc broadcast shape, including newly allocated
    # outputs. Do not broadcast one provided output into another's larger shape.
    shape = np.broadcast_shapes(np.shape(result[0].value), *(x.shape for x in buffers))
    if any(buffer.shape != shape for buffer in buffers):
        raise ValueError('All output buffers must have the ufunc broadcast shape.')
    where = options.get('where', True)
    if type(where) is np.ndarray:
        # A mask can alias either output; retain it across both final writes.
        where = where.copy()
    returned = []
    staged = []
    for value, target in zip(result, outputs):
        raw = value.value if isinstance(value, Quantity) else value
        if np.shape(raw) != shape:
            raw = np.broadcast_to(raw, shape).copy()
            value = Quantity(raw, value.unit) if isinstance(value, Quantity) else raw
        if target is not None:
            staged.append(
                _stage_output(
                    value, target, where=where, casting=options.get('casting', 'same_kind')
                )
            )
        returned.append(value if target is None else target)
    # No caller storage changes until every output has converted and cast.
    for buffer, temporary, mask in staged:
        np.copyto(buffer, temporary, where=mask, casting='no')
    return tuple(returned)


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


def _stage_output(
    result: Any,
    target: Any,
    *,
    where: Any = True,
    casting: Any = 'same_kind',
    broadcast: bool = False,
) -> tuple[Any, Any, Any]:
    if isinstance(result, Quantity):
        if not isinstance(target, Quantity) or type(target.value) is not np.ndarray:
            raise TypeError('Quantity out requires a plain numeric ndarray buffer.')
        if result.dimension != target.dimension:
            raise DimensionError('out has an incompatible dimension.')
        factor = result.unit.factor / target.unit.factor
        buffer = target.value
    else:
        kinds = 'b' if np.asarray(result).dtype.kind == 'b' else 'iu'
        if type(target) is not np.ndarray or target.dtype.kind not in kinds:
            raise TypeError('Unitless out requires a plain ndarray with the matching result kind.')
        factor = ONE
        buffer = target
    if not buffer.flags.writeable:
        raise ValueError('out buffer is read-only.')
    value = result.value if isinstance(result, Quantity) else result
    if not broadcast and np.shape(value) != buffer.shape:
        raise ValueError('out must have exactly the result shape.')
    if isinstance(result, Quantity):
        value = scale_magnitude(value, factor, where=where)
    # Validate broadcasting and casting before touching shared storage. Casting
    # also runs in temporary storage so e.g. float narrowing errors cannot
    # partially overwrite the caller's buffer.
    value = np.broadcast_to(value, buffer.shape)
    mask = np.broadcast_to(where, buffer.shape)
    if np.may_share_memory(mask, buffer):
        mask = mask.copy()
    staged = np.empty(buffer.shape, dtype=buffer.dtype)
    np.copyto(staged, value, where=mask, casting=casting)
    return buffer, staged, mask


def _dispatch_ufunc(
    ufunc: Any, method: str, inputs: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    if not all(isinstance(x, Unit) or _is_supported_operand(x) for x in inputs):
        return NotImplemented
    if any(x is not None for x in kwargs.get('out', ())):
        return NotImplemented
    kwargs = {k: v for k, v in kwargs.items() if k != 'out'}
    if method == 'reduce' and ufunc in (np.add, np.multiply):
        if not isinstance(inputs[0], Quantity) or kwargs.keys() - {
            'axis',
            'dtype',
            'keepdims',
            'initial',
            'where',
        }:
            return NotImplemented
        handler = _sum if ufunc is np.add else _prod
        return handler(inputs[0], **{'axis': 0, **kwargs})
    if method == 'accumulate' and ufunc is np.add:
        if kwargs.keys() - {'axis', 'dtype'}:
            return NotImplemented
        if kwargs.get('dtype') is not None and np.dtype(kwargs['dtype']).kind not in 'iufc':
            return NotImplemented
        a, unit = _operand_parts(inputs[0])
        return Quantity(getattr(ufunc, method)(a, **kwargs), unit)
    if method != '__call__':
        return NotImplemented
    if kwargs.keys() - {'where', 'dtype', 'casting', 'order', 'subok'}:
        return NotImplemented
    _validate_where(kwargs)
    # Explicit dtype=object/str/etc must not escape the supported numerical domain.
    if 'dtype' in kwargs and np.dtype(kwargs['dtype']).kind not in 'biufc':
        return NotImplemented
    if 'where' in kwargs:
        # A masked temporary intentionally has unspecified unused elements.
        kwargs['out'] = (None,) * ufunc.nout
    if ufunc in (np.add, np.subtract):
        if any(isinstance(x, Unit) for x in inputs):
            return NotImplemented
        a, b, unit = _align_operands(*inputs)
        return Quantity(ufunc(a, b, **kwargs), unit)
    if ufunc in _COMPARISON_UFUNCS.values():
        a, u = _operand_parts(inputs[0])
        b, v = _operand_parts(inputs[1])
        if u.dimension != v.dimension:
            if ufunc in (np.equal, np.not_equal):
                # Broadcast even when dimensions alone determine the answer.
                shape = np.broadcast_shapes(np.shape(a), np.shape(b))
                result = np.full(shape, ufunc is np.not_equal, dtype=bool)
                return np.equal(result, True, **kwargs)
            raise DimensionError('Cannot compare quantities with different dimensions.')
        # A fixed canonical domain makes mixed scalar/array comparison symmetric.
        ratio = u.factor / v.factor
        if ratio == ONE:
            return ufunc(a, b, **kwargs)
        return ufunc(scale_magnitude(a, u.factor), scale_magnitude(b, v.factor), **kwargs)
    if ufunc in (np.multiply, np.divide, np.true_divide, np.matmul):
        a, u = _operand_parts(inputs[0])
        b, v = _operand_parts(inputs[1])
        # Unit multiplication constructs a Quantity without copying the array.
        if ufunc is np.multiply and not kwargs:
            if isinstance(inputs[0], Unit):
                return Quantity(b, u * v)
            if isinstance(inputs[1], Unit):
                return Quantity(a, u * v)
        if ufunc in (np.divide, np.true_divide) and isinstance(inputs[1], Unit) and not kwargs:
            return Quantity(a, u / v)
        unit = u / v if ufunc in (np.divide, np.true_divide) else u * v
        return Quantity(ufunc(a, b, **kwargs), unit)
    a, unit = _operand_parts(inputs[0])
    if ufunc in (np.modf, np.frexp):
        first, second = ufunc(a, **kwargs)
        # Decompose the stored coordinates. Both modf parts, and the frexp
        # mantissa, carry the input unit; the binary exponent is a bare integer.
        return Quantity(first, unit), Quantity(second, unit) if ufunc is np.modf else second
    if ufunc in (np.positive, np.negative, np.absolute, np.fabs, np.conjugate):
        return Quantity(ufunc(a, **kwargs), unit)
    if ufunc in (np.isfinite, np.isnan, np.isinf, np.signbit):
        return ufunc(a, **kwargs)
    if ufunc in (np.square, np.sqrt, np.cbrt, np.reciprocal):
        power = {np.square: 2, np.sqrt: Fraction(1, 2), np.cbrt: Fraction(1, 3), np.reciprocal: -1}[
            ufunc
        ]
        return Quantity(ufunc(a, **kwargs), unit**power)
    if ufunc in (np.power, np.float_power):
        exponent, exponent_unit = _operand_parts(inputs[1])
        if not exponent_unit.is_dimensionless():
            raise DimensionError('Exponents must be dimensionless.')
        exponent = scale_magnitude(exponent, exponent_unit.factor)
        if np.iscomplexobj(exponent) or not np.all(np.isfinite(exponent)):
            raise TypeError('Exponents must be finite real values.')
        if unit.is_dimensionless():
            return Quantity(ufunc(scale_magnitude(a, unit.factor), exponent, **kwargs), Unit())
        if np.ndim(exponent) != 0:
            raise TypeError('Dimensional powers require one scalar exponent.')
        scalar = exponent.item() if isinstance(exponent, (np.ndarray, np.generic)) else exponent
        raw_exponent = inputs[1].value if isinstance(inputs[1], Quantity) else inputs[1]
        rational, irrational = exponent_unit.factor.decompose()
        if isinstance(raw_exponent, (Fraction, Decimal)) and irrational == ONE:
            power = Fraction(raw_exponent) * rational
        else:
            power = as_exponent(scalar) if type(scalar) in (int, float) else Fraction(str(scalar))
        return Quantity(ufunc(a, exponent, **kwargs), unit**power)
    if ufunc in (np.sin, np.cos, np.tan, np.exp, np.expm1, np.log, np.log2, np.log10, np.log1p):
        if not unit.is_dimensionless():
            raise DimensionError('This function requires dimensionless input.')
        return Quantity(ufunc(scale_magnitude(a, unit.factor), **kwargs), Unit())
    return NotImplemented


def apply_ufunc(name: str, *inputs: Any) -> Any:
    """Enter NumPy dispatch, including other operands' protocol overrides."""
    return getattr(np, name)(*inputs)


def compare(left: Any, right: Any, op: Any) -> Any:
    """Map Python comparisons at the optional NumPy boundary."""
    return _COMPARISON_UFUNCS[op](left, right)


def _convert_tolerance(value: Any, unit: Unit, name: str) -> Any:
    raw, source = _operand_parts(value)
    if not _is_supported_operand(value):
        raise TypeError(f'{name} must be a real numeric scalar or array.')
    if np.iscomplexobj(raw) or not np.all(np.isfinite(raw)) or np.any(np.less(raw, 0)):
        raise ValueError(f'{name} must be finite and nonnegative.')
    # Bare zero absolute tolerance is meaningful in any dimension.
    if source.dimension != unit.dimension:
        if name == 'abs_tol' and not isinstance(value, Quantity) and np.all(np.equal(raw, 0)):
            return raw
        raise DimensionError(f'{name} has an incompatible dimension.')
    result = scale_magnitude(raw, source.factor / unit.factor)
    if not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be finite after unit conversion.')
    return result


def isclose(
    left: Any, right: Any, rel_tol: Any, abs_tol: Any, *, symmetric: bool, equal_nan: bool = False
) -> Any:
    if not _is_supported_operand(left) or not _is_supported_operand(right):
        raise TypeError('isclose requires supported numeric magnitudes.')
    if symmetric:
        a, u = _operand_parts(left)
        b, v = _operand_parts(right)
        if u.dimension != v.dimension:
            raise DimensionError('Cannot compare quantities with different dimensions.')
        # Evaluate the symmetric formula in the same coherent coordinates in
        # both operand orders, including its absolute tolerance.
        unit = u / u.factor
        a, b = scale_magnitude(a, u.factor), scale_magnitude(b, v.factor)
    else:
        a, b, unit = _align_operands(left, right)
    relative = _convert_tolerance(rel_tol, Unit(), 'rel_tol')
    absolute = _convert_tolerance(abs_tol, unit, 'abs_tol')
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        raise TypeError('isclose requires real magnitudes.')
    # Enter a floating domain before subtraction/abs (unsigned and signed ints
    # could otherwise wrap). Preserve longdouble when present.
    dtype = np.result_type(a, b, np.float64)
    a, b = np.asarray(a, dtype=dtype), np.asarray(b, dtype=dtype)
    with np.errstate(invalid='ignore', over='ignore'):
        reference = np.maximum(np.abs(a), np.abs(b)) if symmetric else np.abs(b)
        limit = (
            np.maximum(relative * reference, absolute)
            if symmetric
            else absolute + relative * reference
        )
        result = (a == b) | (np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) <= limit))
    if equal_nan:
        result = result | (np.isnan(a) & np.isnan(b))
    return result


def dispatch_function(
    func: Any, types: tuple[type, ...], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    if any(t not in (Quantity, np.ndarray) for t in types):
        return NotImplemented
    if func in (np.isclose, np.allclose):
        return _close_comparison(func, *args, **kwargs)
    if func in (np.concatenate, np.stack):
        return _join_arrays(func, *args, **kwargs)
    if func is np.trapezoid:
        return _trapezoid(*args, **kwargs)
    if not args or not isinstance(args[0], Quantity) or not _is_supported_operand(args[0]):
        return NotImplemented
    handler = _QUANTITY_FUNCTIONS.get(func)
    if handler is None:
        return NotImplemented
    return handler(*args, **kwargs)


def _close_comparison(func: Any, *args: Any, **kwargs: Any) -> Any:
    if not 2 <= len(args) <= 5 or kwargs.keys() - {'rtol', 'atol', 'equal_nan'}:
        return NotImplemented
    options: dict[str, Any] = dict(zip(('rtol', 'atol', 'equal_nan'), args[2:]))
    if options.keys() & kwargs.keys():
        raise TypeError('Multiple values for tolerance argument.')
    options.update(kwargs)
    if not all(_is_supported_operand(x) for x in args[:2]):
        return NotImplemented
    result = isclose(
        args[0],
        args[1],
        options.get('rtol', 1e-5),
        options.get('atol', 1e-8),
        symmetric=False,
        equal_nan=options.get('equal_nan', False),
    )
    return bool(np.all(result)) if func is np.allclose else result


def _join_arrays(
    func: Any,
    arrays: Any,
    axis: Any = 0,
    out: Any = None,
    *,
    dtype: Any = None,
    casting: str = 'same_kind',
) -> Any:
    if out is not None and dtype is not None:
        raise TypeError('Cannot provide both dtype and out for concatenate/stack.')
    arrays = tuple(arrays)
    if not arrays or not all(_is_supported_operand(x) for x in arrays):
        return NotImplemented
    first = next((x for x in arrays if isinstance(x, Quantity)), None)
    if first is None:
        return NotImplemented
    values = []
    for x in arrays:
        raw, unit = _operand_parts(x)
        if unit.dimension != first.dimension:
            raise DimensionError('All concatenated inputs must have compatible dimensions.')
        values.append(scale_magnitude(raw, unit.factor / first.unit.factor))
    if dtype is not None and np.dtype(dtype).kind not in 'iufc':
        raise TypeError('Only numeric output dtypes are supported.')
    result = Quantity(func(values, axis=axis, dtype=dtype, casting=casting), first.unit)
    return _write_output(result, out, casting=casting)


def _reshape(
    q: Any, shape: Any, order: Literal['C', 'F', 'A'] = 'C', *, copy: bool | None = None
) -> Any:
    return Quantity(np.reshape(q.value, shape, order=order, copy=copy), q.unit)


def _transpose(q: Any, axes: Any = None) -> Any:
    return Quantity(np.transpose(q.value, axes), q.unit)


def _trapezoid(y: Any, x: Any = None, dx: Any = 1.0, axis: int = -1) -> Any:
    if not _is_supported_operand(y):
        return NotImplemented
    value, unit = _operand_parts(y)
    if x is None:
        if not _is_supported_operand(dx):
            return NotImplemented
        spacing, coordinate_unit = _operand_parts(dx)
        if np.ndim(spacing) != 0:
            raise TypeError('dx must be a numeric scalar or scalar Quantity.')
        result = np.trapezoid(value, dx=spacing, axis=axis)
    else:
        if not _is_supported_operand(x):
            return NotImplemented
        coordinates, coordinate_unit = _operand_parts(x)
        result = np.trapezoid(value, x=coordinates, axis=axis)
    return Quantity(result, unit * coordinate_unit)


def _argmin_max(
    func: Any, q: Any, axis: Any = None, out: Any = None, *, keepdims: bool = False
) -> Any:
    return _write_output(func(q.value, axis=axis, keepdims=keepdims), out)


def _median(
    q: Any, axis: Any = None, out: Any = None, overwrite_input: bool = False, keepdims: bool = False
) -> Any:
    if overwrite_input:
        raise TypeError('overwrite_input is not supported for Quantity.')
    return _write_output(Quantity(np.median(q.value, axis=axis, keepdims=keepdims), q.unit), out)


def _cumsum(q: Any, axis: Any = None, dtype: Any = None, out: Any = None) -> Any:
    _validate_dtype(dtype)
    return _write_output(Quantity(np.cumsum(q.value, axis=axis, dtype=dtype), q.unit), out)


def _mean(
    q: Any,
    axis: Any = None,
    dtype: Any = None,
    out: Any = None,
    keepdims: bool = False,
    *,
    where: Any = True,
) -> Any:
    _validate_dtype(dtype)
    _validate_where({'where': where})
    result = Quantity(
        np.mean(q.value, axis=axis, dtype=dtype, keepdims=keepdims, where=where), q.unit
    )
    return _write_output(result, out)


def _sum(
    q: Any,
    axis: Any = None,
    dtype: Any = None,
    out: Any = None,
    keepdims: bool = False,
    initial: Any = 0,
    where: Any = True,
) -> Any:
    _validate_dtype(dtype)
    _validate_where({'where': where})
    value, unit = _operand_parts(q)
    initial_value = None
    if initial is not None:
        if not _is_supported_operand(initial):
            raise TypeError('initial must be a numeric scalar or scalar Quantity.')
        initial_value, initial_unit = _operand_parts(initial)
        if np.ndim(initial_value) != 0:
            raise TypeError('initial must be a numeric scalar or scalar Quantity.')
        if initial_unit.dimension != unit.dimension:
            # Bare zero is the additive identity in any dimension. A Quantity
            # zero still has its declared dimension and must be compatible.
            if isinstance(initial, Quantity) or initial != 0:
                raise DimensionError('initial has an incompatible dimension.')
        else:
            initial_value = scale_magnitude(initial_value, initial_unit.factor / unit.factor)
    result = Quantity(
        np.add.reduce(
            value, axis=axis, dtype=dtype, keepdims=keepdims, initial=initial_value, where=where
        ),
        unit,
    )
    return _write_output(result, out)


def _prod(
    q: Any,
    axis: Any = None,
    dtype: Any = None,
    out: Any = None,
    keepdims: bool = False,
    *,
    initial: Any = 1,
    where: Any = True,
) -> Any:
    _validate_dtype(dtype)
    _validate_where({'where': where})
    initial_value, initial_unit = None, Unit()
    if initial is not None:
        if not _is_supported_operand(initial):
            raise TypeError('initial must be a numeric scalar or scalar Quantity.')
        initial_value, initial_unit = _operand_parts(initial)
        if np.ndim(initial_value) != 0:
            raise TypeError('initial must be a numeric scalar or scalar Quantity.')

    value, unit = _operand_parts(q)
    mask = np.broadcast_to(where, np.shape(value))
    # Different counts still have one coherent unit for dimensionless inputs.
    if unit.is_dimensionless():
        value = scale_magnitude(value, unit.factor, where=mask)
        unit = Unit()
    result = np.multiply.reduce(
        value, axis=axis, dtype=dtype, keepdims=keepdims, initial=initial_value, where=where
    )
    if not unit.is_dimensionless():
        shape = np.shape(value)
        axes = range(len(shape)) if axis is None else axis if isinstance(axis, tuple) else (axis,)
        # NumPy has already validated axes, including its scalar 0/-1 special case.
        count = math.prod(shape[i] for i in axes) if shape else 1
        if np.ndim(where) == 0:
            count = count if bool(where) else 0
        else:
            counts = np.sum(mask, axis=axis)
            if np.size(counts):
                count = int(np.asarray(counts).flat[0])
                if np.any(counts != count):
                    raise DimensionError(
                        'Product reductions require equal element counts per result.'
                    )
            # With no output elements, retain the nominal reduction extent.
        unit = unit**count
    return _write_output(Quantity(result, unit * initial_unit), out)


def _std_var(
    func: Any,
    q: Any,
    axis: Any = None,
    dtype: Any = None,
    out: Any = None,
    ddof: Any = 0,
    keepdims: bool = False,
    *,
    where: Any = True,
) -> Any:
    _validate_dtype(dtype)
    _validate_where({'where': where})
    unit = q.unit**2 if func is np.var else q.unit
    result = Quantity(
        func(q.value, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, where=where), unit
    )
    return _write_output(result, out)


def _validate_dtype(dtype: Any) -> None:
    if dtype is not None and np.dtype(dtype).kind not in 'iufc':
        raise TypeError('Only numeric output dtypes are supported.')


def power(q: Any, exponent: Any) -> Any:
    if isinstance(exponent, (Fraction, Decimal)):
        exact = as_exponent(exponent)
        unit = Unit() if q.is_dimensionless() else q.unit**exact
        value = scale_magnitude(q.value, q.unit.factor) if q.is_dimensionless() else q.value
        return Quantity(np.power(value, float(exponent)), unit)
    return apply_ufunc('power', q, exponent)


# These handlers share the validated Quantity-first input contract. Sequence
# and mixed-input functions retain their separate dispatch paths above.
_QUANTITY_FUNCTIONS = {
    np.reshape: _reshape,
    np.transpose: _transpose,
    np.argmin: partial(_argmin_max, np.argmin),
    np.argmax: partial(_argmin_max, np.argmax),
    np.median: _median,
    np.cumsum: _cumsum,
    np.prod: _prod,
    np.std: partial(_std_var, np.std),
    np.var: partial(_std_var, np.var),
    np.sum: _sum,
    np.mean: _mean,
}
