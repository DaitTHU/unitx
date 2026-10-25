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
from typing import Any, Literal, cast

import numpy as np  # pyright: ignore[reportMissingImports]

from .._coordinates.resolve import _Coordinate
from .._coordinates.resolve import convert_point as convert
from .._operands import _Point, linear_unit
from ..affinequantity import AffineQuantity
from ..dimension import DIMENSIONLESS
from ..exceptions import DimensionError
from ..factor import Factor, one
from ..logquantity import LogQuantity
from ..quantity import Quantity
from ..unit import Unit
from . import as_exponent
from .conversion import compatible

_UNSET = object()

_COMPARISON_UFUNCS = {
    operator.eq: np.equal,
    operator.ne: np.not_equal,
    operator.lt: np.less,
    operator.le: np.less_equal,
    operator.gt: np.greater,
    operator.ge: np.greater_equal,
}


def _wrap(value: Any, unit: Unit | _Coordinate) -> Any:
    return unit.__rmatmul__(value) if isinstance(unit, _Coordinate) else Quantity(value, unit)


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
    if isinstance(value, (Quantity, _Point)):
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
    if factor == one:
        return value
    rational, irrational = factor.decompose()
    multiplier = (
        (rational.numerator if rational.denominator == 1 else float(rational))
        if irrational == one
        else float(factor.value)
    )
    # Do not pass a Fraction to NumPy (it would create object dtype).
    return np.multiply(value, multiplier, where=where, out=None)


def _operand_parts(value: Any) -> tuple[Any, Unit | _Coordinate]:
    if isinstance(value, (Quantity, _Point)):
        return _numeric_value(value.value), value.unit
    if isinstance(value, (Unit, _Coordinate)):
        return 1, value
    return _numeric_value(value), Unit()


def _linear_parts(value: Any) -> tuple[Any, Unit]:
    raw, unit = _operand_parts(value)
    if not isinstance(unit, Unit):
        raise TypeError('This operation requires linear quantities or differences.')
    return raw, unit


def _align_operands(left: Any, right: Any) -> tuple[Any, Any, Unit]:
    a, u = _linear_parts(left)
    b, v = _linear_parts(right)
    if isinstance(u, _Coordinate) or isinstance(v, _Coordinate):
        raise TypeError('Use point/difference arithmetic for affine coordinates.')
    if not compatible(u, v):
        raise DimensionError('Cannot operate on incompatible dimensions or logarithmic degrees.')
    # Addition retains the first Quantity's unit, even with a bare left input.
    target = u if isinstance(left, (Quantity, _Point)) else v
    return (
        scale_magnitude(a, u.factor / target.factor),
        scale_magnitude(b, v.factor / target.factor),
        target,
    )


def _selection_parts(left: Any, right: Any) -> tuple[Any, Any, Unit | _Coordinate]:
    """Align selection operands without crossing point/difference boundaries."""
    if not all(_is_supported_operand(x) for x in (left, right)):
        error_message = 'Selection requires numeric scalars, ndarrays, quantities or points.'
        raise TypeError(error_message)
    if isinstance(left, _Point) or isinstance(right, _Point):
        if not isinstance(left, _Point) or not isinstance(right, _Point):
            error_message = 'Selection cannot mix points and differences.'
            raise TypeError(error_message)
        if not left.unit._compatible_with(right.unit):
            error_message = 'Selection requires compatible points with the same convention.'
            raise DimensionError(error_message)
        return (
            _numeric_value(left.value),
            convert(_numeric_value(right.value), right.unit, left.unit),
            left.unit,
        )
    return _align_operands(left, right)


def _validate_where(kwargs: dict[str, Any]) -> None:
    mask = kwargs.get('where', True)
    if (
        type(mask) is not bool
        and not (type(mask) is np.ndarray and mask.dtype == np.dtype(bool))
        and type(mask) is not np.bool_
    ):
        raise TypeError('where must be a unitless boolean mask.')


def _snapshot_index(key: Any) -> Any:
    """Keep index arrays independent from storage modified by assignment."""
    if isinstance(key, np.ndarray):
        if type(key) is not np.ndarray or key.dtype.kind not in 'biu':
            raise TypeError('Array indices must be plain boolean or integer ndarrays.')
        return key.copy()
    if isinstance(key, (tuple, list)):
        return type(key)(_snapshot_index(item) for item in key)
    if isinstance(key, slice):
        return slice(
            *(
                None if item is None else operator.index(item)
                for item in (key.start, key.stop, key.step)
            )
        )
    if key is None or key is Ellipsis or isinstance(key, (bool, np.bool_)):
        return key
    return operator.index(key)


def assign_item(target: Quantity[Any] | _Point[Any], key: Any, value: Any) -> None:
    """Convert and stage selected values before modifying caller storage."""
    buffer = target.value
    if type(buffer) is not np.ndarray:
        raise TypeError('Indexed assignment requires a plain numeric ndarray magnitude.')
    if not buffer.flags.writeable:
        raise ValueError('Assignment destination is read-only.')
    if not _is_supported_operand(value):
        raise TypeError('Assignment requires a numeric scalar, ndarray, or Quantity.')
    raw, unit = _operand_parts(value)
    if unit.dimension != target.dimension:
        raise DimensionError('Assigned value has an incompatible dimension.')
    key = _snapshot_index(key)
    if isinstance(unit, _Coordinate) != isinstance(target.unit, _Coordinate):
        raise DimensionError(
            'Assignment requires matching point/difference semantics; convert explicitly.'
        )
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
        elif not isinstance(target, (Quantity, _Point)):
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
        elif not isinstance(target, (Quantity, _Point)):
            return NotImplemented
    options = {k: v for k, v in kwargs.items() if k != 'out'}
    result = _dispatch_ufunc(ufunc, '__call__', inputs, options)
    if result is NotImplemented:
        return NotImplemented
    buffers = [
        x.value if isinstance(x, (Quantity, _Point)) else x for x in outputs if x is not None
    ]
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
        raw = value.value if isinstance(value, (Quantity, _Point)) else value
        if np.shape(raw) != shape:
            raw = np.broadcast_to(raw, shape).copy()
            value = _wrap(raw, value.unit) if isinstance(value, (Quantity, _Point)) else raw
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
    if isinstance(result, (Quantity, _Point)):
        if not isinstance(target, (Quantity, _Point)) or type(target.value) is not np.ndarray:
            raise TypeError('Quantity out requires a plain numeric ndarray buffer.')
        if result.dimension != target.dimension:
            raise DimensionError('out has an incompatible dimension.')
        if isinstance(result.unit, _Coordinate) != isinstance(target.unit, _Coordinate):
            raise DimensionError('out must preserve point/difference semantics.')
        if (
            not isinstance(result.unit, _Coordinate)
            and not isinstance(target.unit, _Coordinate)
            and not compatible(result.unit, target.unit)
        ):
            raise DimensionError('out has an incompatible logarithmic degree.')
        if isinstance(result.unit, _Coordinate):
            factor = one
        else:
            assert isinstance(target.unit, Unit)
            factor = result.unit.factor / target.unit.factor
        buffer = target.value
    else:
        kinds = 'b' if np.asarray(result).dtype.kind == 'b' else 'iu'
        if type(target) is not np.ndarray or target.dtype.kind not in kinds:
            raise TypeError('Unitless out requires a plain ndarray with the matching result kind.')
        factor = one
        buffer = target
    if not buffer.flags.writeable:
        raise ValueError('out buffer is read-only.')
    value = result.value if isinstance(result, (Quantity, _Point)) else result
    if not broadcast and np.shape(value) != buffer.shape:
        raise ValueError('out must have exactly the result shape.')
    if isinstance(result, (Quantity, _Point)):
        if isinstance(result.unit, _Coordinate):
            if where is not True:
                value = np.where(where, value, 0)
            value = convert(value, result.unit, cast(Any, target).unit)
        else:
            value = scale_magnitude(value, factor, where=where)
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


def _dispatch_ufunc(
    ufunc: Any, method: str, inputs: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    if not all(isinstance(x, (Unit, _Coordinate)) or _is_supported_operand(x) for x in inputs):
        return NotImplemented
    if any(x is not None for x in kwargs.get('out', ())):
        return NotImplemented
    kwargs = {k: v for k, v in kwargs.items() if k != 'out'}
    if ufunc is np.matmul and method == '__call__' and isinstance(inputs[1], (Unit, _Coordinate)):
        if isinstance(inputs[0], (Quantity, _Point, Unit, _Coordinate)):
            raise TypeError('Only a bare numeric magnitude can construct a point with @.')
        if kwargs:
            return NotImplemented
        return (
            inputs[1].__rmatmul__(inputs[0])
            if isinstance(inputs[1], _Coordinate)
            else AffineQuantity(inputs[0], inputs[1])
        )
    inputs = tuple(linear_unit(x) if isinstance(x, _Coordinate) else x for x in inputs)
    if any(isinstance(x, _Point) for x in inputs):
        return _affine_ufunc(ufunc, method, inputs, kwargs)
    if method == 'reduce' and ufunc in (np.add, np.multiply):
        if not isinstance(inputs[0], (Quantity, _Point)) or kwargs.keys() - {
            'axis',
            'dtype',
            'keepdims',
            'initial',
            'where',
        }:
            return NotImplemented
        handler = _sum if ufunc is np.add else _prod
        return handler(ufunc.reduce, inputs[0], **{'axis': 0, **kwargs})
    if method == 'accumulate' and ufunc is np.add:
        if kwargs.keys() - {'axis', 'dtype'}:
            return NotImplemented
        if kwargs.get('dtype') is not None and np.dtype(kwargs['dtype']).kind not in 'iufc':
            return NotImplemented
        a, unit = _linear_parts(inputs[0])
        return _wrap(getattr(ufunc, method)(a, **kwargs), unit)
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
        return _wrap(ufunc(a, b, **kwargs), unit)
    if ufunc in (np.minimum, np.maximum):
        a, b, unit = _selection_parts(*inputs)
        return _wrap(ufunc(a, b, **kwargs), unit)
    if ufunc in (np.hypot, np.arctan2):
        a, b, unit = _align_operands(*inputs)
        if unit.dimension != unit.dimension.physical:
            raise DimensionError('Convert logarithmic quantities to physical ratios first.')
        return _wrap(ufunc(a, b, **kwargs), Unit('rad') if ufunc is np.arctan2 else unit)
    if ufunc in _COMPARISON_UFUNCS.values():
        a, u = _linear_parts(inputs[0])
        b, v = _linear_parts(inputs[1])
        if not compatible(u, v):
            if ufunc in (np.equal, np.not_equal):
                # Broadcast even when dimensions alone determine the answer.
                shape = np.broadcast_shapes(np.shape(a), np.shape(b))
                result = np.full(shape, ufunc is np.not_equal, dtype=bool)
                return np.equal(result, True, **kwargs)
            raise DimensionError('Cannot compare quantities with different dimensions.')
        # A fixed canonical domain makes mixed scalar/array comparison symmetric.
        ratio = u.factor / v.factor
        if ratio == one:
            return ufunc(a, b, **kwargs)
        return ufunc(scale_magnitude(a, u.factor), scale_magnitude(b, v.factor), **kwargs)
    if ufunc in (np.multiply, np.divide, np.true_divide, np.matmul):
        a, u = _linear_parts(inputs[0])
        b, v = _linear_parts(inputs[1])
        assert isinstance(u, Unit) and isinstance(v, Unit)
        # Unit multiplication constructs a Quantity without copying the array.
        if ufunc is np.multiply and not kwargs:
            if isinstance(inputs[0], Unit):
                return _wrap(b, u * v)
            if isinstance(inputs[1], Unit):
                return _wrap(a, u * v)
        if ufunc in (np.divide, np.true_divide) and isinstance(inputs[1], Unit) and not kwargs:
            return _wrap(a, u / v)
        unit = u / v if ufunc in (np.divide, np.true_divide) else u * v
        return _wrap(ufunc(a, b, **kwargs), unit)
    a, unit = _linear_parts(inputs[0])
    if ufunc in (np.modf, np.frexp):
        first, second = ufunc(a, **kwargs)
        # Decompose the stored coordinates. Both modf parts, and the frexp
        # mantissa, carry the input unit; the binary exponent is a bare integer.
        return _wrap(first, unit), _wrap(second, unit) if ufunc is np.modf else second
    if ufunc in (np.positive, np.negative, np.absolute, np.fabs, np.conjugate):
        return _wrap(ufunc(a, **kwargs), unit)
    if ufunc in (np.isfinite, np.isnan, np.isinf, np.signbit):
        return ufunc(a, **kwargs)
    if ufunc in (np.square, np.sqrt, np.cbrt, np.reciprocal):
        power = {np.square: 2, np.sqrt: Fraction(1, 2), np.cbrt: Fraction(1, 3), np.reciprocal: -1}[
            ufunc
        ]
        return _wrap(ufunc(a, **kwargs), unit**power)
    if ufunc in (np.power, np.float_power):
        exponent, exponent_unit = _linear_parts(inputs[1])
        if exponent_unit.dimension != DIMENSIONLESS:
            raise DimensionError('Exponents must be dimensionless.')
        exponent = scale_magnitude(exponent, exponent_unit.factor)
        if np.iscomplexobj(exponent) or not np.all(np.isfinite(exponent)):
            raise TypeError('Exponents must be finite real values.')
        if unit.dimension == DIMENSIONLESS:
            return _wrap(ufunc(scale_magnitude(a, unit.factor), exponent, **kwargs), Unit())
        if np.ndim(exponent) != 0:
            raise TypeError('Dimensional powers require one scalar exponent.')
        scalar = exponent.item() if isinstance(exponent, (np.ndarray, np.generic)) else exponent
        raw_exponent = inputs[1].value if isinstance(inputs[1], (Quantity, _Point)) else inputs[1]
        rational, irrational = exponent_unit.factor.decompose()
        if isinstance(raw_exponent, (Fraction, Decimal)) and irrational == one:
            power = Fraction(raw_exponent) * rational
        else:
            power = as_exponent(scalar) if type(scalar) in (int, float) else Fraction(str(scalar))
        return _wrap(ufunc(a, exponent, **kwargs), unit**power)
    if ufunc in (np.arcsin, np.arccos, np.arctan):
        if unit.dimension != DIMENSIONLESS:
            raise DimensionError('This function requires dimensionless input.')
        return _wrap(ufunc(scale_magnitude(a, unit.factor), **kwargs), Unit('rad'))
    if ufunc in (np.sin, np.cos, np.tan, np.exp, np.expm1, np.log, np.log2, np.log10, np.log1p):
        if unit.dimension != DIMENSIONLESS:
            raise DimensionError('This function requires dimensionless input.')
        return _wrap(ufunc(scale_magnitude(a, unit.factor), **kwargs), Unit())
    return NotImplemented


def apply_ufunc(name: str, *inputs: Any) -> Any:
    """Enter NumPy dispatch, including other operands' protocol overrides."""
    return getattr(np, name)(*inputs)


def compare(left: Any, right: Any, op: Any) -> Any:
    """Map Python comparisons at the optional NumPy boundary."""
    return _COMPARISON_UFUNCS[op](left, right)


def _convert_tolerance(value: Any, unit: Unit, name: str) -> Any:
    raw, source = _linear_parts(value)
    if not _is_supported_operand(value):
        raise TypeError(f'{name} must be a real numeric scalar or array.')
    if np.iscomplexobj(raw) or not np.all(np.isfinite(raw)) or np.any(np.less(raw, 0)):
        raise ValueError(f'{name} must be finite and nonnegative.')
    # Bare zero absolute tolerance is meaningful in any dimension.
    if (
        isinstance(source, _Coordinate)
        or isinstance(unit, _Coordinate)
        or not compatible(source, unit)
    ):
        if (
            name == 'abs_tol'
            and not isinstance(value, (Quantity, _Point))
            and np.all(np.equal(raw, 0))
        ):
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
    if any(
        isinstance(x, (Quantity, _Point)) and isinstance(x.unit, _Coordinate) for x in (left, right)
    ):
        from .._coordinates.points import _isclose

        return _isclose(left, right, rel_tol, abs_tol, equal_nan=equal_nan)
    if symmetric:
        a, u = _linear_parts(left)
        b, v = _linear_parts(right)
        if not compatible(u, v):
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
    if any(t not in (Quantity, AffineQuantity, LogQuantity, np.ndarray) for t in types):
        return NotImplemented
    if func is np.where:
        return _where(*args, **kwargs)
    if func is np.clip:
        return _clip(*args, **kwargs)
    if func is np.interp:
        return _interp(*args, **kwargs)
    if args and isinstance(args[0], (Quantity, _Point)) and isinstance(args[0].unit, _Coordinate):
        if func not in (
            np.isclose,
            np.allclose,
            np.reshape,
            np.transpose,
            np.argmin,
            np.argmax,
            np.nanargmin,
            np.nanargmax,
            np.median,
            np.mean,
            np.std,
            np.var,
            np.nanmedian,
            np.nanmean,
            np.nanstd,
            np.nanvar,
            np.min,
            np.max,
            np.amin,
            np.amax,
            np.nanmin,
            np.nanmax,
        ):
            raise TypeError(
                'This NumPy function is undefined for affine points; convert explicitly.'
            )
    if func in (np.isclose, np.allclose):
        return _close_comparison(func, *args, **kwargs)
    if func in (np.concatenate, np.stack):
        return _join_arrays(func, *args, **kwargs)
    if func is np.trapezoid:
        return _trapezoid(*args, **kwargs)
    if func is np.diff:
        return _diff(*args, **kwargs)
    if func is np.gradient:
        return _gradient(*args, **kwargs)
    if (
        not args
        or not isinstance(args[0], (Quantity, _Point))
        or not _is_supported_operand(args[0])
    ):
        return NotImplemented
    handler = _QUANTITY_FUNCTIONS.get(func)
    if handler is None:
        return NotImplemented
    return handler(*args, **kwargs)


def _where(condition: Any, x: Any, y: Any, /) -> Any:
    _validate_where({'where': condition})
    a, b, unit = _selection_parts(x, y)
    return _wrap(np.where(condition, a, b), unit)


def _clip(
    a: Any,
    /,
    a_min: Any = _UNSET,
    a_max: Any = _UNSET,
    out: Any = None,
    *,
    min: Any = _UNSET,
    max: Any = _UNSET,
    **kwargs: Any,
) -> Any:
    if not isinstance(a, (Quantity, _Point)) or not _is_supported_operand(a):
        error_message = 'clip requires a Quantity or point input.'
        raise TypeError(error_message)
    if a_min is _UNSET and a_max is _UNSET:
        a_min = None if min is _UNSET else min
        a_max = None if max is _UNSET else max
    elif a_min is _UNSET or a_max is _UNSET or min is not _UNSET or max is not _UNSET:
        error_message = 'Supply both a_min/a_max, or use only min/max.'
        raise TypeError(error_message)
    if kwargs.keys() - {'where', 'dtype', 'casting', 'order', 'subok'}:
        error_message = 'Unsupported clip keyword.'
        raise TypeError(error_message)
    _validate_where(kwargs)
    if 'dtype' in kwargs and np.dtype(kwargs['dtype']).kind not in 'iufc':
        error_message = 'clip requires a numeric output dtype.'
        raise TypeError(error_message)
    lower = None if a_min is None else _selection_parts(a, a_min)[1]
    upper = None if a_max is None else _selection_parts(a, a_max)[1]
    result = _wrap(np.clip(_numeric_value(a.value), lower, upper, **kwargs), a.unit)
    return _write_output(
        result,
        out,
        where=kwargs.get('where', True),
        casting=kwargs.get('casting', 'same_kind'),
        broadcast=True,
    )


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
    first = next((x for x in arrays if isinstance(x, (Quantity, _Point))), None)
    if first is None:
        return NotImplemented
    values = []
    for x in arrays:
        raw, unit = _operand_parts(x)
        if unit.dimension != first.dimension:
            raise DimensionError('All concatenated inputs must have compatible dimensions.')
        if isinstance(unit, _Coordinate) != isinstance(first.unit, _Coordinate):
            raise DimensionError('Concatenation requires matching point/difference semantics.')
        values.append(_numeric_value(convert(raw, unit, first.unit)))
    if dtype is not None and np.dtype(dtype).kind not in 'iufc':
        raise TypeError('Only numeric output dtypes are supported.')
    result = _wrap(func(values, axis=axis, dtype=dtype, casting=casting), first.unit)
    return _write_output(result, out, casting=casting)


def _reshape(
    q: Any, shape: Any, order: Literal['C', 'F', 'A'] = 'C', *, copy: bool | None = None
) -> Any:
    return _wrap(np.reshape(q.value, shape, order=order, copy=copy), q.unit)


def _transpose(q: Any, axes: Any = None) -> Any:
    return _wrap(np.transpose(q.value, axes), q.unit)


def _interp(
    x: Any, xp: Any, fp: Any, left: Any = None, right: Any = None, period: Any = None
) -> Any:
    if period is not None:
        error_message = 'period is not supported for Quantity interpolation.'
        raise TypeError(error_message)
    inputs = tuple(np.asarray(v) if isinstance(v, (list, tuple)) else v for v in (x, xp, fp))
    if not all(_is_supported_operand(v) for v in inputs):
        return NotImplemented
    query, query_unit = _linear_parts(inputs[0])
    coordinates, coordinate_unit = _linear_parts(inputs[1])
    samples, sample_unit = _linear_parts(inputs[2])
    if not compatible(query_unit, coordinate_unit):
        error_message = 'x and xp must have compatible dimensions.'
        raise DimensionError(error_message)
    query = scale_magnitude(query, query_unit.factor / coordinate_unit.factor)
    # Check the coordinates actually used by interp's double-precision kernel.
    # Compare adjacent values directly to avoid integer diff overflow.
    coordinates = np.asarray(coordinates).astype(np.float64, casting='safe', copy=False)
    if coordinates.ndim != 1 or coordinates.size == 0:
        error_message = 'xp must be a nonempty one-dimensional coordinate array.'
        raise ValueError(error_message)
    if not np.all(np.isfinite(coordinates)) or not np.all(coordinates[1:] > coordinates[:-1]):
        error_message = 'xp must contain finite, strictly increasing coordinates.'
        raise ValueError(error_message)
    boundaries = {}
    for name, boundary in (('left', left), ('right', right)):
        if boundary is None:
            continue
        if not _is_supported_operand(boundary):
            error_message = f'{name} must be a numeric scalar or scalar Quantity.'
            raise TypeError(error_message)
        value, unit = _linear_parts(boundary)
        if np.ndim(value) != 0:
            error_message = f'{name} must be a numeric scalar or scalar Quantity.'
            raise TypeError(error_message)
        if not compatible(unit, sample_unit):
            error_message = f'{name} has an incompatible dimension.'
            raise DimensionError(error_message)
        boundaries[name] = scale_magnitude(value, unit.factor / sample_unit.factor)
    return Quantity(np.interp(query, coordinates, samples, **boundaries), sample_unit)


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
    if isinstance(unit, _Coordinate) or isinstance(coordinate_unit, _Coordinate):
        raise TypeError('Integration requires explicit physical units for affine coordinates.')
    return _wrap(result, unit * coordinate_unit)


def _diff(a: Any, n: int = 1, axis: int = -1, prepend: Any = _UNSET, append: Any = _UNSET) -> Any:
    if not _is_supported_operand(a):
        return NotImplemented
    # Like NumPy, the no-op retains the input and ignores axis and boundaries.
    if n == 0:
        return a
    value, unit = _linear_parts(a)
    boundaries = {}
    for name, boundary in (('prepend', prepend), ('append', append)):
        if boundary is _UNSET:
            continue
        if not _is_supported_operand(boundary):
            return NotImplemented
        raw, source = _linear_parts(boundary)
        if not compatible(source, unit):
            raise DimensionError(f'{name} has an incompatible dimension.')
        boundaries[name] = scale_magnitude(raw, source.factor / unit.factor)
    return _wrap(np.diff(value, n=n, axis=axis, **boundaries), unit)


def _gradient(f: Any, *varargs: Any, axis: Any = None, edge_order: Literal[1, 2] = 1) -> Any:
    if not all(_is_supported_operand(x) for x in (f, *varargs)):
        return NotImplemented
    value, unit = _linear_parts(f)
    spacings = [_linear_parts(x) for x in varargs]
    result = np.gradient(value, *(x[0] for x in spacings), axis=axis, edge_order=edge_order)
    # NumPy validates axes, coordinate shapes, and argument counts. Scalar
    # spacing applies to every selected axis; otherwise each axis has its own.
    if not isinstance(result, tuple):
        return _wrap(result, unit / (spacings[0][1] if spacings else Unit()))
    if not spacings or len(spacings) == 1 and np.ndim(spacings[0][0]) == 0:
        coordinate_units = [spacings[0][1] if spacings else Unit()] * len(result)
    else:
        coordinate_units = [x[1] for x in spacings]
    return tuple(_wrap(raw, unit / source) for raw, source in zip(result, coordinate_units))


def _argmin_max(
    func: Any, q: Any, axis: Any = None, out: Any = None, *, keepdims: bool = False
) -> Any:
    return _write_output(func(q.value, axis=axis, keepdims=keepdims), out)


def _median(
    func: Any,
    q: Any,
    axis: Any = None,
    out: Any = None,
    overwrite_input: bool = False,
    keepdims: bool = False,
) -> Any:
    if overwrite_input:
        error_message = 'overwrite_input is not supported for Quantity.'
        raise TypeError(error_message)
    return _write_output(_wrap(func(q.value, axis=axis, keepdims=keepdims), q.unit), out)


def _cumsum(q: Any, axis: Any = None, dtype: Any = None, out: Any = None) -> Any:
    _validate_dtype(dtype)
    return _write_output(_wrap(np.cumsum(q.value, axis=axis, dtype=dtype), q.unit), out)


def _mean(
    func: Any,
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
    result = _wrap(func(q.value, axis=axis, dtype=dtype, keepdims=keepdims, where=where), q.unit)
    return _write_output(result, out)


def _statistic_parameter(value: Any, name: str) -> Any:
    """Normalize dimensionless weights or probabilities for NumPy statistics."""
    if isinstance(value, Quantity):
        if value.dimension != DIMENSIONLESS:
            error_message = f'{name} must be dimensionless.'
            raise DimensionError(error_message)
        return scale_magnitude(value.value, value.unit.factor)
    if isinstance(value, (list, tuple)):
        value = np.asarray(value)
    if not _is_supported_operand(value) or isinstance(value, _Point):
        error_message = f'{name} must be numeric dimensionless values.'
        raise TypeError(error_message)
    return _numeric_value(value)


def _average(
    a: Any,
    axis: Any = None,
    weights: Any = None,
    returned: bool = False,
    *,
    keepdims: bool = False,
) -> Any:
    if weights is not None:
        weights = _statistic_parameter(weights, 'weights')
    value = _numeric_value(a.value)
    if returned:
        average, weight_sum = np.average(
            value, axis=axis, weights=weights, returned=True, keepdims=keepdims
        )
        return Quantity(average, a.unit), Quantity(weight_sum, Unit())
    return Quantity(np.average(value, axis=axis, weights=weights, keepdims=keepdims), a.unit)


def _quantile(
    func: Any,
    a: Any,
    q: Any,
    axis: Any = None,
    out: Any = None,
    overwrite_input: bool = False,
    method: str = 'linear',
    keepdims: bool = False,
    *,
    weights: Any = None,
) -> Any:
    if overwrite_input:
        error_message = 'overwrite_input is not supported for Quantity.'
        raise TypeError(error_message)
    q = _statistic_parameter(q, 'q')
    value = _numeric_value(a.value)
    if weights is not None:
        weights = _statistic_parameter(weights, 'weights')
        if func in (np.nanquantile, np.nanpercentile):
            # NumPy 2.1 mishandles tuple axes with weights; its NaN kernels
            # also index reduced-axis weights without broadcasting in 2.1/2.5.
            if isinstance(axis, tuple):
                error_message = 'Weighted NaN quantiles require one integer axis or axis=None.'
                raise TypeError(error_message)
            if np.ndim(value) > 1 and axis is not None and np.shape(weights) != np.shape(value):
                error_message = (
                    'NaN quantiles require full-shape weights for multidimensional samples.'
                )
                raise ValueError(error_message)
    result = func(value, q, axis=axis, method=method, keepdims=keepdims, weights=weights)
    return _write_output(Quantity(result, a.unit), out)


def _sum(
    func: Any,
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
    value, unit = _linear_parts(q)
    initial_value = None
    if initial is not None:
        if not _is_supported_operand(initial):
            error_message = 'initial must be a numeric scalar or scalar Quantity.'
            raise TypeError(error_message)
        initial_value, initial_unit = _linear_parts(initial)
        if np.ndim(initial_value) != 0:
            error_message = 'initial must be a numeric scalar or scalar Quantity.'
            raise TypeError(error_message)
        if not compatible(initial_unit, unit):
            # Bare zero is the additive identity in any dimension. A Quantity
            # zero still has its declared dimension and must be compatible.
            if isinstance(initial, (Quantity, _Point)) or initial != 0:
                error_message = 'initial has an incompatible dimension.'
                raise DimensionError(error_message)
        else:
            initial_value = scale_magnitude(initial_value, initial_unit.factor / unit.factor)
    result = _wrap(
        func(value, axis=axis, dtype=dtype, keepdims=keepdims, initial=initial_value, where=where),
        unit,
    )
    return _write_output(result, out)


def _prod(
    func: Any,
    q: Any,
    axis: Any = None,
    dtype: Any = None,
    out: Any = None,
    keepdims: bool = False,
    initial: Any = 1,
    where: Any = True,
) -> Any:
    _validate_dtype(dtype)
    _validate_where({'where': where})
    initial_value, initial_unit = None, Unit()
    if initial is not None:
        if not _is_supported_operand(initial):
            error_message = 'initial must be a numeric scalar or scalar Quantity.'
            raise TypeError(error_message)
        initial_value, initial_unit = _linear_parts(initial)
        if np.ndim(initial_value) != 0:
            error_message = 'initial must be a numeric scalar or scalar Quantity.'
            raise TypeError(error_message)

    value, unit = _linear_parts(q)
    mask = np.broadcast_to(where, np.shape(value))
    # Different counts still have one coherent unit for dimensionless inputs.
    if unit.dimension == DIMENSIONLESS:
        value = scale_magnitude(value, unit.factor, where=mask)
        unit = Unit()
    result = func(
        value, axis=axis, dtype=dtype, keepdims=keepdims, initial=initial_value, where=where
    )
    if unit.dimension != DIMENSIONLESS:
        shape = np.shape(value)
        axes = range(len(shape)) if axis is None else axis if isinstance(axis, tuple) else (axis,)
        # NumPy has already validated axes, including its scalar 0/-1 special case.
        count = math.prod(shape[i] for i in axes) if shape else 1
        if func is np.nanprod:
            mask = mask & ~np.isnan(value)
        if np.ndim(where) == 0:
            count = count if bool(where) else 0
        if np.ndim(where) != 0 or func is np.nanprod:
            counts = np.sum(mask, axis=axis)
            if np.size(counts):
                count = int(np.asarray(counts).flat[0])
                if np.any(counts != count):
                    error_message = 'Product reductions require equal element counts per result.'
                    raise DimensionError(error_message)
            # With no output elements, retain the nominal reduction extent.
        unit = unit**count
    assert isinstance(unit, Unit) and isinstance(initial_unit, Unit)
    return _write_output(_wrap(result, unit * initial_unit), out)


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
    delta = q.unit.delta_unit if isinstance(q.unit, _Coordinate) else q.unit
    unit = delta**2 if func in (np.var, np.nanvar) else delta
    result = _wrap(
        func(q.value, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, where=where), unit
    )
    return _write_output(result, out)


def _validate_dtype(dtype: Any) -> None:
    if dtype is not None and np.dtype(dtype).kind not in 'iufc':
        raise TypeError('Only numeric output dtypes are supported.')


def power(q: Any, exponent: Any) -> Any:
    if isinstance(exponent, (Fraction, Decimal)):
        exact = as_exponent(exponent)
        unit = Unit() if q.dimension == DIMENSIONLESS else q.unit**exact
        value = scale_magnitude(q.value, q.unit.factor) if q.dimension == DIMENSIONLESS else q.value
        return _wrap(np.power(value, float(exponent)), unit)
    return apply_ufunc('power', q, exponent)


# These handlers share the validated Quantity-first input contract. Sequence
# and mixed-input functions retain their separate dispatch paths above.


def affine_magnitude(value: Any, a: Any, b: Any) -> Any:
    if np.iscomplexobj(value):
        raise TypeError('Affine coordinates require real magnitudes.')
    a = int(a) if isinstance(a, Fraction) and a.denominator == 1 else _numeric_value(a)
    b = int(b) if isinstance(b, Fraction) and b.denominator == 1 else _numeric_value(b)
    result = np.multiply(value, a) if a != 1 else value
    return np.add(result, b) if b != 0 else result


def logarithm_magnitude(value: Any) -> Any:
    if np.iscomplexobj(value) or np.any(value < 0):
        raise ValueError('Logarithmic conversion requires nonnegative real values.')
    with np.errstate(divide='ignore'):
        return np.log10(value)


def exponential_magnitude(value: Any) -> Any:
    if np.iscomplexobj(value):
        raise TypeError('Logarithmic conversion requires real magnitudes.')
    return np.power(10.0, value)


def comparison_constant(left: Any, right: Any, value: bool) -> Any:
    return np.full(np.broadcast_shapes(np.shape(left), np.shape(right)), value, dtype=bool)


def zero_tolerance(value: Any) -> None:
    if np.iscomplexobj(value) or not np.all(np.equal(value, 0)):
        raise ValueError(
            'Affine relative/bare tolerance must be zero; use a difference absolute tolerance.'
        )


def affine_close_magnitude(
    a: Any, b: Any, tolerance: Any, relative: Any, *, equal_nan: bool
) -> Any:
    if np.iscomplexobj(tolerance) or not np.all(np.isfinite(tolerance)) or np.any(tolerance < 0):
        raise ValueError('Absolute tolerance must be finite and nonnegative.')
    a, b, tolerance, _ = np.broadcast_arrays(a, b, tolerance, relative)
    # Floating subtraction avoids signed/unsigned integer overflow.
    dtype = np.result_type(a, b, np.float64)
    a, b = np.asarray(a, dtype=dtype), np.asarray(b, dtype=dtype)
    with np.errstate(invalid='ignore', over='ignore'):
        result = (a == b) | (np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) <= tolerance))
    return result | (np.isnan(a) & np.isnan(b)) if equal_nan else result


def _affine_ufunc(ufunc: Any, method: str, inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    from .._coordinates.points import _addition_parts, _compare

    if method != '__call__':
        raise TypeError('Affine points cannot be accumulated or reduced with arithmetic ufuncs.')
    if kwargs.keys() - {'where', 'dtype', 'casting', 'order', 'subok'}:
        return NotImplemented
    _validate_where(kwargs)
    if 'dtype' in kwargs and np.dtype(kwargs['dtype']).kind not in 'biuf':
        raise TypeError('Affine operations require real numeric output.')
    if 'where' in kwargs:
        kwargs = {**kwargs, 'out': (None,) * ufunc.nout}
    if any(isinstance(x, (_Coordinate, Unit)) for x in inputs):
        raise TypeError('Affine unit objects cannot participate in this ufunc.')
    if 'where' in kwargs:
        mask = kwargs['where']
        inputs = tuple(
            _wrap(np.where(mask, x.value, 0), x.unit)
            if isinstance(x, (Quantity, _Point))
            else np.where(mask, x, 0)
            for x in inputs
        )
    if ufunc in (np.add, np.subtract):
        a, b, unit = _addition_parts(*inputs, subtract=ufunc is np.subtract)
        return _wrap(ufunc(a, b, **kwargs), unit)
    if ufunc in _COMPARISON_UFUNCS.values():
        op = next(op for op, func in _COMPARISON_UFUNCS.items() if func is ufunc)
        left, right = inputs
        if not isinstance(left, (Quantity, _Point)):
            left = _wrap(left, Unit())
        if not isinstance(left, _Point) and isinstance(right, _Point):
            reverse = {
                operator.eq: operator.eq,
                operator.ne: operator.ne,
                operator.lt: operator.gt,
                operator.le: operator.ge,
                operator.gt: operator.lt,
                operator.ge: operator.le,
            }
            result = _compare(right, left, reverse[op])
        else:
            result = _compare(left, right, op)
        return np.equal(result, True, **kwargs) if kwargs else result
    q = inputs[0]
    if ufunc is np.positive:
        return _wrap(np.positive(q.value, **kwargs), q.unit)
    if ufunc in (np.isfinite, np.isnan, np.isinf):
        return ufunc(q.value, **kwargs)
    if ufunc in (np.minimum, np.maximum):
        a, b, unit = _selection_parts(*inputs)
        return _wrap(ufunc(a, b, **kwargs), unit)
    raise TypeError('This ufunc is undefined for affine points; convert to physical units first.')


def _extremum(
    func: Any,
    q: Any,
    axis: Any = None,
    out: Any = None,
    keepdims: bool = False,
    initial: Any = _UNSET,
    where: Any = True,
) -> Any:
    _validate_where({'where': where})
    kwargs = dict(axis=axis, keepdims=keepdims, where=where)
    if initial is not _UNSET:
        if not isinstance(initial, (Quantity, _Point)):
            error_message = 'initial must carry the result coordinate/unit.'
            raise TypeError(error_message)
        kwargs['initial'] = _selection_parts(q, initial)[1]
    return _write_output(_wrap(func(q.value, **kwargs), q.unit), out)


_QUANTITY_FUNCTIONS = {
    np.reshape: _reshape,
    np.transpose: _transpose,
    np.argmin: partial(_argmin_max, np.argmin),
    np.argmax: partial(_argmin_max, np.argmax),
    np.nanargmin: partial(_argmin_max, np.nanargmin),
    np.nanargmax: partial(_argmin_max, np.nanargmax),
    np.median: partial(_median, np.median),
    np.nanmedian: partial(_median, np.nanmedian),
    np.cumsum: _cumsum,
    np.prod: partial(_prod, np.multiply.reduce),
    np.nanprod: partial(_prod, np.nanprod),
    np.std: partial(_std_var, np.std),
    np.var: partial(_std_var, np.var),
    np.nanstd: partial(_std_var, np.nanstd),
    np.nanvar: partial(_std_var, np.nanvar),
    np.sum: partial(_sum, np.add.reduce),
    np.nansum: partial(_sum, np.nansum),
    np.mean: partial(_mean, np.mean),
    np.nanmean: partial(_mean, np.nanmean),
    np.average: _average,
    np.quantile: partial(_quantile, np.quantile),
    np.percentile: partial(_quantile, np.percentile),
    np.nanquantile: partial(_quantile, np.nanquantile),
    np.nanpercentile: partial(_quantile, np.nanpercentile),
    np.min: partial(_extremum, np.min),
    np.max: partial(_extremum, np.max),
    np.amin: partial(_extremum, np.amin),
    np.amax: partial(_extremum, np.amax),
    np.nanmin: partial(_extremum, np.nanmin),
    np.nanmax: partial(_extremum, np.nanmax),
}
