"""Optional NumPy kernels. Imported only for NumPy magnitudes or dispatch.

An identity scale retains storage. Integral scales use a Python int multiplier;
nonintegral rational scales use one Python float multiplier, and irrational
scales use Factor's approximation. NumPy then controls promotion and errors.
No Fraction/Decimal object arrays are produced. Explicit outputs write
only after computing, converting, and casting a temporary result.
"""

import math
import operator
import sys
import warnings
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
from ..exceptions import _WARNING_SKIP_PREFIXES, DimensionError, UnitOperationError
from ..factor import Factor, one
from ..logquantity import LogQuantity
from ..quantity import Quantity
from ..unit import Unit
from . import as_exponent
from .conversion import compatible
from .numpy_typing import _UNSET

_MATRIX_VECTOR_UFUNCS = tuple(
    value for name in ('matvec', 'vecmat') if isinstance(value := getattr(np, name, None), np.ufunc)
)

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


def astype_magnitude(value: Any, dtype: Any) -> Any:
    _validate_dtype(dtype)
    return value.astype(dtype)


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
        error_message = 'This operation requires linear quantities or differences.'
        raise UnitOperationError(error_message)
    return raw, unit


def _align_operands(left: Any, right: Any) -> tuple[Any, Any, Unit]:
    a, u = _linear_parts(left)
    b, v = _linear_parts(right)
    if isinstance(u, _Coordinate) or isinstance(v, _Coordinate):
        error_message = 'Use point/difference arithmetic for affine coordinates.'
        raise UnitOperationError(error_message)
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
            raise UnitOperationError(error_message)
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
    if method != '__call__':
        from .numpy_writes import dispatch_method

        return dispatch_method(ufunc, method, inputs, kwargs)
    if ufunc in (np.modf, np.frexp, np.divmod) and method == '__call__':
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
    if method != '__call__':
        return NotImplemented
    allowed = {'where', 'dtype', 'casting', 'order', 'subok'}
    if ufunc is np.vecdot:
        allowed |= {'axis', 'axes'}
    if kwargs.keys() - allowed:
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
    if ufunc in (np.minimum, np.maximum, np.fmin, np.fmax):
        a, b, unit = _selection_parts(*inputs)
        return _wrap(ufunc(a, b, **kwargs), unit)
    if ufunc in (np.remainder, np.fmod, np.floor_divide, np.divmod):
        a, b, unit = _align_operands(*inputs)
        result = ufunc(a, b, **kwargs)
        if ufunc is np.divmod:
            return _wrap(result[0], Unit()), _wrap(result[1], unit)
        return _wrap(result, Unit() if ufunc is np.floor_divide else unit)
    if ufunc is np.copysign:
        a, unit = _linear_parts(inputs[0])
        b, _ = _linear_parts(inputs[1])
        return _wrap(ufunc(a, b, **kwargs), unit)
    if ufunc in (np.logaddexp, np.logaddexp2):
        a, u = _linear_parts(inputs[0])
        b, v = _linear_parts(inputs[1])
        if u.dimension != DIMENSIONLESS or v.dimension != DIMENSIONLESS:
            error_message = 'This function requires dimensionless inputs.'
            raise DimensionError(error_message)
        return _wrap(
            ufunc(scale_magnitude(a, u.factor), scale_magnitude(b, v.factor), **kwargs), Unit()
        )
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
    if ufunc in (
        np.multiply,
        np.divide,
        np.true_divide,
        np.matmul,
        np.vecdot,
        *_MATRIX_VECTOR_UFUNCS,
    ):
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
    if ufunc in (
        np.positive,
        np.negative,
        np.absolute,
        np.fabs,
        np.conjugate,
        np.floor,
        np.ceil,
        np.trunc,
        np.rint,
    ):
        return _wrap(ufunc(a, **kwargs), unit)
    if ufunc is np.ldexp:
        exponent, exponent_unit = _linear_parts(inputs[1])
        if exponent_unit.dimension != DIMENSIONLESS:
            error_message = 'ldexp requires dimensionless integer exponents.'
            raise DimensionError(error_message)
        exponent = scale_magnitude(exponent, exponent_unit.factor)
        return _wrap(ufunc(a, exponent, **kwargs), unit)
    if ufunc is np.nextafter:
        a, b, unit = _align_operands(*inputs)
        return _wrap(ufunc(a, b, **kwargs), unit)
    if ufunc is np.spacing:
        return _wrap(ufunc(a, **kwargs), unit)
    if ufunc is np.heaviside:
        b, v = _linear_parts(inputs[1])
        if v.dimension != DIMENSIONLESS:
            error_message = 'heaviside requires a dimensionless value at zero.'
            raise DimensionError(error_message)
        return _wrap(ufunc(a, scale_magnitude(b, v.factor), **kwargs), Unit())
    if ufunc is np.sign:
        return _wrap(ufunc(a, **kwargs), Unit())
    if ufunc in (np.degrees, np.rad2deg, np.radians, np.deg2rad):
        if unit.dimension != DIMENSIONLESS:
            error_message = 'Angle conversion requires dimensionless input.'
            raise DimensionError(error_message)
        source = Unit('rad') if ufunc in (np.degrees, np.rad2deg) else Unit('deg')
        target = Unit('deg') if ufunc in (np.degrees, np.rad2deg) else Unit('rad')
        return _wrap(ufunc(scale_magnitude(a, unit.factor / source.factor), **kwargs), target)
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
    if ufunc in (
        np.sin,
        np.cos,
        np.tan,
        np.sinh,
        np.cosh,
        np.tanh,
        np.arcsinh,
        np.arccosh,
        np.arctanh,
        np.exp,
        np.expm1,
        np.exp2,
        np.log,
        np.log2,
        np.log10,
        np.log1p,
    ):
        if unit.dimension != DIMENSIONLESS:
            raise DimensionError('This function requires dimensionless input.')
        return _wrap(ufunc(scale_magnitude(a, unit.factor), **kwargs), Unit())
    # SciPy's public ufunc is already loaded when it dispatches here. Identity
    # lookup avoids importing SciPy or accepting an unrelated same-name ufunc.
    if ufunc is getattr(sys.modules.get('scipy.special'), 'jv', None):
        a, u = _linear_parts(inputs[0])
        b, v = _linear_parts(inputs[1])
        if u.dimension != DIMENSIONLESS or v.dimension != DIMENSIONLESS:
            error_message = 'Bessel order and argument must be dimensionless.'
            raise DimensionError(error_message)
        return _wrap(
            ufunc(scale_magnitude(a, u.factor), scale_magnitude(b, v.factor), **kwargs), Unit()
        )
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
    if func in (
        np.block,
        np.nan_to_num,
        np.ediff1d,
        np.cumulative_sum,
        np.cumulative_prod,
        np.array_equal,
        np.array_equiv,
        np.isreal,
        np.iscomplex,
        np.isrealobj,
        np.iscomplexobj,
        np.shares_memory,
        np.may_share_memory,
        np.all,
        np.any,
        np.choose,
        np.select,
        np.unstack,
        np.i0,
        np.isposinf,
        np.isneginf,
        np.resize,
        np.trim_zeros,
        np.astype,
    ):
        from .numpy_misc import dispatch

        return dispatch(func, *args, **kwargs)
    if func in (np.copyto, np.put, np.put_along_axis, np.putmask, np.place, np.fill_diagonal):
        from .numpy_writes import FUNCTIONS

        return FUNCTIONS[func](*args, **kwargs)
    if func in (
        np.full_like,
        np.linspace,
        np.geomspace,
        np.logspace,
        np.meshgrid,
        np.pad,
        np.lib.stride_tricks.sliding_window_view,
    ):
        from .numpy_creation import FUNCTIONS

        return FUNCTIONS[func](*args, **kwargs)
    if func in _FFT_FUNCTIONS:
        return _fft(func, *args, **kwargs)
    if func in (np.cov, np.corrcoef):
        return _covariance(func, *args, **kwargs)
    if func in (
        np.histogram,
        np.histogram_bin_edges,
        np.histogram2d,
        np.histogramdd,
        np.digitize,
        np.bincount,
    ):
        from .numpy_histograms import dispatch

        return dispatch(func, *args, **kwargs)
    if func is np.ptp:
        return _ptp(*args, **kwargs)
    if func in (
        np.dot,
        np.linalg.matmul,
        np.linalg.outer,
        np.linalg.cross,
        np.linalg.tensordot,
        np.inner,
        np.outer,
        np.vdot,
        np.linalg.vecdot,
        np.tensordot,
        np.kron,
        np.cross,
        np.convolve,
        np.correlate,
    ):
        return _binary_product(func, *args, **kwargs)
    if func in (np.einsum, np.einsum_path):
        return _einsum(func, *args, **kwargs)
    if func is np.linalg.multi_dot:
        return _multi_dot(*args, **kwargs)
    if func is np.linalg.norm:
        return _norm(*args, **kwargs)
    if func in _LINALG_FUNCTIONS:
        return _LINALG_FUNCTIONS[func](*args, **kwargs)
    if func is np.trace:
        return _trace(*args, **kwargs)
    if func is np.where:
        return _where(*args, **kwargs)
    if func is np.clip:
        return _clip(*args, **kwargs)
    if func is np.interp:
        return _interp(*args, **kwargs)
    if func in (np.atleast_1d, np.atleast_2d, np.atleast_3d, np.broadcast_arrays):
        return _multiple_arrays(func, *args, **kwargs)
    if func in (np.hstack, np.vstack, np.dstack, np.column_stack):
        return _stack_arrays(func, *args, **kwargs)
    if func is np.append:
        return _append(*args, **kwargs)
    if func is np.insert:
        return _insert(*args, **kwargs)
    if func is np.compress:
        return _compress(*args, **kwargs)
    if func is np.extract:
        return _extract(*args, **kwargs)
    if func is np.lexsort:
        return _lexsort(*args, **kwargs)
    if func is np.searchsorted:
        return _searchsorted(*args, **kwargs)
    if func in (np.isin, np.intersect1d, np.union1d, np.setdiff1d, np.setxor1d):
        return _set_operation(func, *args, **kwargs)
    if args and isinstance(args[0], (Quantity, _Point)) and isinstance(args[0].unit, _Coordinate):
        if func not in _ARRAY_FUNCTIONS and func not in (
            np.isclose,
            np.allclose,
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
            error_message = (
                'This NumPy function does not support coordinate points; convert explicitly.'
            )
            raise TypeError(error_message)
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


def _fft(func: Any, a: Any, *args: Any, **kwargs: Any) -> Any:
    if not _is_supported_operand(a):
        error_message = 'FFT requires numeric scalars, ndarrays or linear quantities.'
        raise TypeError(error_message)
    values, unit = _linear_parts(a)
    # Every supported transform has three options followed by out, after a:
    # n/axis/norm for 1-D transforms, or s/axes/norm for multidimensional ones.
    if len(args) > 4 or (len(args) == 4 and 'out' in kwargs):
        error_message = 'Supply at most three FFT options and one output buffer.'
        raise TypeError(error_message)
    if len(args) == 4:
        args, out = args[:3], args[3]
    else:
        out = kwargs.pop('out', None)
    result = Quantity(_call_with_warnings(func, values, *args, **kwargs), unit)
    return _write_output(result, out)


_FFT_FUNCTIONS = {
    np.fft.fft,
    np.fft.ifft,
    np.fft.fft2,
    np.fft.ifft2,
    np.fft.fftn,
    np.fft.ifftn,
    np.fft.rfft,
    np.fft.irfft,
    np.fft.rfft2,
    np.fft.irfft2,
    np.fft.rfftn,
    np.fft.irfftn,
    np.fft.hfft,
    np.fft.ihfft,
}


def _linalg_parts(value: Any) -> tuple[Any, Unit]:
    if not _is_supported_operand(value):
        error_message = 'Linear algebra requires numeric scalars, ndarrays or linear quantities.'
        raise TypeError(error_message)
    return _linear_parts(value)


def _linalg_cutoff(value: Any, unit: Unit, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = np.asarray(value)
    raw, source = _linalg_parts(value)
    if np.iscomplexobj(raw):
        error_message = f'{name} must be real.'
        raise TypeError(error_message)
    # Bare cutoffs are in the stored numerical domain, as in NumPy. Quantity
    # cutoffs explicitly specify either an absolute unit or a relative scale.
    if isinstance(value, Quantity):
        if not compatible(source, unit):
            error_message = f'{name} has an incompatible dimension or logarithmic degree.'
            raise DimensionError(error_message)
        return scale_magnitude(raw, source.factor / unit.factor)
    return raw


def _solve(a: Any, b: Any) -> Any:
    av, au = _linalg_parts(a)
    bv, bu = _linalg_parts(b)
    return Quantity(np.linalg.solve(av, bv), bu / au)


def _tensorsolve(a: Any, b: Any, axes: Any = None) -> Any:
    av, au = _linalg_parts(a)
    bv, bu = _linalg_parts(b)
    return Quantity(np.linalg.tensorsolve(av, bv, axes=axes), bu / au)


def _lstsq(a: Any, b: Any, rcond: Any = None) -> Any:
    av, au = _linalg_parts(a)
    bv, bu = _linalg_parts(b)
    solution, residuals, rank, singular = np.linalg.lstsq(
        av, bv, rcond=_linalg_cutoff(rcond, Unit(), 'rcond')
    )
    return Quantity(solution, bu / au), Quantity(residuals, bu**2), rank, Quantity(singular, au)


def _inverse(func: Any, a: Any, *args: Any, **kwargs: Any) -> Any:
    values, unit = _linalg_parts(a)
    return Quantity(func(values, *args, **kwargs), unit**-1)


def _pinv(a: Any, rcond: Any = None, hermitian: bool = False, *, rtol: Any = _UNSET) -> Any:
    options = {} if rtol is _UNSET else {'rtol': _linalg_cutoff(rtol, Unit(), 'rtol')}
    return _inverse(
        np.linalg.pinv,
        a,
        rcond=_linalg_cutoff(rcond, Unit(), 'rcond'),
        hermitian=hermitian,
        **options,
    )


def _matrix_power(a: Any, n: Any) -> Any:
    values, unit = _linalg_parts(a)
    result = np.linalg.matrix_power(values, n)
    return Quantity(result, unit ** operator.index(n))


def _det(a: Any) -> Any:
    values, unit = _linalg_parts(a)
    result = np.linalg.det(values)
    return Quantity(result, unit ** values.shape[-1])


def _norm_result(func: Any, x: Any, **kwargs: Any) -> Any:
    values, unit = _linalg_parts(x)
    ord = kwargs['ord']
    if isinstance(ord, (Quantity, _Point, Unit, _Coordinate)):
        error_message = 'Norm order must be a bare number or a matrix norm string.'
        raise TypeError(error_message)
    result = func(values, **kwargs)
    return Quantity(result, Unit() if ord == 0 else unit)


def _norm(x: Any, ord: Any = None, axis: Any = None, keepdims: bool = False) -> Any:
    return _norm_result(np.linalg.norm, x, ord=ord, axis=axis, keepdims=keepdims)


def _vector_norm(x: Any, /, *, axis: Any = None, keepdims: bool = False, ord: Any = 2) -> Any:
    return _norm_result(np.linalg.vector_norm, x, ord=ord, axis=axis, keepdims=keepdims)


def _matrix_norm(x: Any, /, *, keepdims: bool = False, ord: Any = 'fro') -> Any:
    return _norm_result(np.linalg.matrix_norm, x, ord=ord, keepdims=keepdims)


def _cond(x: Any, p: Any = None) -> Any:
    values, _ = _linalg_parts(x)
    if isinstance(p, (Quantity, _Point, Unit, _Coordinate)):
        error_message = 'Condition number order must be a bare number or a matrix norm string.'
        raise TypeError(error_message)
    return Quantity(np.linalg.cond(values, p=p), Unit())


def _matrix_rank(A: Any, tol: Any = None, hermitian: bool = False, *, rtol: Any = None) -> Any:
    values, unit = _linalg_parts(A)
    return np.linalg.matrix_rank(
        values,
        tol=_linalg_cutoff(tol, unit, 'tol'),
        hermitian=hermitian,
        rtol=_linalg_cutoff(rtol, Unit(), 'rtol'),
    )


def _qr(a: Any, mode: str = 'reduced') -> Any:
    values, unit = _linalg_parts(a)
    if mode not in ('reduced', 'complete', 'r'):
        error_message = (
            'Quantity QR supports only reduced, complete and r modes; '
            'packed raw and deprecated modes are unsupported.'
        )
        raise ValueError(error_message)
    # NumPy's result stubs only allow ndarray fields; dispatch replaces them
    # with quantities while retaining the native namedtuple container.
    result: Any = np.linalg.qr(values, mode=mode)
    if mode == 'r':
        return Quantity(result, unit)
    return result._replace(Q=Quantity(result.Q, Unit()), R=Quantity(result.R, unit))


def _cholesky(a: Any, /, *, upper: bool = False) -> Any:
    values, unit = _linalg_parts(a)
    # NumPy 2.1 supports upper at runtime, but its stubs omit the keyword.
    cholesky = cast(Any, np.linalg.cholesky)
    return Quantity(cholesky(values, upper=upper), unit ** Fraction(1, 2))


def _svd(
    a: Any, full_matrices: bool = True, compute_uv: bool = True, hermitian: bool = False
) -> Any:
    values, unit = _linalg_parts(a)
    result: Any = np.linalg.svd(
        values, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian
    )
    if not compute_uv:
        return Quantity(result, unit)
    return result._replace(
        U=Quantity(result.U, Unit()), S=Quantity(result.S, unit), Vh=Quantity(result.Vh, Unit())
    )


def _svdvals(x: Any, /) -> Any:
    values, unit = _linalg_parts(x)
    return Quantity(np.linalg.svdvals(values), unit)


def _eigen(func: Any, a: Any, *args: Any, **kwargs: Any) -> Any:
    values, unit = _linalg_parts(a)
    result = func(values, *args, **kwargs)
    if isinstance(result, tuple):
        return cast(Any, result)._replace(
            eigenvalues=Quantity(result[0], unit), eigenvectors=Quantity(result[1], Unit())
        )
    return Quantity(result, unit)


_LINALG_FUNCTIONS = {
    np.linalg.solve: _solve,
    np.linalg.tensorsolve: _tensorsolve,
    np.linalg.lstsq: _lstsq,
    np.linalg.inv: partial(_inverse, np.linalg.inv),
    np.linalg.pinv: _pinv,
    np.linalg.tensorinv: partial(_inverse, np.linalg.tensorinv),
    np.linalg.matrix_power: _matrix_power,
    np.linalg.det: _det,
    np.linalg.vector_norm: _vector_norm,
    np.linalg.matrix_norm: _matrix_norm,
    np.linalg.cond: _cond,
    np.linalg.matrix_rank: _matrix_rank,
    np.linalg.qr: _qr,
    np.linalg.cholesky: _cholesky,
    np.linalg.svd: _svd,
    np.linalg.svdvals: _svdvals,
    np.linalg.eig: partial(_eigen, np.linalg.eig),
    np.linalg.eigh: partial(_eigen, np.linalg.eigh),
    np.linalg.eigvals: partial(_eigen, np.linalg.eigvals),
    np.linalg.eigvalsh: partial(_eigen, np.linalg.eigvalsh),
}


def _product_parts(operands: Any) -> tuple[list[Any], Unit]:
    values = []
    unit = Unit()
    for operand in operands:
        if not _is_supported_operand(operand):
            error_message = 'Products require numeric scalars, ndarrays or linear quantities.'
            raise TypeError(error_message)
        value, source = _linear_parts(operand)
        values.append(value)
        unit *= source
    return values, unit


def _binary_product(func: Any, a: Any, b: Any, *args: Any, **kwargs: Any) -> Any:
    values, unit = _product_parts((a, b))
    out = None
    if func in (np.dot, np.outer):
        if args:
            if len(args) != 1 or 'out' in kwargs:
                error_message = 'Supply out once.'
                raise TypeError(error_message)
            out, args = args[0], ()
        else:
            out = kwargs.pop('out', None)
    value = (
        _call_with_warnings(func, *values, *args, **kwargs)
        if func is np.cross
        else func(*values, *args, **kwargs)
    )
    return _write_output(_wrap(value, unit), out)


def _einsum(func: Any, *operands: Any, **kwargs: Any) -> Any:
    if not operands:
        error_message = 'einsum requires operands and subscripts.'
        raise TypeError(error_message)
    if isinstance(operands[0], str):
        indices = range(1, len(operands))
    else:
        if len(operands) < 2:
            error_message = 'einsum requires operands and subscripts.'
            raise TypeError(error_message)
        indices = range(0, len(operands) - 1, 2)
    arguments = list(operands)
    values, unit = _product_parts(arguments[i] for i in indices)
    for i, value in zip(indices, values):
        arguments[i] = value
    if func is np.einsum_path:
        return func(*arguments, **kwargs)
    out = kwargs.pop('out', None)
    _validate_dtype(kwargs.get('dtype'))
    result = _wrap(func(*arguments, **kwargs), unit)
    return _write_output(result, out, casting=kwargs.get('casting', 'safe'))


def _multi_dot(arrays: Any, *, out: Any = None) -> Any:
    values, unit = _product_parts(arrays)
    return _write_output(_wrap(np.linalg.multi_dot(values), unit), out)


def _trace(
    a: Any, offset: int = 0, axis1: int = 0, axis2: int = 1, dtype: Any = None, out: Any = None
) -> Any:
    values, unit = _product_parts((a,))
    _validate_dtype(dtype)
    value = np.trace(values[0], offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)
    return _write_output(_wrap(value, unit), out)


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


def _join_values(arrays: Any) -> tuple[list[Any], Unit | _Coordinate]:
    """Align compatible samples to the first wrapped input's complete unit."""
    arrays = tuple(arrays)
    if not arrays or not all(_is_supported_operand(x) for x in arrays):
        error_message = 'Joining requires numeric scalars, ndarrays, quantities or points.'
        raise TypeError(error_message)
    first = next((x for x in arrays if isinstance(x, (Quantity, _Point))), None)
    if first is None:
        error_message = 'Joining requires at least one Quantity or point.'
        raise TypeError(error_message)
    values = []
    for x in arrays:
        raw, unit = _operand_parts(x)
        if unit.dimension != first.dimension:
            error_message = 'All joined inputs must have compatible dimensions.'
            raise DimensionError(error_message)
        if isinstance(unit, _Coordinate) != isinstance(first.unit, _Coordinate):
            error_message = 'Joining requires matching point/difference semantics.'
            raise DimensionError(error_message)
        if isinstance(unit, _Coordinate) and not unit._compatible_with(cast(Any, first.unit)):
            error_message = 'Joining requires compatible points with the same convention.'
            raise DimensionError(error_message)
        values.append(_numeric_value(convert(raw, unit, first.unit)))
    return values, first.unit


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
        error_message = 'Cannot provide both dtype and out for concatenate/stack.'
        raise TypeError(error_message)
    values, unit = _join_values(arrays)
    if dtype is not None and np.dtype(dtype).kind not in 'iufc':
        error_message = 'Only numeric output dtypes are supported.'
        raise TypeError(error_message)
    result = _wrap(func(values, axis=axis, dtype=dtype, casting=casting), unit)
    return _write_output(result, out, casting=casting)


def _stack_arrays(func: Any, arrays: Any, **kwargs: Any) -> Any:
    values, unit = _join_values(arrays)
    if kwargs.get('dtype') is not None:
        _validate_dtype(kwargs['dtype'])
    return _wrap(func(values, **kwargs), unit)


def _append(arr: Any, values: Any, axis: Any = None) -> Any:
    (a, b), unit = _join_values((arr, values))
    return _wrap(np.append(a, b, axis=axis), unit)


def _insert(arr: Any, obj: Any, values: Any, axis: Any = None) -> Any:
    (a, b), unit = _join_values((arr, values))
    return _wrap(np.insert(a, _snapshot_index(obj), b, axis=axis), unit)


def _split_arrays(func: Any, q: Any, indices_or_sections: Any, *args: Any, **kwargs: Any) -> Any:
    results = func(q.value, _snapshot_index(indices_or_sections), *args, **kwargs)
    return [_wrap(value, q.unit) for value in results]


def _take(
    a: Any,
    indices: Any,
    axis: Any = None,
    out: Any = None,
    mode: Literal['raise', 'wrap', 'clip'] = 'raise',
) -> Any:
    result = np.take(a.value, _snapshot_index(indices), axis=axis, mode=mode)
    return _write_output(_wrap(result, a.unit), out)


def _take_along_axis(arr: Any, indices: Any, axis: Any = _UNSET) -> Any:
    indices = _snapshot_index(indices)
    # NumPy 2.1 requires axis; newer versions supply a default. Let the
    # installed kernel own that version-specific signature behavior.
    options = {} if axis is _UNSET else {'axis': axis}
    return _wrap(np.take_along_axis(arr.value, indices, **options), arr.unit)


def _boolean_condition(condition: Any) -> Any:
    if isinstance(condition, (Quantity, _Point)) or (
        isinstance(condition, np.ndarray) and type(condition) is not np.ndarray
    ):
        error_message = 'condition must be a bare boolean scalar, sequence or ndarray.'
        raise TypeError(error_message)
    if isinstance(condition, (list, tuple)) and not condition:
        return np.array([], dtype=bool)
    condition = np.asarray(condition)
    if condition.dtype.kind != 'b':
        error_message = 'condition must contain only bare booleans.'
        raise TypeError(error_message)
    return condition


def _compress(condition: Any, a: Any, axis: Any = None, out: Any = None) -> Any:
    if not isinstance(a, (Quantity, _Point)) or not _is_supported_operand(a):
        error_message = 'compress requires a Quantity or point input.'
        raise TypeError(error_message)
    result = np.compress(_boolean_condition(condition), a.value, axis=axis)
    return _write_output(_wrap(result, a.unit), out)


def _extract(condition: Any, arr: Any) -> Any:
    if not isinstance(arr, (Quantity, _Point)) or not _is_supported_operand(arr):
        error_message = 'extract requires a Quantity or point input.'
        raise TypeError(error_message)
    return _wrap(np.extract(_boolean_condition(condition), arr.value), arr.unit)


def _delete(arr: Any, obj: Any, axis: Any = None) -> Any:
    return _wrap(np.delete(arr.value, _snapshot_index(obj), axis=axis), arr.unit)


def _lexsort(keys: Any, axis: Any = -1) -> Any:
    keys = tuple(keys)
    if not keys or not all(_is_supported_operand(key) for key in keys):
        error_message = 'lexsort requires numeric arrays, quantities or points as keys.'
        raise TypeError(error_message)
    values = tuple(_operand_parts(key)[0] for key in keys)
    return np.lexsort(values, axis=axis)


def _searchsorted(a: Any, v: Any, side: Any = 'left', sorter: Any = None) -> Any:
    left, right, _ = _selection_parts(a, v)
    if sorter is not None:
        sorter = _snapshot_index(sorter)
    return np.searchsorted(left, right, side=side, sorter=sorter)


def _set_operation(func: Any, ar1: Any, ar2: Any, *args: Any, **kwargs: Any) -> Any:
    left, right, unit = _selection_parts(ar1, ar2)
    result = func(left, right, *args, **kwargs)
    if func is np.isin:
        return result
    if isinstance(result, tuple):
        return (_wrap(result[0], unit), *result[1:])
    return _wrap(result, unit)


def _unique(func: Any, q: Any, *args: Any, **kwargs: Any) -> Any:
    result = func(q.value, *args, **kwargs)
    if isinstance(result, tuple):
        values = _wrap(result[0], q.unit)
        if func is np.unique:
            return (values, *result[1:])
        return cast(Any, result)._replace(values=values)
    return _wrap(result, q.unit)


def _partition(func: Any, q: Any, kth: Any, *args: Any, **kwargs: Any) -> Any:
    result = func(q.value, _snapshot_index(kth), *args, **kwargs)
    return result if func is np.argpartition else _wrap(result, q.unit)


def _reshape(
    q: Any, shape: Any, order: Literal['C', 'F', 'A'] = 'C', *, copy: bool | None = None
) -> Any:
    return _wrap(np.reshape(q.value, shape, order=order, copy=copy), q.unit)


def _transpose(q: Any, axes: Any = None) -> Any:
    return _wrap(np.transpose(q.value, axes), q.unit)


def _squeeze(q: Any, axis: Any = None) -> Any:
    return _wrap(np.squeeze(q.value, axis=axis), q.unit)


def _expand_dims(q: Any, axis: Any) -> Any:
    return _wrap(np.expand_dims(q.value, axis), q.unit)


def _moveaxis(q: Any, source: Any, destination: Any) -> Any:
    return _wrap(np.moveaxis(q.value, source, destination), q.unit)


def _swapaxes(q: Any, axis1: Any, axis2: Any) -> Any:
    return _wrap(np.swapaxes(q.value, axis1, axis2), q.unit)


def _rollaxis(q: Any, axis: Any, start: Any = 0) -> Any:
    return _wrap(np.rollaxis(q.value, axis, start), q.unit)


def _ravel(q: Any, order: Literal['C', 'F', 'A', 'K'] = 'C') -> Any:
    return _wrap(np.ravel(q.value, order=order), q.unit)


def _array_transform(func: Any, q: Any, *args: Any, **kwargs: Any) -> Any:
    """Transform storage without changing values, units or coordinates."""
    return _wrap(func(q.value, *args, **kwargs), q.unit)


def _round(q: Any, decimals: int = 0, out: Any = None) -> Any:
    return _write_output(_wrap(np.round(q.value, decimals=decimals), q.unit), out)


def _fix(q: Any, out: Any = None) -> Any:
    return _write_output(_wrap(_call_with_warnings(np.fix, q.value), q.unit), out)


def _call_with_warnings(func: Any, *args: Any, **kwargs: Any) -> Any:
    # Forward NumPy deprecations at the user's call site,
    # respecting their filters, before committing any output.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', DeprecationWarning)
        value = func(*args, **kwargs)
    for warning in caught:
        warnings.warn(
            warning.message,
            warning.category,
            skip_file_prefixes=(*_WARNING_SKIP_PREFIXES, *np.__path__),
        )
    return value


def _angle(q: Any, deg: bool = False) -> Any:
    return _wrap(np.angle(q.value, deg=deg), Unit('deg') if deg else Unit('rad'))


def _sinc(x: Any) -> Any:
    values, unit = _linear_parts(x)
    if unit.dimension != DIMENSIONLESS:
        error_message = 'sinc requires dimensionless linear input.'
        raise DimensionError(error_message)
    return Quantity(np.sinc(scale_magnitude(values, unit.factor)), Unit())


def _unwrap(p: Any, discont: Any = None, axis: int = -1, *, period: Any = 2 * math.pi) -> Any:
    if p.dimension != DIMENSIONLESS:
        error_message = 'unwrap requires dimensionless linear phase input.'
        raise DimensionError(error_message)

    def threshold(value: Any, name: str) -> Any:
        if not _is_supported_operand(value):
            error_message = f'{name} must be a finite real scalar.'
            raise TypeError(error_message)
        raw, unit = _linear_parts(value)
        if unit.dimension != DIMENSIONLESS:
            error_message = f'{name} must be dimensionless.'
            raise DimensionError(error_message)
        raw = scale_magnitude(raw, unit.factor)
        if np.ndim(raw) != 0 or np.iscomplexobj(raw) or not np.isfinite(raw):
            error_message = f'{name} must be a finite real scalar.'
            raise ValueError(error_message)
        if raw < 0 or (name == 'period' and raw == 0):
            error_message = (
                f'{name} must be positive.' if name == 'period' else f'{name} must be nonnegative.'
            )
            raise ValueError(error_message)
        return raw

    period = threshold(period, 'period')
    discont = None if discont is None else threshold(discont, 'discont')
    values = scale_magnitude(p.value, p.unit.factor)
    result = np.unwrap(values, discont=discont, axis=axis, period=period)
    return _wrap(scale_magnitude(result, 1 / p.unit.factor), p.unit)


def _array_metadata(func: Any, q: Any, *args: Any, **kwargs: Any) -> Any:
    return func(q.value, *args, **kwargs)


def _multiple_arrays(func: Any, *arrays: Any, **kwargs: Any) -> Any:
    """Rearrange each operand independently, retaining bare and wrapped inputs."""
    if not all(_is_supported_operand(x) for x in arrays):
        return NotImplemented
    values = tuple(
        x.value if isinstance(x, (Quantity, _Point)) else _numeric_value(x) for x in arrays
    )
    result = func(*values, **kwargs)
    if func is not np.broadcast_arrays and len(arrays) == 1:
        source = arrays[0]
        return _wrap(result, source.unit) if isinstance(source, (Quantity, _Point)) else result
    return type(result)(
        _wrap(raw, source.unit) if isinstance(source, (Quantity, _Point)) else raw
        for raw, source in zip(result, arrays)
    )


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


def _delta_unit(unit: Unit | _Coordinate) -> Unit:
    return unit.delta_unit if isinstance(unit, _Coordinate) else unit


def _ptp(a: Any, axis: Any = None, out: Any = None, keepdims: bool = False) -> Any:
    if not _is_supported_operand(a):
        error_message = 'Statistics require numeric scalars, ndarrays, quantities or points.'
        raise TypeError(error_message)
    values, unit = _operand_parts(a)
    result = Quantity(np.ptp(values, axis=axis, keepdims=keepdims), _delta_unit(unit))
    return _write_output(result, out)


def _cumulative(func: Any, a: Any, axis: Any = None, dtype: Any = None, out: Any = None) -> Any:
    _validate_dtype(dtype)
    values, unit = _linear_parts(a)
    if func in (np.cumprod, np.nancumprod):
        if unit.dimension != DIMENSIONLESS:
            error_message = 'Cumulative products support only dimensionless input.'
            raise DimensionError(error_message)
        values, unit = scale_magnitude(values, unit.factor), Unit()
    result = Quantity(func(values, axis=axis, dtype=dtype), unit)
    return _write_output(result, out)


def _covariance(
    func: Any,
    m: Any = _UNSET,
    y: Any = None,
    rowvar: bool = True,
    bias: Any = _UNSET,
    ddof: Any = _UNSET,
    fweights: Any = None,
    aweights: Any = None,
    *,
    dtype: Any = None,
    x: Any = _UNSET,
) -> Any:
    # corrcoef names its first parameter x; cov names it m.
    if x is not _UNSET:
        if m is not _UNSET or func is np.cov:
            error_message = 'Supply the first covariance input once.'
            raise TypeError(error_message)
        m = x
    if m is _UNSET:
        error_message = 'A covariance input is required.'
        raise TypeError(error_message)
    _validate_dtype(dtype)
    if not _is_supported_operand(m) or (y is not None and not _is_supported_operand(y)):
        error_message = 'Covariance requires numeric scalars, ndarrays, quantities or points.'
        raise TypeError(error_message)
    mv, mu = _operand_parts(m)
    unit = _delta_unit(mu)
    yv = None
    if y is not None:
        yv, yu = _operand_parts(y)
        if func is np.cov:
            delta = _delta_unit(yu)
            if not compatible(unit, delta):
                error_message = 'A joint covariance matrix requires compatible difference units.'
                raise DimensionError(error_message)
            yv = scale_magnitude(yv, delta.factor / unit.factor)
    options = {'rowvar': rowvar, 'dtype': dtype}
    if bias is not _UNSET:
        options['bias'] = bias
    if ddof is not _UNSET:
        options['ddof'] = _statistic_parameter(ddof, 'ddof') if ddof is not None else None
    if func is np.cov:
        options['fweights'] = (
            None if fweights is None else _statistic_parameter(fweights, 'fweights')
        )
        options['aweights'] = (
            None if aweights is None else _statistic_parameter(aweights, 'aweights')
        )
    elif fweights is not None or aweights is not None:
        error_message = 'corrcoef does not accept weights.'
        raise TypeError(error_message)
    result = _call_with_warnings(func, mv, yv, **options)
    return Quantity(result, unit**2 if func is np.cov else Unit())


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
    ddof: Any = _UNSET,
    keepdims: bool = False,
    *,
    where: Any = True,
    mean: Any = _UNSET,
    correction: Any = _UNSET,
) -> Any:
    _validate_dtype(dtype)
    _validate_where({'where': where})
    delta = q.unit.delta_unit if isinstance(q.unit, _Coordinate) else q.unit
    unit = delta**2 if func in (np.var, np.nanvar) else delta
    options = {}
    if ddof is not _UNSET:
        options['ddof'] = _statistic_parameter(ddof, 'ddof')
    if correction is not _UNSET:
        options['correction'] = _statistic_parameter(correction, 'correction')
    if mean is not _UNSET:
        options['mean'] = _selection_parts(q, mean)[1]
    result = _wrap(
        _call_with_warnings(
            func, q.value, axis=axis, dtype=dtype, keepdims=keepdims, where=where, **options
        ),
        unit,
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
        error_message = 'Affine points cannot be accumulated or reduced with arithmetic ufuncs.'
        raise UnitOperationError(error_message)
    if kwargs.keys() - {'where', 'dtype', 'casting', 'order', 'subok'}:
        return NotImplemented
    _validate_where(kwargs)
    if 'dtype' in kwargs and np.dtype(kwargs['dtype']).kind not in 'biuf':
        raise TypeError('Affine operations require real numeric output.')
    if 'where' in kwargs:
        kwargs = {**kwargs, 'out': (None,) * ufunc.nout}
    if any(isinstance(x, (_Coordinate, Unit)) for x in inputs):
        error_message = 'Affine unit objects cannot participate in this ufunc.'
        raise UnitOperationError(error_message)
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
    if ufunc in (np.positive, np.floor, np.ceil, np.trunc, np.rint):
        return _wrap(ufunc(q.value, **kwargs), q.unit)
    if ufunc in (np.isfinite, np.isnan, np.isinf):
        return ufunc(q.value, **kwargs)
    if ufunc in (np.minimum, np.maximum, np.fmin, np.fmax):
        a, b, unit = _selection_parts(*inputs)
        return _wrap(ufunc(a, b, **kwargs), unit)
    error_message = 'This ufunc is undefined for affine points; convert to physical units first.'
    raise UnitOperationError(error_message)


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


_ARRAY_FUNCTIONS = {
    np.fft.fftshift: partial(_array_transform, np.fft.fftshift),
    np.fft.ifftshift: partial(_array_transform, np.fft.ifftshift),
    np.reshape: _reshape,
    np.transpose: _transpose,
    np.squeeze: _squeeze,
    np.expand_dims: _expand_dims,
    np.moveaxis: _moveaxis,
    np.swapaxes: _swapaxes,
    np.rollaxis: _rollaxis,
    np.ravel: _ravel,
    np.matrix_transpose: partial(_array_transform, np.matrix_transpose),
    np.linalg.matrix_transpose: partial(_array_transform, np.linalg.matrix_transpose),
    np.broadcast_to: partial(_array_transform, np.broadcast_to),
    np.flip: partial(_array_transform, np.flip),
    np.fliplr: partial(_array_transform, np.fliplr),
    np.flipud: partial(_array_transform, np.flipud),
    np.rot90: partial(_array_transform, np.rot90),
    np.roll: partial(_array_transform, np.roll),
    np.repeat: partial(_array_transform, np.repeat),
    np.tile: partial(_array_transform, np.tile),
    np.copy: partial(_array_transform, np.copy),
    np.shape: partial(_array_metadata, np.shape),
    np.ndim: partial(_array_metadata, np.ndim),
    np.size: partial(_array_metadata, np.size),
    np.split: partial(_split_arrays, np.split),
    np.array_split: partial(_split_arrays, np.array_split),
    np.hsplit: partial(_split_arrays, np.hsplit),
    np.vsplit: partial(_split_arrays, np.vsplit),
    np.dsplit: partial(_split_arrays, np.dsplit),
    np.take: _take,
    np.take_along_axis: _take_along_axis,
    np.delete: _delete,
    np.diagonal: partial(_array_transform, np.diagonal),
    np.linalg.diagonal: partial(_array_transform, np.linalg.diagonal),
    np.diag: partial(_array_transform, np.diag),
    np.diagflat: partial(_array_transform, np.diagflat),
    np.tril: partial(_array_transform, np.tril),
    np.triu: partial(_array_transform, np.triu),
    np.sort: partial(_array_transform, np.sort),
    np.argsort: partial(_array_metadata, np.argsort),
    np.partition: partial(_partition, np.partition),
    np.argpartition: partial(_partition, np.argpartition),
    np.unique: partial(_unique, np.unique),
    np.unique_all: partial(_unique, np.unique_all),
    np.unique_counts: partial(_unique, np.unique_counts),
    np.unique_inverse: partial(_unique, np.unique_inverse),
    np.unique_values: partial(_unique, np.unique_values),
    np.nonzero: partial(_array_metadata, np.nonzero),
    np.flatnonzero: partial(_array_metadata, np.flatnonzero),
    np.argwhere: partial(_array_metadata, np.argwhere),
    np.count_nonzero: partial(_array_metadata, np.count_nonzero),
    np.zeros_like: partial(_array_transform, np.zeros_like),
    np.ones_like: partial(_array_transform, np.ones_like),
    np.empty_like: partial(_array_transform, np.empty_like),
    np.round: _round,
    np.around: _round,
    np.fix: _fix,
}

_QUANTITY_FUNCTIONS = {
    **_ARRAY_FUNCTIONS,
    np.linalg.trace: partial(_array_transform, np.linalg.trace),
    np.sort_complex: partial(_array_transform, np.sort_complex),
    np.real: partial(_array_transform, np.real),
    np.imag: partial(_array_transform, np.imag),
    np.real_if_close: partial(_array_transform, np.real_if_close),
    np.angle: _angle,
    np.unwrap: _unwrap,
    np.sinc: _sinc,
    np.argmin: partial(_argmin_max, np.argmin),
    np.argmax: partial(_argmin_max, np.argmax),
    np.nanargmin: partial(_argmin_max, np.nanargmin),
    np.nanargmax: partial(_argmin_max, np.nanargmax),
    np.median: partial(_median, np.median),
    np.nanmedian: partial(_median, np.nanmedian),
    np.cumsum: _cumsum,
    np.nancumsum: partial(_cumulative, np.nancumsum),
    np.cumprod: partial(_cumulative, np.cumprod),
    np.nancumprod: partial(_cumulative, np.nancumprod),
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
