"""A finite set of unit-aware real mathematical functions.

Python int/float/Fraction/Decimal inputs use math and return float magnitudes.
Plain real NumPy arrays/scalars use NumPy and its broadcasting/dtype/error rules.
Bare numbers are dimensionless; all results are Quantities. Affine points,
logarithmic units, complex inputs and implicit list/tuple arrays are unsupported.
"""

from __future__ import annotations

import math as _math
from collections.abc import Callable
from fractions import Fraction
from typing import TypeAlias, cast, overload

from ._numerics import Array, Fractionable, Magnitude, is_numpy_object
from ._numerics.conversion import compatible
from .exceptions import DimensionError
from .factor import Factor, SymbolicConstant
from .quantity import Quantity
from .si import si

__all__ = [
    'sqrt',
    'cbrt',
    'sin',
    'cos',
    'tan',
    'asin',
    'acos',
    'atan',
    'exp',
    'expm1',
    'log',
    'log1p',
    'log2',
    'log10',
    'hypot',
    'atan2',
]

_ScalarInput: TypeAlias = Fractionable | Factor | SymbolicConstant | Quantity[Fractionable]
_ArrayInput: TypeAlias = Array | Quantity[Array]
_Input: TypeAlias = _ScalarInput | _ArrayInput
_Result: TypeAlias = Quantity[float] | Quantity[Array]


def _as_quantity(x: _Input) -> Quantity[Magnitude]:
    if isinstance(x, (Factor, SymbolicConstant)):
        x = x.value
    q = x if isinstance(x, Quantity) else Quantity(x, si.one)
    value = q.value
    if is_numpy_object(value):
        # Quantity construction has already checked ndarray type/dtype support.
        if getattr(cast(Array, value).dtype, 'kind') == 'c':
            raise TypeError('unitx.math requires real numeric inputs.')
    elif isinstance(value, bool) or not isinstance(value, Fractionable):
        raise TypeError('unitx.math requires real numeric inputs.')
    if q.dimension != q.dimension.physical:
        raise DimensionError('Convert logarithmic quantities to physical ratios first.')
    return q


def _unary(
    x: _Input,
    function: Callable[[float], float],
    numpy_name: str,
    *,
    power: Fraction | None = None,
    angle: bool = False,
) -> _Result:
    q = _as_quantity(x)
    if power is None and not q.unit.is_dimensionless():
        raise DimensionError('This function requires dimensionless input.')
    if is_numpy_object(q.value):
        from ._numerics.numpy import apply_ufunc

        return cast(Quantity[Array], apply_ufunc(numpy_name, q))
    unit = q.unit**power if power is not None else si.rad if angle else si.one
    value = q.value if power is not None else q.value_in(si.one)
    return Quantity(function(cast(float, value)), unit)


@overload
def sqrt(x: _ArrayInput, /) -> Quantity[Array]: ...


@overload
def sqrt(x: _ScalarInput, /) -> Quantity[float]: ...


def sqrt(x: _Input, /) -> _Result:
    """Return the square root, with an exact half power of the input unit."""
    return _unary(x, _math.sqrt, 'sqrt', power=Fraction(1, 2))


@overload
def cbrt(x: _ArrayInput, /) -> Quantity[Array]: ...


@overload
def cbrt(x: _ScalarInput, /) -> Quantity[float]: ...


def cbrt(x: _Input, /) -> _Result:
    """Return the real cube root, with an exact third power of the input unit."""
    return _unary(x, _math.cbrt, 'cbrt', power=Fraction(1, 3))


@overload
def sin(x: _ArrayInput, /) -> Quantity[Array]: ...


@overload
def sin(x: _ScalarInput, /) -> Quantity[float]: ...


def sin(x: _Input, /) -> _Result:
    """Return sine after converting dimensionless input to coherent values."""
    return _unary(x, _math.sin, 'sin')


@overload
def cos(x: _ArrayInput, /) -> Quantity[Array]: ...


@overload
def cos(x: _ScalarInput, /) -> Quantity[float]: ...


def cos(x: _Input, /) -> _Result:
    """Return cosine after converting dimensionless input to coherent values."""
    return _unary(x, _math.cos, 'cos')


@overload
def tan(x: _ArrayInput, /) -> Quantity[Array]: ...


@overload
def tan(x: _ScalarInput, /) -> Quantity[float]: ...


def tan(x: _Input, /) -> _Result:
    """Return tangent after converting dimensionless input to coherent values."""
    return _unary(x, _math.tan, 'tan')


@overload
def asin(x: _ArrayInput, /) -> Quantity[Array]: ...


@overload
def asin(x: _ScalarInput, /) -> Quantity[float]: ...


def asin(x: _Input, /) -> _Result:
    """Return inverse sine in radians."""
    return _unary(x, _math.asin, 'arcsin', angle=True)


@overload
def acos(x: _ArrayInput, /) -> Quantity[Array]: ...


@overload
def acos(x: _ScalarInput, /) -> Quantity[float]: ...


def acos(x: _Input, /) -> _Result:
    """Return inverse cosine in radians."""
    return _unary(x, _math.acos, 'arccos', angle=True)


@overload
def atan(x: _ArrayInput, /) -> Quantity[Array]: ...


@overload
def atan(x: _ScalarInput, /) -> Quantity[float]: ...


def atan(x: _Input, /) -> _Result:
    """Return inverse tangent in radians."""
    return _unary(x, _math.atan, 'arctan', angle=True)


@overload
def exp(x: _ArrayInput, /) -> Quantity[Array]: ...


@overload
def exp(x: _ScalarInput, /) -> Quantity[float]: ...


def exp(x: _Input, /) -> _Result:
    """Return the exponential of coherent dimensionless input."""
    return _unary(x, _math.exp, 'exp')


@overload
def expm1(x: _ArrayInput, /) -> Quantity[Array]: ...


@overload
def expm1(x: _ScalarInput, /) -> Quantity[float]: ...


def expm1(x: _Input, /) -> _Result:
    """Return exp(x) - 1, preserving accuracy for small coherent x."""
    return _unary(x, _math.expm1, 'expm1')


@overload
def log(x: _ArrayInput, /) -> Quantity[Array]: ...


@overload
def log(x: _ScalarInput, /) -> Quantity[float]: ...


def log(x: _Input, /) -> _Result:
    """Return the natural logarithm of coherent dimensionless input."""
    return _unary(x, _math.log, 'log')


@overload
def log1p(x: _ArrayInput, /) -> Quantity[Array]: ...


@overload
def log1p(x: _ScalarInput, /) -> Quantity[float]: ...


def log1p(x: _Input, /) -> _Result:
    """Return log(1 + x), preserving accuracy for small coherent x."""
    return _unary(x, _math.log1p, 'log1p')


@overload
def log2(x: _ArrayInput, /) -> Quantity[Array]: ...


@overload
def log2(x: _ScalarInput, /) -> Quantity[float]: ...


def log2(x: _Input, /) -> _Result:
    """Return the base-two logarithm of coherent dimensionless input."""
    return _unary(x, _math.log2, 'log2')


@overload
def log10(x: _ArrayInput, /) -> Quantity[Array]: ...


@overload
def log10(x: _ScalarInput, /) -> Quantity[float]: ...


def log10(x: _Input, /) -> _Result:
    """Return the base-ten logarithm of coherent dimensionless input."""
    return _unary(x, _math.log10, 'log10')


@overload
def hypot(x: _ArrayInput, /, *coordinates: _Input) -> Quantity[Array]: ...


@overload
def hypot(*coordinates: _ScalarInput) -> Quantity[float]: ...


@overload
def hypot(*coordinates: _Input) -> _Result: ...


def hypot(*coordinates: _Input) -> _Result:
    """Return the Euclidean norm in the first coordinate's unit.

    All coordinates must have compatible dimensions. With no coordinates,
    return dimensionless zero. Arrays broadcast via pairwise NumPy hypot calls.
    """
    if not coordinates:
        return Quantity(0.0, si.one)
    quantities = [_as_quantity(x) for x in coordinates]
    unit = quantities[0].unit
    if any(not compatible(q.unit, unit) for q in quantities):
        raise DimensionError('hypot requires compatible input dimensions.')
    if any(is_numpy_object(q.value) for q in quantities):
        from ._numerics.numpy import apply_ufunc

        result = Quantity(0, unit)
        for q in quantities:
            result = apply_ufunc('hypot', result, q)
        return cast(Quantity[Array], result)
    values = [float(cast(Fractionable, q.value_in(unit))) for q in quantities]
    return Quantity(_math.hypot(*values), unit)


@overload
def atan2(y: _ArrayInput, x: _Input, /) -> Quantity[Array]: ...


@overload
def atan2(y: _ScalarInput, x: _ArrayInput, /) -> Quantity[Array]: ...


@overload
def atan2(y: _ScalarInput, x: _ScalarInput, /) -> Quantity[float]: ...


def atan2(y: _Input, x: _Input, /) -> _Result:
    """Return atan2(y, x) in radians, requiring compatible input dimensions."""
    yq, xq = _as_quantity(y), _as_quantity(x)
    if not compatible(yq.unit, xq.unit):
        raise DimensionError('atan2 requires compatible input dimensions.')
    if is_numpy_object(yq.value) or is_numpy_object(xq.value):
        from ._numerics.numpy import apply_ufunc

        return cast(Quantity[Array], apply_ufunc('arctan2', yq, xq))
    a = float(cast(Fractionable, yq.value))
    b = float(cast(Fractionable, xq.value_in(yq.unit)))
    return Quantity(_math.atan2(a, b), si.rad)
