from __future__ import annotations

import math as stdmath
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from unitx import AffineQuantity, Quantity, math, time
from unitx.exceptions import DimensionError
from unitx.unit import Unit

if TYPE_CHECKING:
    from numpy.typing import NDArray


np = pytest.importorskip('numpy')


@pytest.mark.parametrize('numpy_first', [True, False])
def test_math_numpy_import_order(numpy_first: bool) -> None:
    imports = 'import numpy as np\nfrom unitx import math, Quantity'
    if not numpy_first:
        imports = 'from unitx import math, Quantity\nimport numpy as np'
    code = (
        imports
        + """
q = math.sqrt(Quantity(np.array([4.0]), 'm2'))
assert q.value[0] == 2
assert q.unit.expression() == 'm'
"""
    )
    subprocess.run([sys.executable, '-c', code], check=True)


def test_math_numpy_roots() -> None:
    values = np.arange(16.0).reshape(4, 4)[:, ::2]
    q = Quantity(values, 'cm2')
    result = math.sqrt(q)
    assert result.unit == Unit('cm')
    np.testing.assert_array_equal(result.value, np.sqrt(values))
    np.testing.assert_array_equal(q.value, values)
    assert math.cbrt(Quantity(np.array([-8.0]), 'm3')).unit == Unit('m')
    assert type(math.sqrt(Quantity(np.array(4.0), 'm2')).value) is np.float64
    assert math.sqrt(Quantity(np.empty(0), 'm2')).value.size == 0


def test_math_numpy_angles() -> None:
    result = math.sin(Quantity(np.array([0.0, 90]), 'deg'))
    np.testing.assert_allclose(result.value, [0, 1])
    for function, native in [
        (math.asin, np.arcsin),
        (math.acos, np.arccos),
        (math.atan, np.arctan),
    ]:
        result = function(Quantity(np.array([0.0, 0.0005]), 'km/m'))
        assert result.unit == Unit('rad')
        np.testing.assert_allclose(result.value, native([0.0, 0.5]))


def test_math_numpy_broadcast() -> None:
    y = Quantity(np.array([[3.0], [5.0]]), 'm')
    x = Quantity(np.array([400.0, 1200]), 'cm')
    result = math.hypot(y, x)
    assert result.unit == Unit('m')
    np.testing.assert_allclose(result.value, np.hypot(y.value, x.value / 100))
    angle = math.atan2(y, x)
    assert angle.unit == Unit('rad')
    np.testing.assert_allclose(angle.value, np.arctan2(y.value, x.value / 100))
    np.testing.assert_allclose(math.hypot(y).value, y.value)
    assert math.hypot(1, np.array([1.0])).value[0] == pytest.approx(stdmath.sqrt(2))
    np.testing.assert_allclose(
        math.hypot(y, x, x).value, np.hypot(np.hypot(y.value, x.value / 100), x.value / 100)
    )


def test_math_numpy_error_domains() -> None:
    with np.errstate(invalid='raise'):
        with pytest.raises(FloatingPointError):
            math.sqrt(np.array([-1.0]))
    for value in [np.array([1j]), np.complex128(1), np.ma.array([1.0])]:
        with pytest.raises(TypeError):
            math.sqrt(value)
    with pytest.raises(DimensionError):
        math.sin(Quantity(np.array([1.0]), 'm'))
    with pytest.raises(DimensionError):
        math.hypot(Quantity(np.array([1.0]), 'm'), 1)
    with pytest.raises(TypeError):
        math.sqrt(AffineQuantity(np.array([1.0]), 'K'))


@pytest.mark.parametrize('value', [np.array([1.0]), np.array(1.0), np.float64(1.0)])
def test_time_numpy_invalid(value: NDArray[np.generic] | np.float64) -> None:
    for function in (time.sleep, time.to_timedelta):
        with pytest.raises(TypeError, match='magnitude'):
            function(Quantity(value, 's'))


def test_math_numpy_ufunc_outputs() -> None:
    q = Quantity(np.array([0.0, 1]), '')
    out = Quantity(np.array([7.0, 7]), 'deg')
    result = q.__array_ufunc__(np.arcsin, '__call__', q, out=(out,), where=np.array([False, True]))
    assert result is out
    np.testing.assert_allclose(out.value, [7, 90])
    y, x = Quantity(np.array([3.0]), 'm'), Quantity(np.array([400.0]), 'cm')
    target = Quantity(np.zeros(1), 'cm')
    assert y.__array_ufunc__(np.hypot, '__call__', y, x, out=(target,)) is target
    np.testing.assert_allclose(target.value, [500])
    angle = Quantity(np.zeros(1), 'deg')
    assert y.__array_ufunc__(np.arctan2, '__call__', y, x, out=(angle,)) is angle
    np.testing.assert_allclose(angle.value, [stdmath.degrees(stdmath.atan2(3, 4))])
    with pytest.raises(DimensionError):
        y.__array_ufunc__(np.arctan2, '__call__', y, x, out=(target,))
    np.testing.assert_allclose(target.value, [500])


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex64', 'complex128'])
@pytest.mark.parametrize('shape', [(), (3,)])
def test_math_numpy_exp_conj(dtype: str, shape: tuple[int, ...]) -> None:
    value = np.full(shape, 0.25, dtype=dtype)
    if value.dtype.kind == 'c':
        value += 0.5j
    q = Quantity(value, 'percent')
    exponential = math.exp(q)
    expected = np.exp(q.n)
    assert type(exponential.value) is type(expected)
    assert exponential.value.dtype == expected.dtype
    np.testing.assert_array_equal(exponential.value, expected)
    conjugate = math.conj(q)
    expected_conjugate = np.conj(value)
    assert conjugate.unit is q.unit
    assert type(conjugate.value) is type(expected_conjugate)
    assert conjugate.value.dtype == expected_conjugate.dtype
    np.testing.assert_array_equal(conjugate.value, expected_conjugate)
