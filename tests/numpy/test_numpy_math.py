"""Stored-coordinate rounding, phase, and compatible Euclidean arithmetic."""

import warnings
from collections.abc import Callable
from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx.exceptions import DimensionError
from unitx.factor import Factor, pi
from unitx.si import si

np = pytest.importorskip('numpy')


@pytest.mark.parametrize(
    'operation',
    [np.sinh, np.cosh, np.tanh, np.arcsinh, np.arccosh, np.arctanh, np.exp2],
)
def test_numpy_dimensionless_math(operation: Callable[..., Any]) -> None:
    values = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    q = Quantity(values, Factor(2) * Unit())
    with np.errstate(invalid='ignore'):
        expected = operation(values * 2)
        result = operation(q)
    assert result.unit == si.one and result.value.dtype == expected.dtype
    np.testing.assert_allclose(result.value, expected, equal_nan=True)
    with pytest.raises(DimensionError):
        operation(Quantity(values, 'm'))
    with pytest.raises(DimensionError):
        operation(Quantity(values, 'dB'))


@pytest.mark.parametrize('operation', [np.logaddexp, np.logaddexp2])
def test_numpy_logaddexp(operation: Callable[..., Any]) -> None:
    q = Quantity(np.array([1.0, 2.0]), Factor(2) * Unit())
    r = Quantity(np.array([[0.0], [1.0]]), 'km/m')
    result = operation(q, r)
    assert result.unit == si.one
    np.testing.assert_allclose(result.value, operation(q.value * 2, r.value * 1000))
    with pytest.raises(DimensionError):
        operation(q, Quantity(1.0, 'm'))


@pytest.mark.parametrize(
    'operation', [np.floor, np.ceil, np.trunc, np.rint, np.fix, np.round, np.around]
)
@pytest.mark.filterwarnings('ignore:numpy.fix is deprecated:DeprecationWarning')
def test_numpy_round_stored_coordinate(array_sample: Any, operation: Callable[..., Any]) -> None:
    q = type(array_sample)(array_sample.value / 3 - 0.25, array_sample.unit)
    result = operation(q)
    assert type(result) is type(q) and result.unit is q.unit
    np.testing.assert_array_equal(result.value, operation(q.value))
    target = type(q)(np.zeros_like(q.value), q.unit)
    assert operation(q, out=target) is target
    np.testing.assert_array_equal(target.value, operation(q.value))


def test_numpy_round_decimals_and_output() -> None:
    q = Quantity(np.array([1.234, -2.345]), Factor(2) * si.m)
    result = np.round(q, 2)
    assert result.unit is q.unit
    np.testing.assert_array_equal(result.value, [1.23, -2.35])
    target = Quantity(np.full(2, -1.0), 'cm')
    assert np.round(q, 2, target) is target
    np.testing.assert_allclose(target.value, [246, -470])
    invalid = Quantity(np.full(2, 7), 'cm')
    with pytest.raises(TypeError):
        np.round(q, 2, invalid)
    np.testing.assert_array_equal(invalid.value, [7, 7])


@pytest.mark.parametrize('operation', [np.fmin, np.fmax])
def test_numpy_nan_selection(operation: Callable[..., Any]) -> None:
    a = Quantity(np.array([np.nan, 1.0, 3.0]), 'm')
    b = Quantity(np.array([200.0, np.nan, 200.0]), 'cm')
    result = operation(a, b)
    assert result.unit is a.unit
    np.testing.assert_array_equal(result.value, operation(a.value, b.value / 100))
    points = AffineQuantity(a.value, 'degC')
    result = operation(points, AffineQuantity(b.value, 'K'))
    assert result.unit is points.unit
    np.testing.assert_allclose(result.value, operation(a.value, b.value - 273.15))
    with pytest.raises(TypeError):
        operation(points, a)


@pytest.mark.parametrize('operation', [np.remainder, np.mod, np.fmod, np.floor_divide, np.divmod])
def test_numpy_euclidean_units(operation: Callable[..., Any]) -> None:
    a = Quantity(np.array([-250.0, 250.0]), 'cm')
    b = Quantity(np.array([[1.0], [-1.0]]), 'm')
    result = operation(a, b)
    expected = operation(a.value, b.value * 100)
    results = result if isinstance(result, tuple) else (result,)
    expectations = expected if isinstance(expected, tuple) else (expected,)
    for i, (q, value) in enumerate(zip(results, expectations)):
        unit = (
            si.one
            if operation is np.floor_divide or (operation is np.divmod and i == 0)
            else a.unit
        )
        assert q.unit == unit
        np.testing.assert_array_equal(q.value, value)
    with pytest.raises(DimensionError):
        operation(a, Quantity(1.0, 's'))
    with pytest.raises(DimensionError):
        operation(a, 1.0)


def test_numpy_divmod_output_staging() -> None:
    q = Quantity(np.array([2.5, -2.5]), 'm')
    divisor = Quantity(100.0, 'cm')
    quotient = Quantity(np.full(2, 9.0), Factor(2) * si.one)
    remainder = Quantity(np.full(2, 8.0), 'cm')
    result = np.divmod(q, divisor, out=(quotient, remainder), where=np.array([True, False]))
    assert result[0] is quotient and result[1] is remainder
    np.testing.assert_allclose(quotient.value, [1, 9])
    np.testing.assert_allclose(remainder.value, [50, 8])
    invalid = Quantity(np.full(2, 8), 'cm')
    with pytest.raises(TypeError):
        np.divmod(q, divisor, out=(quotient, invalid))
    np.testing.assert_allclose(quotient.value, [1, 9])
    np.testing.assert_array_equal(invalid.value, [8, 8])
    with pytest.raises(ValueError, match='overlap'):
        np.divmod(Quantity(np.array([2.5, 3.5]), ''), 1, out=(quotient, quotient))


def test_numpy_sign_and_copysign() -> None:
    q = Quantity(np.array([0.0, -2.0, 3.0]), pi * si.V)
    result = np.sign(q)
    assert result.unit == si.one
    np.testing.assert_array_equal(result.value, [0, -1, 1])
    complex_q = Quantity(np.array([1 + 1j, 0j]), 'm')
    np.testing.assert_array_equal(np.sign(complex_q).value, np.sign(complex_q.value))
    signed = np.copysign(q, Quantity(np.array([-1, 1, -1]), 's'))
    assert signed.unit is q.unit
    np.testing.assert_array_equal(signed.value, [-0.0, 2.0, -3.0])
    assert np.signbit(signed.value[0])


@pytest.mark.parametrize(
    'operation,target',
    [(np.degrees, si.deg), (np.rad2deg, si.deg), (np.radians, si.rad), (np.deg2rad, si.rad)],
)
@pytest.mark.parametrize('unit', [si.deg, si.rad, Factor(2) * si.rad, si.one])
def test_numpy_angular_conversion(operation: Callable[..., Any], target: Unit, unit: Unit) -> None:
    q = Quantity(np.array([0.0, 45.0, 180.0]), unit)
    result = operation(q)
    assert result.unit == target
    np.testing.assert_allclose(result.value, q.value_in(target))
    with pytest.raises(DimensionError):
        operation(Quantity(1.0, 'm'))


@pytest.mark.parametrize('deg', [True, False])
def test_numpy_angle(deg: bool) -> None:
    q = Quantity(np.array([-1 + 0j, 1j, -1j, complex(-0.0, -0.0)]), pi * si.V)
    result = np.angle(q, deg=deg)
    assert result.unit == (si.deg if deg else si.rad)
    np.testing.assert_array_equal(result.value, np.angle(q.value, deg=deg))


@pytest.mark.parametrize('tol', [0.01, 100, 1000])
def test_numpy_real_if_close(tol: float) -> None:
    q = Quantity(np.array([1 + 1e-14j, 2 + 1e-14j]), 'm')
    result = np.real_if_close(q, tol=tol)
    expected = np.real_if_close(q.value, tol=tol)
    assert result.unit is q.unit and result.value.dtype == expected.dtype
    np.testing.assert_array_equal(result.value, expected)
    assert np.shares_memory(result.value, q.value) == np.shares_memory(expected, q.value)


@pytest.mark.parametrize('unit', [si.deg, si.rad, Factor(2) * si.rad])
def test_numpy_unwrap_units(unit: Unit) -> None:
    phase = Quantity(np.array([[170.0, -170.0, -160.0], [10, 20, 30]]), 'deg').to(unit)
    result = np.unwrap(phase, axis=1)
    assert result.unit is phase.unit
    np.testing.assert_allclose(result.value_in(si.deg), [[170, 190, 200], [10, 20, 30]])
    explicit = np.unwrap(phase, 180 * si.deg, axis=1, period=360 * si.deg)
    np.testing.assert_allclose(explicit.value, result.value)
    native = np.unwrap(phase.value_in(si.rad), discont=3.5, axis=1, period=7)
    np.testing.assert_allclose(np.unwrap(phase, 3.5, axis=1, period=7).value_in(si.rad), native)


@pytest.mark.parametrize(
    'options',
    [
        dict(period=0),
        dict(period=-1),
        dict(period=np.inf),
        dict(period=np.array([1, 2])),
        dict(discont=-1),
        dict(discont=np.nan),
    ],
)
def test_numpy_unwrap_invalid_threshold(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        np.unwrap(Quantity(np.array([1.0, 2.0]), 'rad'), **options)


@pytest.mark.parametrize(
    'operation', [np.angle, np.unwrap, np.sign, np.copysign, np.divmod, np.real_if_close]
)
@pytest.mark.parametrize(
    'point', [AffineQuantity(np.array([1.0]), 'degC'), LogQuantity(np.array([1.0]), 'dBm')]
)
def test_numpy_math_invalid_point(operation: Callable[..., Any], point: Any) -> None:
    with pytest.raises(TypeError):
        operation(point, point) if operation in (np.copysign, np.divmod) else operation(point)


def test_numpy_math_masked_output() -> None:
    source = Quantity(np.array([1.0, 2.0]), Factor(2) * si.one)
    target = Quantity(np.full((2, 2), 7.0), 'km/m')
    assert np.sinh(source, out=target, where=np.array([True, False])) is target
    np.testing.assert_allclose(target.value, [[np.sinh(2) / 1000, 7], [np.sinh(2) / 1000, 7]])
    with np.errstate(invalid='raise'):
        with pytest.raises(FloatingPointError):
            np.arccosh(Quantity(np.array([0.0, 1.0]), ''), out=target)
    np.testing.assert_allclose(target.value, [[np.sinh(2) / 1000, 7], [np.sinh(2) / 1000, 7]])


def test_numpy_phase_frequency_workflow() -> None:
    time = Quantity(np.linspace(0, 10, 101), 's')
    signal = Quantity(2 * np.exp(1j * (1.2 * time.value + 0.3)), 'V')
    phase = np.unwrap(np.angle(signal))
    angular_frequency = np.gradient(phase, time)
    assert phase.unit == si.rad
    assert angular_frequency.unit == si.rad / si.s
    np.testing.assert_allclose(angular_frequency.value, 1.2, rtol=1e-12)
    np.testing.assert_allclose(np.abs(signal).value, 2)


def test_numpy_fix_deprecation() -> None:
    with warnings.catch_warnings(record=True) as native:
        warnings.simplefilter('always')
        np.fix(np.array([1.5]))
    if not native:
        pytest.skip('This NumPy version has not deprecated fix')
    target = Quantity(np.array([7.0]), 'm')
    with pytest.warns(DeprecationWarning, match='numpy.fix is deprecated') as caught:
        np.fix(Quantity(np.array([1.5]), 'm'))
    assert caught[0].filename == __file__
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        with pytest.raises(DeprecationWarning, match='numpy.fix is deprecated'):
            np.fix(Quantity(np.array([1.5]), 'm'), out=target)
    np.testing.assert_array_equal(target.value, [7])
