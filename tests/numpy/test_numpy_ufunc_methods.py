"""Ufunc method units, repeated-index updates and staged output buffers."""

from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx.exceptions import DimensionError
from unitx.factor import Factor, pi

np = pytest.importorskip('numpy')


@pytest.mark.parametrize(
    'ufunc', [np.add, np.subtract, np.remainder, np.fmod, np.minimum, np.maximum, np.fmin, np.fmax]
)
@pytest.mark.parametrize('method', ['reduce', 'accumulate', 'reduceat'])
def test_ufunc_methods_linear(ufunc: Any, method: str) -> None:
    q = Quantity(np.arange(1.0, 9.0).reshape(2, 4), Factor(2) * pi * Unit('cm'))
    args = ([0, 2],) if method == 'reduceat' else ()
    actual = getattr(ufunc, method)(q, *args, axis=1)
    expected = getattr(ufunc, method)(q.value, *args, axis=1)
    assert actual.unit == q.unit
    np.testing.assert_array_equal(actual.value, expected)
    target = Quantity(np.zeros(np.shape(expected)), 'm')
    assert getattr(ufunc, method)(q, *args, axis=1, out=target) is target
    np.testing.assert_allclose(target.value, actual.value_in('m'))


@pytest.mark.parametrize('ufunc', [np.minimum, np.maximum, np.fmin, np.fmax])
@pytest.mark.parametrize('family,unit', [(AffineQuantity, 'degC'), (LogQuantity, 'dBm')])
@pytest.mark.parametrize('method', ['reduce', 'accumulate', 'reduceat'])
def test_ufunc_methods_points(ufunc: Any, family: Any, unit: str, method: str) -> None:
    q = family(np.array([3.0, 1.0, 2.0, 4.0]), unit)
    args = ([0, 2],) if method == 'reduceat' else ()
    result = getattr(ufunc, method)(q, *args)
    assert type(result) is family and result.unit == q.unit
    np.testing.assert_array_equal(result.value, getattr(ufunc, method)(q.value, *args))
    with pytest.raises(TypeError):
        np.add.accumulate(q)


@pytest.mark.parametrize(
    'ufunc', [np.multiply, np.divide, np.floor_divide, np.logaddexp, np.logaddexp2]
)
@pytest.mark.parametrize('method', ['reduce', 'accumulate', 'reduceat'])
def test_ufunc_methods_dimensionless(ufunc: Any, method: str) -> None:
    q = Quantity(np.array([100.0, 200.0, 300.0, 400.0]), 'cm/m')
    args = ([0, 2],) if method == 'reduceat' else ()
    actual = getattr(ufunc, method)(q, *args)
    assert actual.unit == Unit()
    np.testing.assert_allclose(actual.value, getattr(ufunc, method)(q.value / 100, *args))
    if ufunc is not np.multiply or method != 'reduce':
        with pytest.raises(DimensionError):
            getattr(ufunc, method)(Quantity(q.value, 'm'), *args)


def test_ufunc_reduce_initial_and_mask() -> None:
    q = Quantity(np.array([1.0, 3.0, 2.0]), 'm')
    result = np.maximum.reduce(q, initial=Quantity(250, 'cm'), where=np.array([True, False, True]))
    assert result == Quantity(2.5, 'm')
    point = AffineQuantity(q.value, 'degC')
    result = np.minimum.reduce(point, initial=AffineQuantity(273.15, 'K'))
    np.testing.assert_allclose(result.value, 0, atol=1e-12)
    with pytest.raises(DimensionError):
        np.maximum.reduce(q, initial=0.0)


@pytest.mark.parametrize(
    'ufunc,unit', [(np.multiply, 'm*s'), (np.add, 'm'), (np.less, None), (np.hypot, 'm')]
)
def test_ufunc_outer(ufunc: Any, unit: str | None) -> None:
    a = Quantity(np.arange(3.0) + 1, 'm')
    b = Quantity(np.arange(2.0) + 1, 's' if ufunc is np.multiply else 'cm')
    result = ufunc.outer(a, b)
    raw_b = b.value if ufunc is np.multiply else b.value / 100
    expected = ufunc.outer(a.value, raw_b)
    if unit is None:
        np.testing.assert_array_equal(result, expected)
    else:
        assert result.unit == Unit(unit)
        np.testing.assert_allclose(result.value, expected)
        target = Quantity(np.zeros((3, 2)), result.unit)
        assert ufunc.outer(a, b, out=target) is target
        np.testing.assert_allclose(target.value, expected)


@pytest.mark.parametrize(
    'ufunc,rhs,expected',
    [
        (np.add, Quantity(50.0, 'cm'), [2.0, 2.0, 3.5]),
        (np.subtract, Quantity(50.0, 'cm'), [0.0, 2.0, 2.5]),
        (np.multiply, Quantity(200.0, 'cm/m'), [4.0, 2.0, 6.0]),
        (np.divide, 2.0, [0.25, 2.0, 1.5]),
        (np.maximum, Quantity(250.0, 'cm'), [2.5, 2.0, 3.0]),
    ],
)
def test_ufunc_at_repeated(ufunc: Any, rhs: Any, expected: list[float]) -> None:
    q = Quantity(np.array([1.0, 2.0, 3.0]), 'm')
    buffer = q.value
    assert ufunc.at(q, [0, 0, 2], rhs) is None
    assert q.value is buffer and q.unit == Unit('m')
    np.testing.assert_allclose(q.value, expected)


@pytest.mark.parametrize('family,unit', [(AffineQuantity, 'degC'), (LogQuantity, 'dBm')])
def test_ufunc_at_points(family: Any, unit: str) -> None:
    q = family(np.array([1.0, 2.0, 3.0]), unit)
    np.add.at(q, [0, 0], Quantity(2.0, q.unit.delta_unit))
    np.testing.assert_array_equal(q.value, [5, 2, 3])
    np.minimum.at(q, [0, 1], family(0.0, unit))
    np.testing.assert_array_equal(q.value, [0, 0, 3])
    with pytest.raises((DimensionError, TypeError)):
        np.subtract.at(q, [0], family(0.0, unit))
    np.testing.assert_array_equal(q.value, [0, 0, 3])


@pytest.mark.parametrize('ufunc', [np.negative, np.absolute, np.conjugate, np.floor])
def test_ufunc_at_unary(ufunc: Any) -> None:
    q = Quantity(np.array([-1.5, 2.5, -3.5]), 'm')
    expected = q.value.copy()
    ufunc.at(expected, [0, 0, 2])
    ufunc.at(q, [0, 0, 2])
    np.testing.assert_array_equal(q.value, expected)


def test_ufunc_at_failures_and_aliases() -> None:
    q = Quantity(np.array([1.0, 2.0, 3.0]), 'm')
    for operation in (
        lambda: np.add.at(q, [0, 9], Quantity(1.0, 'm')),
        lambda: np.add.at(q, [0, 1], Quantity(1.0, 's')),
        lambda: np.multiply.at(q, [0, 1], Quantity(1.0, 's')),
    ):
        with pytest.raises((DimensionError, IndexError)):
            operation()
        np.testing.assert_array_equal(q.value, [1, 2, 3])
    expected = q.value.copy()
    np.add.at(expected, [2, 1, 0], expected.copy())
    np.add.at(q, [2, 1, 0], q)
    np.testing.assert_array_equal(q.value, expected)
    q = Quantity(np.ones(3, dtype=int), 'm')
    with pytest.raises(TypeError):
        np.add.at(q, [0, 0], Quantity(1.0, 'cm'))
    np.testing.assert_array_equal(q.value, 1)


def test_ufunc_reduceat_invalid_preserves_output() -> None:
    q = Quantity(np.array([1.0, 2.0, 3.0]), 'm')
    out = Quantity(np.full(2, 7.0), 'm')
    with pytest.raises(IndexError):
        np.add.reduceat(q, [0, 9], out=out)
    np.testing.assert_array_equal(out.value, 7)
    with pytest.raises(DimensionError):
        np.add.reduceat(q, [0, 1], out=Quantity(np.zeros(2), 's'))


def test_ufunc_at_foreign_protocol() -> None:
    class Foreign:
        def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> str:
            return 'foreign dispatched'

    q = Quantity(np.ones(3), 'm')
    assert q.__array_ufunc__(np.add, 'at', q, [0], Foreign()) is NotImplemented
    np.add.at(q, [0], Foreign())
    np.testing.assert_array_equal(q.value, 1)
