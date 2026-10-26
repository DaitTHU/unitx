"""Array dispersion, covariance and cumulative units."""

from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity, statistics
from unitx.exceptions import DimensionError
from unitx.factor import Factor, pi
from unitx.si import si

np = pytest.importorskip('numpy')


@pytest.mark.parametrize(
    'constructor,unit', [(Quantity, 'cm'), (AffineQuantity, 'degC'), (LogQuantity, 'dBm')]
)
@pytest.mark.parametrize('rowvar', [False, True])
def test_numpy_covariance(constructor: Any, unit: str, rowvar: bool) -> None:
    q = constructor(np.arange(12.0).reshape(3, 4), unit)
    result = np.cov(
        m=q,
        rowvar=rowvar,
        ddof=1,
        aweights=Quantity(np.arange(1, 1 + q.shape[1 if rowvar else 0]), 'cm/m'),
    )
    delta = q.unit if constructor is Quantity else q.unit.delta_unit
    assert result.unit == delta**2
    expected = np.cov(
        q.value, rowvar=rowvar, ddof=1, aweights=np.arange(1, 1 + q.shape[1 if rowvar else 0])
    )
    np.testing.assert_allclose(result.value, expected)


def test_numpy_covariance_joint() -> None:
    x = Quantity(np.array([1.0, 2, 4]), Factor(2) * pi * si.m)
    y = x.to('cm')
    result = np.cov(x, y)
    assert result.unit == x.unit**2
    np.testing.assert_allclose(result.value, np.cov(x.value, x.value))
    scalar = statistics.variance([1 * si.m, 2 * si.m, 4 * si.m])
    np.testing.assert_allclose(
        np.cov(Quantity(np.array([1.0, 2, 4]), 'm')).value, float(scalar.value)
    )
    with pytest.raises(DimensionError):
        np.cov(x, Quantity(y.value, 's'))


def test_numpy_correlation_independent_dimensions() -> None:
    x = Quantity(np.array([1.0, 2, 3]), 'm')
    y = AffineQuantity(np.array([20.0, 30, 35]), 'degC')
    result = np.corrcoef(x=x, y=y)
    assert result.unit == si.one
    np.testing.assert_allclose(result.value, np.corrcoef(x.value, y.value))


def test_numpy_covariance_complex() -> None:
    q = Quantity(np.array([[1 + 2j, 2 - 3j, 4 + 1j], [3j, 2, 1 + 1j]]), 'm')
    result = np.cov(q, dtype=np.complex128)
    assert result.unit == si.m**2 and result.value.dtype == np.complex128
    np.testing.assert_allclose(result.value, np.cov(q.value))


@pytest.mark.parametrize('operation', [np.std, np.var, np.nanstd, np.nanvar])
@pytest.mark.parametrize(
    'constructor,unit', [(Quantity, 'm'), (AffineQuantity, 'degC'), (LogQuantity, 'dBm')]
)
def test_numpy_dispersion_precomputed_mean(operation: Any, constructor: Any, unit: str) -> None:
    q = constructor(np.arange(6.0).reshape(2, 3), unit)
    avg = np.mean(q, axis=1, keepdims=True)
    if constructor is Quantity:
        avg = avg.to('cm')
    elif constructor is AffineQuantity:
        avg = avg.to('K')
    result = operation(q, axis=1, mean=avg, correction=1)
    expected = operation(
        q.value, axis=1, mean=np.mean(q.value, axis=1, keepdims=True), correction=1
    )
    delta = q.unit if constructor is Quantity else q.unit.delta_unit
    assert result.unit == (delta**2 if operation in (np.var, np.nanvar) else delta)
    np.testing.assert_allclose(result.value, expected)
    target = Quantity(np.full(2, 7.0), result.unit)
    with pytest.raises(DimensionError if constructor is Quantity else TypeError):
        operation(q, axis=1, mean=Quantity(np.ones((2, 1)), 's'), out=target)
    np.testing.assert_array_equal(target.value, 7)


@pytest.mark.parametrize('operation', [np.std, np.var, np.nanstd, np.nanvar])
def test_numpy_dispersion_invalid_correction(operation: Any) -> None:
    q = Quantity(np.arange(4.0), 'm')
    target = Quantity(np.full((), 7.0), 'm2' if operation in (np.var, np.nanvar) else 'm')
    with pytest.raises(ValueError):
        operation(q, ddof=1, correction=1, out=target)
    np.testing.assert_array_equal(target.value, 7)


@pytest.mark.parametrize(
    'constructor,unit', [(Quantity, 'm'), (AffineQuantity, 'degC'), (LogQuantity, 'dBm')]
)
def test_numpy_ptp(constructor: Any, unit: str) -> None:
    q = constructor(np.arange(6.0).reshape(2, 3), unit)
    result = np.ptp(q, axis=1, keepdims=True)
    assert type(result) is Quantity
    assert result.unit == (q.unit if constructor is Quantity else q.unit.delta_unit)
    np.testing.assert_array_equal(result.value, [[2], [2]])


@pytest.mark.parametrize('operation', [np.cumprod, np.nancumprod])
def test_numpy_cumulative_product(operation: Any) -> None:
    q = Quantity(np.array([1.0, np.nan, 2]), Factor(2) * si.one)
    result = operation(q)
    assert result.unit == si.one
    np.testing.assert_allclose(result.value, operation(q.value * 2), equal_nan=True)
    target = Quantity(np.zeros(3), 'cm/m')
    assert operation(q, out=target) is target
    np.testing.assert_allclose(target.value, result.value * 100, equal_nan=True)
    with pytest.raises(DimensionError):
        operation(Quantity(q.value, 'm'))


def test_numpy_nancumsum_alias() -> None:
    q = Quantity(np.array([1.0, np.nan, 3]), 'm')
    assert np.nancumsum(q, out=q) is q
    np.testing.assert_array_equal(q.value, [1, 1, 4])
    with pytest.raises(TypeError):
        np.nancumsum(AffineQuantity(q.value, 'degC'))
