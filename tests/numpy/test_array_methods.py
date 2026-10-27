"""Direct array methods share physical semantics and native storage behavior."""

from __future__ import annotations

from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx._numerics import Array
from unitx.exceptions import DimensionError
from unitx.factor import Factor
from unitx.si import si

np = pytest.importorskip('numpy')


class _ArrayQuantity(Quantity[Array]):
    pass


def test_quantity_array_methods_subclass() -> None:
    q = _ArrayQuantity(np.arange(4.0), si.m)
    for result in (q.ravel(), q.copy(), q.sum(), q.mean(), q.std()):
        assert type(result) is Quantity


@pytest.mark.parametrize(
    'cls,unit', [(Quantity, si.m), (AffineQuantity, si.degC), (LogQuantity, si.dBm)]
)
@pytest.mark.parametrize('dtype', ['float32', 'int64'])
def test_array_methods_shape(cls: Any, unit: Any, dtype: str) -> None:
    values = np.arange(12, dtype=dtype)
    q = cls(values, unit)
    for result in (q.reshape(3, 4), q.reshape((3, -1)), q.reshape(np.int64(3), 4)):
        assert type(result) is cls and result.unit is unit
        assert result.value.dtype == values.dtype
        assert np.shares_memory(result.value, values)
        np.testing.assert_array_equal(result.value, values.reshape(3, 4))
    independent = q.reshape(3, 4, copy=True)
    assert not np.shares_memory(independent.value, values)
    q = cls(values.reshape(3, 4).T, unit)
    with pytest.raises(ValueError, match='copy'):
        q.reshape(-1, copy=False)
    flat = q.ravel(order='K')
    assert type(flat) is cls and flat.unit is unit
    assert np.shares_memory(flat.value, values)
    np.testing.assert_array_equal(flat.value, values)
    copied = q.copy()
    assert type(copied) is cls and copied.unit is unit
    assert copied.value.flags.c_contiguous
    assert not np.shares_memory(copied.value, q.value)
    assert q.copy(order='F').value.flags.f_contiguous
    np.testing.assert_array_equal(copied.value, q.value)


@pytest.mark.parametrize(
    'value', [np.float32(2), np.array(2, dtype=np.float32), np.array([], dtype=np.complex64)]
)
def test_quantity_array_methods_scalar(value: Any) -> None:
    q = Quantity(value, Factor(2) * si.cm)
    for result in (q.reshape(-1), q.ravel(), q.copy()):
        assert isinstance(result.value, np.ndarray)
        assert result.value.dtype == value.dtype
        assert result.unit is q.unit
    if value.size == 1:
        # Native reshape of a NumPy scalar to () can remain a NumPy scalar.
        result = q.reshape(())
        assert type(result.value) is type(np.reshape(value, ()))
        assert result.value.dtype == value.dtype


@pytest.mark.parametrize('method', ['reshape', 'ravel', 'copy', 'sum', 'mean', 'std'])
def test_quantity_array_methods_scalar_invalid(method: str) -> None:
    q = Quantity(3, si.m)
    with pytest.raises(AttributeError, match='NumPy magnitude'):
        getattr(q, method)(-1) if method == 'reshape' else getattr(q, method)()


@pytest.mark.parametrize('method', ['sum', 'mean', 'std'])
@pytest.mark.parametrize('dtype', ['int16', 'float32', 'complex64'])
def test_quantity_array_methods_reduction(method: str, dtype: str) -> None:
    values = np.arange(24, dtype=dtype).reshape(3, 8)[:, ::2]
    q = Quantity(values, Factor(3) * si.cm)
    mask = np.array([True, True, False, True])
    actual = getattr(q, method)(axis=1, where=mask, keepdims=True)
    expected = getattr(np, method)(values, axis=1, where=mask, keepdims=True)
    assert isinstance(actual, Quantity) and actual.unit is q.unit
    assert type(actual.value) is type(expected) and actual.value.dtype == expected.dtype
    np.testing.assert_allclose(actual.value, expected)
    scalar = getattr(q, method)()
    expected_scalar = getattr(np, method)(values)
    assert type(scalar.value) is type(expected_scalar)
    assert scalar.value.dtype == expected_scalar.dtype
    np.testing.assert_allclose(scalar.value, expected_scalar)


def test_quantity_sum_initial() -> None:
    q = Quantity(np.array([1, 2]), si.m)
    assert q.sum(initial=100 * si.cm) == 4 * si.m
    assert q.sum(initial=0) == 3 * si.m


def test_quantity_sum_initial_invalid() -> None:
    q = Quantity(np.array([1, 2]), si.m)
    with pytest.raises(DimensionError, match='initial'):
        q.sum(initial=0 * si.s)


@pytest.mark.parametrize(
    'cls,unit,delta',
    [(Quantity, si.m, si.m), (AffineQuantity, si.degC, si.K), (LogQuantity, si.dBm, si.dB)],
)
def test_array_methods_mean_std(cls: Any, unit: Any, delta: Unit) -> None:
    q = cls(np.array([[10.0, 20, 30], [20, 30, 40]]), unit)
    center = q.mean(axis=1, keepdims=True)
    assert type(center) is cls and center.unit is unit
    spread = q.std(axis=1, mean=center, correction=1)
    assert isinstance(spread, Quantity) and spread.unit == delta
    np.testing.assert_allclose(center.value, [[20], [30]])
    np.testing.assert_allclose(spread.value, [10, 10])
    target = Quantity(np.zeros(2, dtype=np.float32), delta)
    assert q.std(axis=1, ddof=1, out=target) is target
    np.testing.assert_allclose(target.value, [10, 10])
    if cls is not Quantity:
        assert not hasattr(q, 'sum')
        point_target = cls(np.zeros(2), si.K if cls is AffineQuantity else si.dBW)
        assert q.mean(axis=1, out=point_target) is point_target
        np.testing.assert_allclose(point_target.value, center.to(point_target.unit).value.ravel())


@pytest.mark.parametrize('method', ['sum', 'mean', 'std'])
def test_quantity_array_methods_out(method: str) -> None:
    q = Quantity(np.array([[1.0, 2], [3, 4]]), si.m)
    target = Quantity(q.value[:, 0], si.cm)
    expected = getattr(np, method)(q.value.copy(), axis=1) * 100
    assert getattr(q, method)(1, None, target) is target
    np.testing.assert_allclose(target.value, expected)


def _assert_out_failure(method: str, target: Quantity[Array], error: type[Exception]) -> None:
    q = Quantity(np.array([[1.0, 2], [3, 4]]), si.m)
    before = q.value.copy()
    saved = target.value.copy()
    with pytest.raises(error):
        getattr(q, method)(axis=1, out=target)
    np.testing.assert_array_equal(target.value, saved)
    np.testing.assert_array_equal(q.value, before)


@pytest.mark.parametrize('method', ['sum', 'mean', 'std'])
def test_quantity_array_methods_out_unit_invalid(method: str) -> None:
    _assert_out_failure(method, Quantity(np.zeros(2), si.s), DimensionError)


@pytest.mark.parametrize('method', ['sum', 'mean', 'std'])
def test_quantity_array_methods_out_shape_invalid(method: str) -> None:
    _assert_out_failure(method, Quantity(np.zeros(3), si.m), ValueError)


@pytest.mark.parametrize('method', ['sum', 'mean', 'std'])
def test_quantity_array_methods_out_dtype_invalid(method: str) -> None:
    _assert_out_failure(method, Quantity(np.ones(2, dtype=int), si.cm), TypeError)


def test_quantity_std_invalid() -> None:
    q = Quantity(np.arange(4.0), si.m)
    out = Quantity(np.full((), -1.0), si.m)
    with pytest.raises(ValueError, match='ddof.*correction'):
        q.std(ddof=1, correction=1, out=out)
    assert out.value == -1


def test_quantity_std_mean_invalid() -> None:
    q = Quantity(np.arange(4.0), si.m)
    out = Quantity(np.full((), -1.0), si.m)
    with pytest.raises(DimensionError):
        q.std(mean=Quantity(np.zeros(1), si.s), out=out)
    assert out.value == -1


def test_quantity_trapezoid_dx() -> None:
    # Only dx carries units: NumPy's own arithmetic reaches the direct sum method.
    result = np.trapezoid(np.array([1.0, 2, 3]), dx=2 * si.s)
    assert isinstance(result, Quantity) and result == 8 * si.s
