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
    for result in (q.ravel(), q.copy(), q.transpose(), q.squeeze(), q.sum(), q.mean(), q.std()):
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


@pytest.mark.parametrize(
    'method',
    ['reshape', 'ravel', 'copy', 'transpose', 'squeeze', 'sum', 'mean', 'std', 'var', 'min', 'max'],
)
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


@pytest.mark.parametrize(
    'cls,unit', [(Quantity, Factor(3) * si.cm), (AffineQuantity, si.degC), (LogQuantity, si.dBm)]
)
@pytest.mark.parametrize('dtype', ['float32', 'int64'])
def test_array_methods_axes(cls: Any, unit: Any, dtype: str) -> None:
    values = np.arange(12, dtype=dtype).reshape(2, 1, 6)[:, :, ::2]
    q = cls(values, unit)
    for result, expected in (
        (q.transpose(), np.transpose(values)),
        (q.transpose(None), np.transpose(values)),
        (q.transpose((2, 0, 1)), np.transpose(values, (2, 0, 1))),
        (q.transpose(2, 0, 1), np.transpose(values, (2, 0, 1))),
        (q.squeeze(), np.squeeze(values)),
        (q.squeeze(axis=1), np.squeeze(values, axis=1)),
        (q.transpose(2, 0, 1).squeeze(-1), np.squeeze(values.transpose(2, 0, 1), -1)),
    ):
        assert type(result) is cls and result.unit is q.unit
        assert result.value.dtype == values.dtype
        assert np.shares_memory(result.value, values)
        np.testing.assert_array_equal(result.value, expected)


@pytest.mark.parametrize('value', [np.float32(2), np.array(2, dtype=np.float32)])
@pytest.mark.parametrize(
    'cls,unit', [(Quantity, si.m), (AffineQuantity, si.degC), (LogQuantity, si.dBm)]
)
def test_array_methods_axes_scalar(cls: Any, unit: Any, value: Any) -> None:
    q = cls(value, unit)
    for result in (q.transpose(), q.transpose(()), q.squeeze()):
        assert type(result) is cls and result.unit is q.unit
        assert type(result.value) is type(value)
        assert result.value.dtype == value.dtype
        np.testing.assert_array_equal(result.value, value)


@pytest.mark.parametrize(
    'method,args,error',
    [
        ('transpose', (1, 1, 0), ValueError),
        ('transpose', (0, 1), ValueError),
        ('transpose', ((1, 0), 2), TypeError),
        ('squeeze', (0,), ValueError),
        ('squeeze', (3,), IndexError),
    ],
)
def test_quantity_array_methods_axes_invalid(
    method: str, args: tuple[Any, ...], error: Any
) -> None:
    values = np.arange(6.0).reshape(2, 1, 3)
    saved = values.copy()
    q = Quantity(values, si.m)
    with pytest.raises(error):
        getattr(q, method)(*args)
    np.testing.assert_array_equal(values, saved)


@pytest.mark.parametrize('method', ['min', 'max'])
@pytest.mark.parametrize(
    'cls,unit,target',
    [(Quantity, si.m, si.cm), (AffineQuantity, si.degC, si.K), (LogQuantity, si.dBW, si.dBm)],
)
def test_array_methods_extrema(method: str, cls: Any, unit: Any, target: Any) -> None:
    values = np.arange(6.0).reshape(2, 3)[:, ::-1]
    q = cls(values, unit)
    initial = cls(100.0 if method == 'min' else -100.0, unit).to(target)
    mask = np.array([True, False, True])
    expected = getattr(np, method)(values, axis=1, where=mask, initial=initial.value_in(unit))
    result = getattr(q, method)(axis=1, initial=initial, where=mask)
    assert type(result) is cls and result.unit == q.unit
    np.testing.assert_array_equal(result.value, expected)
    # Overlapping output must be staged and converted into its fixed coordinate.
    out = cls(values[:, 0], target)
    assert getattr(q, method)(1, out, initial=initial, where=mask) is out
    np.testing.assert_allclose(out.value, cls(expected, unit).value_in(target))
    empty = cls(np.empty(0), unit)
    assert getattr(empty, method)(initial=initial) == initial
    with pytest.raises(ValueError, match='zero-size'):
        getattr(empty, method)()


@pytest.mark.parametrize('method', ['min', 'max'])
@pytest.mark.parametrize('initial', [0, None])
def test_quantity_extrema_initial_invalid(method: str, initial: Any) -> None:
    q = Quantity(np.arange(3.0), si.m)
    out = Quantity(np.array(-1.0), si.cm)
    with pytest.raises(TypeError, match='initial must carry'):
        getattr(q, method)(initial=initial, out=out)
    assert out.value == -1


@pytest.mark.parametrize('method', ['min', 'max', 'var'])
@pytest.mark.parametrize('value', [np.float32(2), np.array(2, dtype=np.float32)])
def test_quantity_reductions_scalar(method: str, value: Any) -> None:
    q = Quantity(value, si.m)
    result = getattr(q, method)()
    assert result.unit == (si.m**2 if method == 'var' else si.m)
    assert result.value == getattr(np, method)(value)
    assert result.value.dtype == np.dtype('float32')


@pytest.mark.parametrize(
    'cls,unit', [(Quantity, si.cm), (AffineQuantity, si.degC), (LogQuantity, si.dBm)]
)
def test_array_methods_var(cls: Any, unit: Any) -> None:
    q = cls(np.array([[1.0, 2, 5], [3, 4, 7]]), unit)
    delta = q.unit if cls is Quantity else q.unit.delta_unit
    result = q.var(axis=1, correction=1, mean=q.mean(axis=1, keepdims=True))
    assert type(result) is Quantity and result.unit == delta**2
    np.testing.assert_allclose(result.value, [13 / 3, 13 / 3])
    out = Quantity(q.value[:, 0], Factor(100) * delta**2)
    assert q.var(axis=1, out=out, ddof=1) is out
    np.testing.assert_allclose(out.value, np.full(2, 13 / 300))


def test_quantity_var_complex() -> None:
    q = Quantity(np.array([1 + 1j, 3 + 5j], dtype=np.complex64), si.m)
    result = q.var()
    assert result.value.dtype == np.dtype('float32')
    assert result == 5 * si.m**2


def test_quantity_var_invalid() -> None:
    out = Quantity(np.array(-1.0), si.m**2)
    with pytest.raises(ValueError, match='ddof.*correction'):
        Quantity(np.arange(4.0), si.m).var(out=out, ddof=1, correction=1)
    assert out.value == -1


def test_quantity_var_mean_invalid() -> None:
    out = Quantity(np.array(-1.0), si.m**2)
    with pytest.raises(DimensionError):
        Quantity(np.arange(4.0), si.m).var(out=out, mean=1 * si.s)
    assert out.value == -1


@pytest.mark.parametrize('method', ['min', 'max'])
def test_quantity_extrema_initial_unit_invalid(method: str) -> None:
    out = Quantity(np.array(-1.0), si.cm)
    with pytest.raises(DimensionError):
        getattr(Quantity(np.arange(3.0), si.m), method)(initial=1 * si.s, out=out)
    assert out.value == -1


@pytest.mark.parametrize('method', ['min', 'max', 'var'])
def test_quantity_reductions_out_invalid(method: str) -> None:
    _assert_out_failure(method, Quantity(np.zeros(2), si.s), DimensionError)


@pytest.mark.parametrize('method', ['min', 'max', 'var'])
def test_quantity_reductions_out_shape_invalid(method: str) -> None:
    unit = si.m**2 if method == 'var' else si.m
    _assert_out_failure(method, Quantity(np.zeros(3), unit), ValueError)


@pytest.mark.parametrize('method', ['min', 'max', 'var'])
def test_quantity_reductions_out_dtype_invalid(method: str) -> None:
    unit = si.m**2 if method == 'var' else si.m
    _assert_out_failure(method, Quantity(np.ones(2, dtype=int), unit), TypeError)
