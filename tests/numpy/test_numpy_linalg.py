"""Linear equations and matrix measures with one uniform unit per operand."""

from collections.abc import Callable
from fractions import Fraction
from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx.exceptions import DimensionError
from unitx.factor import Factor, pi

np = pytest.importorskip('numpy')


@pytest.mark.parametrize('dtype', [np.int64, np.float32, np.complex128])
@pytest.mark.parametrize('batch', [False, True])
def test_numpy_solve(dtype: Any, batch: bool) -> None:
    av = np.array([[3, 1], [1, 2]], dtype=dtype)
    bv = np.array([9, 8], dtype=dtype)
    if batch:
        av = np.stack([av, 2 * av])
        bv = np.stack([bv, bv])[..., None]
    a = Quantity(av, Factor(2) * pi * Unit('ohm'))
    b = Quantity(bv, 'mV')
    a.value.flags.writeable = b.value.flags.writeable = False
    result = np.linalg.solve(a=a, b=b)
    assert result.unit == b.unit / a.unit
    expected = np.linalg.solve(av, bv)
    assert result.value.dtype == expected.dtype
    np.testing.assert_allclose(result.value, expected)
    np.testing.assert_allclose((a @ result).value_in('mV'), bv)
    bare_a = np.linalg.solve(av, b)
    bare_b = np.linalg.solve(a, bv)
    assert bare_a.unit == b.unit and bare_b.unit == a.unit**-1
    np.testing.assert_allclose(bare_a.value, expected)
    np.testing.assert_allclose(bare_b.value, expected)


@pytest.mark.parametrize('axes', [None, (0, 1)])
def test_numpy_tensorsolve(axes: Any) -> None:
    a = Quantity(np.eye(6).reshape(2, 3, 2, 3), 'ohm')
    b = Quantity(np.arange(6.0).reshape(2, 3), 'V')
    result = np.linalg.tensorsolve(a, b, axes=axes)
    assert result.unit == b.unit / a.unit
    np.testing.assert_allclose(result.value, np.linalg.tensorsolve(a.value, b.value, axes=axes))
    np.testing.assert_allclose(np.tensordot(a, result, axes=2).value_in('V'), b.value)


@pytest.mark.parametrize('dtype', [np.float32, np.complex128])
@pytest.mark.parametrize('multiple', [False, True])
def test_numpy_lstsq(dtype: Any, multiple: bool) -> None:
    a = Quantity(np.array([[1, 0], [1, 1], [1, 2]], dtype=dtype), Factor(2) * Unit('m'))
    bv = np.array([1, 2, 4], dtype=dtype)
    if multiple:
        bv = np.stack([bv, 2 * bv], axis=-1)
    b = Quantity(bv, 'V')
    x, residuals, rank, singular = np.linalg.lstsq(a, b, rcond=-1)
    expected = np.linalg.lstsq(a.value, b.value, rcond=-1)
    assert x.unit == b.unit / a.unit and residuals.unit == b.unit**2
    assert singular.unit == a.unit and type(rank) is type(expected[2])
    assert rank == expected[2]
    for actual, raw in zip(
        (x.value, residuals.value, singular.value), (expected[0], expected[1], expected[3])
    ):
        np.testing.assert_allclose(actual, raw)
        assert actual.dtype == raw.dtype
    np.testing.assert_allclose(
        residuals.value, np.sum(np.abs(a.value @ x.value - bv) ** 2, axis=0), rtol=1e-5
    )


def test_numpy_lstsq_empty_residuals() -> None:
    a = Quantity(np.ones((2, 3)), 'm')
    b = Quantity(np.ones(2), 'V')
    x, residuals, rank, singular = np.linalg.lstsq(a, b)
    assert residuals.unit == b.unit**2 and residuals.shape == (0,)
    assert rank == 1 and singular.unit == a.unit
    np.testing.assert_allclose((a @ x).value, b.value)


@pytest.mark.parametrize('operation', [np.linalg.inv, np.linalg.pinv])
@pytest.mark.parametrize('batch', [False, True])
def test_numpy_inverse(operation: Callable[..., Any], batch: bool) -> None:
    values = np.array([[2 + 1j, 0], [0, 4 - 1j]])
    if batch:
        values = np.stack([values, 2 * values])
    a = Quantity(values, Factor(2) * pi * Unit('m'))
    result = operation(a)
    assert result.unit == a.unit**-1
    np.testing.assert_allclose(result.value, operation(values))
    np.testing.assert_allclose(
        (a @ result).value_in(''), np.broadcast_to(np.eye(2), values.shape), atol=1e-14
    )


@pytest.mark.parametrize('hermitian', [False, True])
@pytest.mark.parametrize('name', ['rcond', 'rtol'])
def test_numpy_pinv_cutoff(hermitian: bool, name: str) -> None:
    a = Quantity(np.array([[1.0, 0], [0, 0.001]]), 'm')
    cutoff = Quantity(1.0, Factor(Fraction(1, 100)) * Unit())
    result = np.linalg.pinv(a, hermitian=hermitian, **{name: cutoff})
    assert result.unit == a.unit**-1
    np.testing.assert_allclose(
        result.value, np.linalg.pinv(a.value, hermitian=hermitian, **{name: 0.01})
    )


@pytest.mark.parametrize(
    'options',
    [{}, {'rcond': None}, {'rtol': None}, {'rcond': -1}, {'rcond': [1e-8, 1e-4]}],
)
def test_numpy_pinv_native_cutoff(options: dict[str, Any]) -> None:
    a = Quantity(np.stack([np.diag([1.0, 5e-16]), np.diag([1.0, 0.001])]), 'm')
    result = np.linalg.pinv(a, **options)
    assert result.unit == a.unit**-1
    np.testing.assert_allclose(result.value, np.linalg.pinv(a.value, **options))


def test_numpy_lstsq_relative_cutoff() -> None:
    a = Quantity(np.diag([1.0, 0.001]), 'm')
    b = Quantity(np.ones(2), 'V')
    result = np.linalg.lstsq(a, b, Quantity(1.0, Factor(Fraction(1, 100)) * Unit()))
    assert result[2] == 1
    np.testing.assert_allclose(result[0].value, [1, 0])


def test_numpy_tensorinv() -> None:
    a = Quantity(np.eye(6).reshape(2, 3, 6), 'm')
    inverse = np.linalg.tensorinv(a, ind=2)
    assert inverse.unit == a.unit**-1
    np.testing.assert_allclose(inverse.value, np.linalg.tensorinv(a.value, ind=2))
    np.testing.assert_allclose(
        np.tensordot(a, inverse, axes=1).value_in(''), np.eye(6).reshape(2, 3, 2, 3)
    )


@pytest.mark.parametrize('n', [-2, -1, 0, 1, 2, 3, np.int64(2)])
def test_numpy_matrix_power(n: Any) -> None:
    a = Quantity(np.array([[2.0, 1], [0, 3]]), Factor(2) * pi * Unit('m'))
    result = np.linalg.matrix_power(a, n)
    assert result.unit == a.unit ** int(n)
    np.testing.assert_allclose(result.value, np.linalg.matrix_power(a.value, n))


@pytest.mark.parametrize('size', [0, 1, 2, 3])
def test_numpy_determinant(size: int) -> None:
    a = Quantity(np.broadcast_to(2 * np.eye(size), (2, size, size)), Factor(2) * Unit('cm'))
    result = np.linalg.det(a)
    assert result.unit == a.unit**size
    np.testing.assert_allclose(result.value, np.linalg.det(a.value))
    np.testing.assert_allclose(result.value_in(Unit('m') ** size), np.linalg.det(a.to('m').value))


@pytest.mark.parametrize('p', [None, 1, -1, 2, -2, 'fro', np.inf, -np.inf])
def test_numpy_condition(p: Any) -> None:
    a = Quantity(np.array([[2.0, 0], [0, 4]]), Factor(2) * pi * Unit('cm'))
    result = np.linalg.cond(a, p=p)
    assert result.unit == Unit()
    np.testing.assert_allclose(result.value, np.linalg.cond(a.value, p=p))
    np.testing.assert_allclose(np.linalg.cond(a.to('m'), p=p).value, result.value)


def test_numpy_matrix_rank_cutoffs() -> None:
    a = Quantity(np.diag([1.0, 0.005]), 'm')
    assert np.linalg.matrix_rank(A=a) == 2
    assert np.linalg.matrix_rank(a, tol=Quantity(1, 'cm')) == 1
    assert np.linalg.matrix_rank(a.to('cm'), tol=Quantity(1, 'cm')) == 1
    assert np.linalg.matrix_rank(a, tol=0.01) == 1
    relative = Quantity(1.0, Factor(Fraction(1, 100)) * Unit())
    assert np.linalg.matrix_rank(a, rtol=relative) == 1
    assert np.linalg.matrix_rank(a, hermitian=True, rtol=relative) == 1
    stacked = Quantity(np.stack([a.value, a.value * 2]), a.unit)
    ranks = np.linalg.matrix_rank(stacked, tol=Quantity(np.array([1.0, 0.1]), 'cm'))
    np.testing.assert_array_equal(ranks, [1, 2])
    assert type(ranks) is np.ndarray and ranks.dtype.kind in 'iu'


@pytest.mark.parametrize('operation', [np.linalg.solve, np.linalg.inv, np.linalg.det])
def test_numpy_linalg_invalid_shape(operation: Callable[..., Any]) -> None:
    a = Quantity(np.ones(3), 'm')
    with pytest.raises(np.linalg.LinAlgError):
        operation(a, a) if operation is np.linalg.solve else operation(a)


def test_numpy_linalg_invalid_singular() -> None:
    a = Quantity(np.ones((2, 2)), 'm')
    saved = a.value.copy()
    for operation in (np.linalg.inv, np.linalg.solve):
        with pytest.raises(np.linalg.LinAlgError):
            operation(a, a) if operation is np.linalg.solve else operation(a)
    np.testing.assert_array_equal(a.value, saved)


@pytest.mark.parametrize('constructor', [AffineQuantity, LogQuantity])
@pytest.mark.parametrize(
    'operation', [np.linalg.inv, np.linalg.det, np.linalg.cond, np.linalg.matrix_rank]
)
def test_numpy_linalg_invalid_point(constructor: Any, operation: Callable[..., Any]) -> None:
    point = constructor(np.eye(2), 'degC' if constructor is AffineQuantity else 'dBm')
    with pytest.raises(TypeError, match='linear'):
        operation(point)


def test_numpy_linalg_invalid_cutoff() -> None:
    a = Quantity(np.eye(2), 'm')
    with pytest.raises(DimensionError, match='rtol'):
        np.linalg.pinv(a, rtol=Quantity(1, 's'))
    with pytest.raises(DimensionError, match='tol'):
        np.linalg.matrix_rank(a, tol=Quantity(1, 's'))
    with pytest.raises(DimensionError, match='rtol'):
        np.linalg.pinv(a, rtol=Quantity(1, 'dB'))
    with pytest.raises(TypeError, match='real'):
        np.linalg.pinv(a, rcond=1j)


def test_numpy_linalg_invalid_cutoff_conflict() -> None:
    a = Quantity(np.eye(2), 'm')
    with pytest.raises(ValueError):
        np.linalg.pinv(a, rcond=0.01, rtol=0.02)
    with pytest.raises(ValueError):
        np.linalg.matrix_rank(a, tol=0.01, rtol=0.02)
