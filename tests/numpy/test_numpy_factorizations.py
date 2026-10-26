"""Matrix decompositions retain units in reconstruction equations."""

from collections.abc import Callable
from fractions import Fraction
from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx.factor import Factor, pi

np = pytest.importorskip('numpy')


@pytest.mark.parametrize('mode', ['reduced', 'complete', 'r'])
@pytest.mark.parametrize('shape', [(3, 2), (2, 3), (0, 2)])
@pytest.mark.parametrize('dtype', [np.int64, np.float32, np.complex128])
@pytest.mark.parametrize('batch', [False, True])
def test_numpy_qr(mode: str, shape: tuple[int, int], dtype: Any, batch: bool) -> None:
    values = np.arange(1, 1 + np.prod(shape)).reshape(shape).astype(dtype)
    if dtype is np.complex128:
        values += 1j
    if batch:
        values = np.stack([values, values * 2])
    a = Quantity(values, Factor(2) * pi * Unit('cm'))
    a.value.flags.writeable = False
    result = np.linalg.qr(a=a, mode=mode)
    expected = np.linalg.qr(values, mode=mode)
    if mode == 'r':
        assert result.unit is a.unit and result.value.dtype == expected.dtype
        np.testing.assert_allclose(result.value, expected)
        return
    q, r = result
    assert type(result) is type(expected) and result.Q is q and result.R is r
    assert q.unit == Unit() and r.unit is a.unit
    for actual, native in zip(result, expected):
        assert actual.value.dtype == native.dtype
        np.testing.assert_allclose(actual.value, native)
    np.testing.assert_allclose((q @ r).value_in(a.unit), values, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize('upper', [False, True])
@pytest.mark.parametrize('dtype', [np.int64, np.float32, np.complex128])
@pytest.mark.parametrize('batch', [False, True])
def test_numpy_cholesky(upper: bool, dtype: Any, batch: bool) -> None:
    values = np.array([[4, 1], [1, 3]], dtype=dtype)
    if dtype is np.complex128:
        values += np.array([[0, -1j], [1j, 0]])
    if batch:
        values = np.stack([values, values * 2])
    a = Quantity(values, Factor(2) * pi * Unit('cm'))
    a.value.flags.writeable = False
    result = np.linalg.cholesky(a, upper=upper)
    expected = np.linalg.cholesky(values, upper=upper)
    assert result.unit == a.unit ** Fraction(1, 2)
    assert result.value.dtype == expected.dtype
    np.testing.assert_allclose(result.value, expected)
    adjoint = np.conjugate(np.swapaxes(result, -1, -2))
    reconstructed = adjoint @ result if upper else result @ adjoint
    np.testing.assert_allclose(reconstructed.value_in(a.unit), values, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize('full_matrices', [False, True])
@pytest.mark.parametrize('compute_uv', [False, True])
@pytest.mark.parametrize('hermitian', [False, True])
@pytest.mark.parametrize('dtype', [np.int64, np.float32, np.complex128])
@pytest.mark.parametrize('batch', [False, True])
def test_numpy_svd(
    full_matrices: bool, compute_uv: bool, hermitian: bool, dtype: Any, batch: bool
) -> None:
    values = np.array([[3, 1], [1, 2]], dtype=dtype)
    if dtype is np.complex128:
        values += np.array([[0, -1j], [1j, 0]])
    if not hermitian:
        values = np.concatenate([values, values[:1]])
    if batch:
        values = np.stack([values, values * 2])
    a = Quantity(values, Factor(2) * pi * Unit('V'))
    a.value.flags.writeable = False
    result = np.linalg.svd(a, full_matrices, compute_uv, hermitian)
    expected = np.linalg.svd(values, full_matrices, compute_uv, hermitian)
    if not compute_uv:
        assert result.unit is a.unit and result.value.dtype == expected.dtype
        np.testing.assert_allclose(result.value, expected)
        return
    u, s, vh = result
    assert type(result) is type(expected)
    assert result.U is u and result.S is s and result.Vh is vh
    assert u.unit == vh.unit == Unit() and s.unit is a.unit
    for actual, native in zip(result, expected):
        assert actual.value.dtype == native.dtype
        np.testing.assert_allclose(actual.value, native)
    k = min(values.shape[-2:])
    reconstructed = (u[..., :k] * s[..., None, :]) @ vh[..., :k, :]
    np.testing.assert_allclose(reconstructed.value_in(a.unit), values, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize('shape', [(2, 3), (2, 3, 2), (0, 2)])
def test_numpy_svdvals(shape: tuple[int, ...]) -> None:
    a = Quantity(np.arange(np.prod(shape), dtype=float).reshape(shape), Factor(2) * Unit('cm'))
    result = np.linalg.svdvals(a)
    assert result.unit is a.unit
    np.testing.assert_allclose(result.value, np.linalg.svd(a.value, compute_uv=False))
    np.testing.assert_allclose(result.value_in('m'), np.linalg.svdvals(a.to('m')).value, atol=1e-14)


@pytest.mark.parametrize(
    'operation', [np.linalg.eig, np.linalg.eigh, np.linalg.eigvals, np.linalg.eigvalsh]
)
@pytest.mark.parametrize('dtype', [np.int64, np.float32, np.complex128])
@pytest.mark.parametrize('batch', [False, True])
def test_numpy_eigen(operation: Callable[..., Any], dtype: Any, batch: bool) -> None:
    values = np.array([[0, -1], [1, 0]], dtype=dtype)
    if operation in (np.linalg.eigh, np.linalg.eigvalsh):
        values = np.array([[3, 1], [1, 2]], dtype=dtype)
        if dtype is np.complex128:
            values += np.array([[0, -1j], [1j, 0]])
    if batch:
        values = np.stack([values, values * 2])
    a = Quantity(values, Factor(2) * pi * Unit('Hz'))
    a.value.flags.writeable = False
    result = operation(a=a)
    expected = operation(values)
    if operation in (np.linalg.eigvals, np.linalg.eigvalsh):
        assert result.unit is a.unit and result.value.dtype == expected.dtype
        np.testing.assert_allclose(result.value, expected)
        return
    w, v = result
    assert type(result) is type(expected)
    assert result.eigenvalues is w and result.eigenvectors is v
    assert w.unit is a.unit and v.unit == Unit()
    for actual, native in zip(result, expected):
        assert actual.value.dtype == native.dtype
        np.testing.assert_allclose(actual.value, native)
    np.testing.assert_allclose(
        (a @ v).value_in(a.unit), (v * w[..., None, :]).value_in(a.unit), atol=2e-6
    )


@pytest.mark.parametrize('uplo', ['L', 'U'])
def test_numpy_eigh_triangle(uplo: str) -> None:
    a = Quantity(np.array([[3 + 2j, 5 + 1j], [1 - 2j, 4 - 3j]]), 'm')
    for operation in (np.linalg.eigh, np.linalg.eigvalsh):
        result = operation(a, UPLO=uplo)
        expected = operation(a.value, UPLO=uplo)
        if operation is np.linalg.eigh:
            for actual, native in zip(result, expected):
                np.testing.assert_allclose(actual.value, native)
        else:
            np.testing.assert_allclose(result.value, expected)


@pytest.mark.parametrize('mode', ['raw', 'economic', 'full', 'invalid'])
def test_numpy_qr_invalid_mode(mode: str) -> None:
    a = Quantity(np.array([[3.0, 1], [1, 2]]), 'm')
    saved = a.value.copy()
    with pytest.raises(ValueError, match='supports only'):
        np.linalg.qr(a, mode)
    np.testing.assert_array_equal(a.value, saved)


@pytest.mark.parametrize(
    'operation',
    [
        np.linalg.qr,
        np.linalg.cholesky,
        np.linalg.svd,
        np.linalg.svdvals,
        np.linalg.eig,
        np.linalg.eigh,
        np.linalg.eigvals,
        np.linalg.eigvalsh,
    ],
)
@pytest.mark.parametrize('constructor', [AffineQuantity, LogQuantity])
def test_numpy_factorization_invalid_point(operation: Callable[..., Any], constructor: Any) -> None:
    point = constructor(np.eye(2), 'degC' if constructor is AffineQuantity else 'dBm')
    with pytest.raises(TypeError, match='linear'):
        operation(point)


@pytest.mark.parametrize(
    'operation',
    [
        np.linalg.qr,
        np.linalg.cholesky,
        np.linalg.svd,
        np.linalg.svdvals,
        np.linalg.eig,
        np.linalg.eigh,
        np.linalg.eigvals,
        np.linalg.eigvalsh,
    ],
)
def test_numpy_factorization_invalid_shape(operation: Callable[..., Any]) -> None:
    a = Quantity(np.array([1.0, 2]), 'm')
    saved = a.value.copy()
    with pytest.raises(np.linalg.LinAlgError):
        operation(a)
    np.testing.assert_array_equal(a.value, saved)


def test_numpy_cholesky_invalid_definiteness() -> None:
    a = Quantity(np.array([[1.0, 2], [2, 1]]), 'm')
    saved = a.value.copy()
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(a)
    np.testing.assert_array_equal(a.value, saved)
