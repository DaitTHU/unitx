"""Norms of linear quantities, including zero-order nonzero counts."""

from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx.factor import Factor, pi

np = pytest.importorskip('numpy')


@pytest.mark.parametrize('ord', [0, 1, 2, -1, np.inf])
@pytest.mark.parametrize('axis', [None, 0, (1, 2)])
@pytest.mark.parametrize('keepdims', [False, True])
def test_numpy_vector_norm(ord: Any, axis: Any, keepdims: bool) -> None:
    q = Quantity(np.arange(1.0, 25).reshape(2, 3, 4) * (1 + 1j), Factor(2) * Unit('V'))
    result = np.linalg.vector_norm(q, ord=ord, axis=axis, keepdims=keepdims)
    assert result.unit == (Unit() if ord == 0 else q.unit)
    np.testing.assert_allclose(
        result.value, np.linalg.vector_norm(q.value, ord=ord, axis=axis, keepdims=keepdims)
    )


@pytest.mark.parametrize('ord', ['fro', 'nuc', 1, -1, 2, -2, np.inf])
@pytest.mark.parametrize('keepdims', [False, True])
def test_numpy_matrix_norm(ord: Any, keepdims: bool) -> None:
    q = Quantity(np.arange(1.0, 25).reshape(2, 3, 4), 'V')
    result = np.linalg.matrix_norm(q, ord=ord, keepdims=keepdims)
    assert result.unit == q.unit
    np.testing.assert_allclose(
        result.value, np.linalg.matrix_norm(q.value, ord=ord, keepdims=keepdims)
    )


@pytest.mark.parametrize('ord', [None, 0, 1, 2, 3, -1, -2, np.inf, -np.inf])
@pytest.mark.parametrize('axis', [None, 0, -1])
@pytest.mark.parametrize('keepdims', [False, True])
@pytest.mark.parametrize('dtype', [np.int64, np.float32, np.complex128])
def test_numpy_norm_vector(ord: Any, axis: Any, keepdims: bool, dtype: Any) -> None:
    values = np.array([0, 3, 4], dtype=dtype)
    if dtype is np.complex128:
        values += np.array([0, 4j, -3j])
    q = Quantity(values, Factor(2) * pi * Unit('V'))
    q.value.flags.writeable = False
    with np.errstate(divide='ignore'):
        expected = np.linalg.norm(values, ord=ord, axis=axis, keepdims=keepdims)
        result = np.linalg.norm(q, ord=ord, axis=axis, keepdims=keepdims)
    assert type(result) is Quantity
    assert result.unit == (Unit() if ord == 0 else q.unit)
    assert type(result.value) is type(expected) and result.value.dtype == expected.dtype
    np.testing.assert_allclose(result.value, expected)


@pytest.mark.parametrize('ord', [None, 'fro', 'nuc', 1, -1, 2, -2, np.inf, -np.inf])
@pytest.mark.parametrize('keepdims', [False, True])
def test_numpy_norm_matrix(ord: Any, keepdims: bool) -> None:
    q = Quantity(np.arange(1, 25, dtype=np.float64).reshape(2, 3, 4) * (1 + 2j), 'cm')
    result = np.linalg.norm(q, ord=ord, axis=(1, 2), keepdims=keepdims)
    assert result.unit == q.unit
    np.testing.assert_allclose(
        result.value, np.linalg.norm(q.value, ord=ord, axis=(1, 2), keepdims=keepdims)
    )


def test_numpy_norm_normalization() -> None:
    q = Quantity(np.array([3.0, 4.0]), Factor(2) * Unit('V'))
    norm = np.linalg.norm(q)
    assert norm.unit == q.unit and norm.value == 5
    normalized = q / norm
    assert normalized.unit == Unit()
    np.testing.assert_allclose(normalized.value, [0.6, 0.8])
    assert np.linalg.norm(normalized).value == pytest.approx(1)
    np.testing.assert_allclose(np.linalg.norm(q.to('mV')).value, norm.value_in('mV'))


@pytest.mark.parametrize('shape', [(0,), (2, 0), (2, 3, 4)])
def test_numpy_norm_default(shape: tuple[int, ...]) -> None:
    q = Quantity(np.zeros(shape), 'V')
    result = np.linalg.norm(x=q)
    assert result.unit == q.unit and result.value == np.linalg.norm(q.value)


@pytest.mark.parametrize('options', [{'ord': 'fro'}, {'axis': 2}, {'axis': (0, 0)}])
def test_numpy_norm_invalid(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        np.linalg.norm(Quantity(np.array([3.0, 4.0]), 'V'), **options)


@pytest.mark.parametrize(
    'point', [AffineQuantity(np.array([1.0]), 'degC'), LogQuantity(np.array([1.0]), 'dBm')]
)
def test_numpy_norm_invalid_point(point: Any) -> None:
    with pytest.raises(TypeError, match='linear'):
        np.linalg.norm(point)


def test_numpy_norm_invalid_order() -> None:
    with pytest.raises(TypeError, match='bare'):
        np.linalg.norm(Quantity(np.array([3.0, 4.0]), 'V'), ord=Quantity(2, ''))
