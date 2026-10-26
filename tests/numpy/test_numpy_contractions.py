"""Unit products, native contraction semantics and staged writes."""

import warnings
from collections.abc import Callable
from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity
from unitx.exceptions import DimensionError
from unitx.factor import Factor, pi
from unitx.si import si

np = pytest.importorskip('numpy')


@pytest.mark.parametrize(
    'operation', [np.dot, np.inner, np.outer, np.vdot, np.kron, np.convolve, np.correlate]
)
def test_numpy_binary_product(operation: Callable[..., Any]) -> None:
    a = Quantity(np.array([1 + 2j, 3 - 1j]), Factor(2) * pi * si.m)
    b = Quantity(np.array([2 - 1j, 4 + 2j]), si.N)
    result = operation(a, b)
    assert result.unit == a.unit * b.unit
    np.testing.assert_array_equal(result.value, operation(a.value, b.value))
    bare = operation(a, b.value)
    assert bare.unit == a.unit
    np.testing.assert_array_equal(bare.value, result.value)


@pytest.mark.parametrize('axes', [0, 1, 2, ([0, 2], [2, 0])])
def test_numpy_tensordot(axes: Any) -> None:
    a = Quantity(np.arange(8.0).reshape(2, 2, 2), 'A')
    b = Quantity(np.arange(8.0).reshape(2, 2, 2), 's')
    result = np.tensordot(a=a, b=b, axes=axes)
    assert result.unit == a.unit * b.unit
    np.testing.assert_array_equal(result.value, np.tensordot(a.value, b.value, axes=axes))


@pytest.mark.parametrize('axis', [-1, 0])
def test_numpy_vecdot(axis: int) -> None:
    a = Quantity(np.array([[1 + 1j, 2], [3j, 4]]), 'm')
    b = Quantity(np.array([[2, 3j], [4, 5]]), 'N')
    expected = np.vecdot(a.value, b.value, axis=axis)
    for operation in (np.vecdot, np.linalg.vecdot):
        result = operation(a, b, axis=axis)
        assert result.unit == a.unit * b.unit
        np.testing.assert_array_equal(result.value, expected)
    target = Quantity(np.zeros(2, dtype=complex), 'J')
    assert np.vecdot(a, b, axis=axis, out=target) is target
    np.testing.assert_array_equal(target.value, expected)


@pytest.mark.parametrize('mode', ['valid', 'same', 'full'])
def test_numpy_discrete_product_mode(mode: str) -> None:
    a, b = Quantity(np.arange(3.0), 'V'), Quantity(np.array([2.0, 1.0]), 's')
    for operation in (np.convolve, np.correlate):
        result = operation(a, b, mode)
        assert result.unit == a.unit * b.unit
        np.testing.assert_array_equal(result.value, operation(a.value, b.value, mode))


def test_numpy_einsum_forms() -> None:
    a = Quantity(np.arange(6.0).reshape(2, 3), 'm')
    b = Quantity(np.arange(12.0).reshape(3, 4), 'N')
    expected = a.value @ b.value
    for result in (np.einsum('ij,jk->ik', a, b), np.einsum(a, [0, 1], b, [1, 2], [0, 2])):
        assert result.unit == a.unit * b.unit
        np.testing.assert_array_equal(result.value, expected)
    path, info = np.einsum_path('ij,jk->ik', a, b, optimize='optimal')
    assert path == np.einsum_path('ij,jk->ik', a.value, b.value, optimize='optimal')[0]
    assert isinstance(info, str)
    np.testing.assert_array_equal(np.einsum('ij,jk->ik', a, b, optimize=path).value, expected)
    assert np.einsum('ij,ij->', a, a).unit == a.unit**2
    view = np.einsum('ij->ji', a)
    assert view.unit == a.unit and np.shares_memory(view.value, a.value)
    view[0, 0] = Quantity(10, 'm')
    assert a.value[0, 0] == 10


def test_numpy_multi_dot() -> None:
    a = Quantity(np.arange(6.0).reshape(2, 3), 'V')
    b = Quantity(np.arange(12.0).reshape(3, 4), 'A')
    c = Quantity(np.arange(8.0).reshape(4, 2), 's')
    expected = np.linalg.multi_dot([a.value, b.value, c.value])
    result = np.linalg.multi_dot([a, b, c])
    assert result.unit == a.unit * b.unit * c.unit
    np.testing.assert_array_equal(result.value, expected)
    target = Quantity(np.zeros((2, 2)), 'J')
    assert np.linalg.multi_dot([a, b, c], out=target) is target
    np.testing.assert_array_equal(target.value, expected)


@pytest.mark.parametrize('operation', [np.dot, np.outer])
def test_numpy_product_output(operation: Callable[..., Any]) -> None:
    a = Quantity(np.array([1.0, 2.0]), 'cm')
    b = Quantity(np.array([3.0, 4.0]), 'N')
    expected = operation(a.value, b.value) / 100
    target = Quantity(np.zeros(np.shape(expected)), 'J')
    assert operation(a, b, target) is target
    np.testing.assert_allclose(target.value, expected)
    invalid = Quantity(np.full(np.shape(expected), 7), 'J')
    with pytest.raises(TypeError):
        operation(a, b, out=invalid)
    np.testing.assert_array_equal(invalid.value, 7)


def test_numpy_einsum_output_alias() -> None:
    q = Quantity(np.array([[1.0, 2.0], [3.0, 4.0]]), 'cm')
    assert np.einsum('ij->ji', q, out=q) is q
    np.testing.assert_array_equal(q.value, [[1, 3], [2, 4]])
    invalid = Quantity(np.full((2, 2), 7.0), 's')
    with pytest.raises(DimensionError):
        np.einsum('ij->ji', q, out=invalid)
    np.testing.assert_array_equal(invalid.value, 7)
    narrow = Quantity(np.full((2, 2), 7, dtype=np.float32), 'cm')
    with pytest.raises(TypeError):
        np.einsum('ij->ji', q, out=narrow)
    np.testing.assert_array_equal(narrow.value, 7)
    assert np.einsum('ij->ji', q, out=narrow, casting='same_kind') is narrow


def test_numpy_trace() -> None:
    q = Quantity(np.arange(24).reshape(2, 3, 4), Factor(2) * si.m)
    result = np.trace(q, offset=1, axis1=1, axis2=2, dtype=float)
    assert result.unit == q.unit
    np.testing.assert_array_equal(result.value, np.trace(q.value, 1, 1, 2, dtype=float))
    target = Quantity(np.zeros(2), 'cm')
    assert np.trace(q, 1, 1, 2, float, target) is target
    np.testing.assert_array_equal(target.value, result.value * 200)


def test_numpy_cross() -> None:
    a, b = Quantity(np.eye(3), 'm'), Quantity(np.arange(9.0).reshape(3, 3), 'N')
    result = np.cross(a, b, axis=0)
    assert result.unit == a.unit * b.unit
    np.testing.assert_array_equal(result.value, np.cross(a.value, b.value, axis=0))
    x, y = np.array([1.0, 2.0]), np.array([2.0, 1.0])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        try:
            expected = np.cross(x, y)
        except ValueError:
            with pytest.raises(ValueError):
                np.cross(Quantity(x, 'm'), Quantity(y, 'N'))
            return
    with pytest.warns(DeprecationWarning) as caught:
        result = np.cross(Quantity(x, 'm'), Quantity(y, 'N'))
    assert caught[0].filename == __file__
    np.testing.assert_array_equal(result.value, expected)


@pytest.mark.parametrize(
    'point', [AffineQuantity(np.ones(3), 'degC'), LogQuantity(np.ones(3), 'dBm')]
)
@pytest.mark.parametrize(
    'operation',
    [
        np.dot,
        np.inner,
        np.outer,
        np.vdot,
        np.tensordot,
        np.kron,
        np.cross,
        np.convolve,
        np.correlate,
        np.vecdot,
    ],
)
def test_numpy_product_invalid_point(point: Any, operation: Callable[..., Any]) -> None:
    with pytest.raises(TypeError):
        operation(point, np.ones(3))


def test_numpy_product_invalid_shape() -> None:
    target = Quantity(np.full(2, 7.0), 'm')
    with pytest.raises(ValueError):
        np.dot(Quantity(np.ones((2, 3)), 'm'), np.ones(4), out=target)
    np.testing.assert_array_equal(target.value, 7)


def test_numpy_einsum_invalid_dtype() -> None:
    with pytest.raises(TypeError):
        np.einsum('i->', Quantity(np.ones(3), 'm'), dtype=object)


def test_numpy_dot_invalid_operand() -> None:
    with pytest.raises(TypeError):
        np.dot(Quantity(np.ones(3), 'm'), [1, 2, 3])
