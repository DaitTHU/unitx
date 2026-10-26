"""Array allocation and complex components in numerical calculations."""

from collections.abc import Callable
from operator import attrgetter
from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx._numerics import Array
from unitx.factor import Factor, pi

np = pytest.importorskip('numpy')

ArrayQuantity = Quantity[Array] | AffineQuantity[Array] | LogQuantity[Array]


@pytest.mark.parametrize('operation', [np.zeros_like, np.ones_like, np.empty_like])
@pytest.mark.parametrize('order', ['C', 'F', 'A', 'K'])
@pytest.mark.parametrize('shape', [None, (), (0, 3), (3, 4)])
def test_numpy_allocation_storage(
    array_sample: ArrayQuantity, operation: Callable[..., Any], order: str, shape: Any
) -> None:
    array_sample.value.flags.writeable = False
    options = dict(
        dtype=np.complex128 if isinstance(array_sample, Quantity) else np.float32,
        order=order,
        shape=shape,
        subok=False,
        device='cpu',
    )
    expected = operation(array_sample.value, **options)
    result = operation(array_sample, **options)
    assert type(result) is type(array_sample) and result.unit == array_sample.unit
    assert result.value.dtype == expected.dtype and result.shape == expected.shape
    assert result.value.strides == expected.strides and result.value.flags.writeable
    assert not np.shares_memory(result.value, array_sample.value)
    if operation is not np.empty_like:
        np.testing.assert_array_equal(result.value, expected)
    else:
        result.value[...] = 42
        np.testing.assert_array_equal(result.value, np.full(result.shape, 42))


@pytest.mark.parametrize('operation', [np.zeros_like, np.ones_like, np.empty_like])
@pytest.mark.parametrize('dtype', [bool, object, str])
def test_numpy_allocation_invalid_dtype(operation: Callable[..., Any], dtype: Any) -> None:
    with pytest.raises(TypeError):
        operation(Quantity(np.array([1, 2]), 'm'), dtype=dtype)


@pytest.mark.parametrize('operation', [np.real, np.imag, attrgetter('real'), attrgetter('imag')])
@pytest.mark.parametrize(
    'value', [np.complex128(1 + 2j), np.array(1 + 2j), np.array([1 + 2j, 3 + 4j]), np.empty((0, 2))]
)
def test_numpy_complex_component(operation: Callable[..., Any], value: Any) -> None:
    q = Quantity(value, Factor(2) * pi * Unit('V/C'))
    expected = operation(value)
    result = operation(q)
    assert type(result) is Quantity and result.unit == q.unit
    assert type(result.value) is type(expected) and result.value.dtype == expected.dtype
    assert np.shares_memory(result.value, value) == np.shares_memory(expected, value)
    assert result.value.flags.writeable == expected.flags.writeable
    np.testing.assert_array_equal(result.value, expected)


@pytest.mark.parametrize('attributes', [False, True])
def test_numpy_complex_component_shared_write(attributes: bool) -> None:
    q = Quantity(np.array([1 + 2j]), 'V/C')
    real = q.real if attributes else np.real(q)
    imag = q.imag if attributes else np.imag(q)
    real[0] = Quantity(3, q.unit)
    imag[0] = Quantity(4, q.unit)
    np.testing.assert_array_equal(q.value, [3 + 4j])


@pytest.mark.parametrize('operation', [np.real, np.imag])
@pytest.mark.parametrize(
    'point', [AffineQuantity(np.array([1.0]), 'degC'), LogQuantity(np.array([1.0]), 'dBm')]
)
def test_numpy_complex_component_invalid_point(operation: Callable[..., Any], point: Any) -> None:
    with pytest.raises(TypeError, match='undefined'):
        operation(point)


@pytest.mark.parametrize('dtype', [int, float, complex, np.float32])
@pytest.mark.parametrize('value', [np.array([1, 2], dtype=np.int16), np.float32(2)])
def test_quantity_astype_numpy(value: Any, dtype: Any) -> None:
    q = Quantity(value, Factor(2) * pi * Unit('cm'))
    result = q.astype(dtype)
    expected = value.astype(dtype)
    assert result.unit is q.unit and type(result.value) is type(expected)
    assert result.value.dtype == expected.dtype
    np.testing.assert_array_equal(result.value, expected)
    assert not np.shares_memory(result.value, value)


@pytest.mark.parametrize('dtype', [object, bool, str])
def test_quantity_astype_numpy_invalid(dtype: Any) -> None:
    q = Quantity(np.array([1.0, 2.0]), 'm')
    with pytest.raises(TypeError, match='numeric output dtypes'):
        q.astype(dtype)
    np.testing.assert_array_equal(q.value, [1, 2])


def test_numpy_complex_integration() -> None:
    omega = Quantity(np.array([1.0, 2.0, 4.0]), 'Hz')
    weights = np.zeros_like(omega)
    weights[:-1] += np.diff(omega) / 2
    weights[1:] += np.diff(omega) / 2
    spectrum = Quantity(np.array([1 + 2j, 3 + 4j, 5 + 6j]), 'J s')
    allocated = np.empty_like(spectrum, shape=(2, 3))
    allocated[:] = spectrum
    integrated = np.sum(allocated * weights, axis=-1)
    assert integrated.unit.dimension == Unit('J').dimension
    np.testing.assert_allclose(np.real(integrated).value, [10, 10])
    np.testing.assert_allclose(np.imag(integrated).value, [13, 13])
