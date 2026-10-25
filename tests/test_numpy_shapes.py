"""Shape transformations retain units, coordinates and NumPy storage behavior."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx._numerics import Array
from unitx.factor import Factor, pi

np = pytest.importorskip('numpy')

ArrayQuantity = Quantity[Array] | AffineQuantity[Array] | LogQuantity[Array]

_TRANSFORMS = [
    pytest.param(np.squeeze, (), {}, id='squeeze'),
    pytest.param(np.squeeze, (), {'axis': (0, -2)}, id='squeeze-tuple'),
    pytest.param(np.expand_dims, (0,), {}, id='expand-dims'),
    pytest.param(np.expand_dims, (), {'axis': (0, -1)}, id='expand-dims-tuple'),
    pytest.param(np.moveaxis, (0, -1), {}, id='moveaxis'),
    pytest.param(np.moveaxis, (), {'source': (0, 3), 'destination': (3, 0)}, id='moveaxis-tuple'),
    pytest.param(np.swapaxes, (), {'axis1': 1, 'axis2': -1}, id='swapaxes'),
    pytest.param(np.rollaxis, (3,), {}, id='rollaxis'),
    pytest.param(np.rollaxis, (), {'axis': 0, 'start': 4}, id='rollaxis-end'),
    pytest.param(np.rollaxis, (3, -3), {}, id='rollaxis-negative-start'),
    pytest.param(np.ravel, (), {}, id='ravel'),
]


@pytest.fixture(params=['linear', 'affine', 'logarithmic'])
def sample(request: pytest.FixtureRequest) -> ArrayQuantity:
    values = np.arange(24, dtype=np.int16).reshape(1, 2, 1, 12)[..., ::4]
    if request.param == 'linear':
        return Quantity(values, Factor(2) * pi * Unit('cm'))
    if request.param == 'affine':
        return AffineQuantity(values, 'degC')
    return LogQuantity(values, 'dBm')


@pytest.mark.parametrize('operation,args,kwargs', _TRANSFORMS)
@pytest.mark.parametrize('readonly', [False, True])
def test_numpy_shape_transform_storage(
    sample: ArrayQuantity,
    operation: Callable[..., Any],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    readonly: bool,
) -> None:
    sample.value.flags.writeable = not readonly
    before = sample.value.copy()
    expected = operation(sample.value, *args, **kwargs)
    result = operation(sample, *args, **kwargs)
    assert type(result) is type(sample)
    assert result.unit == sample.unit
    assert type(result.value) is type(expected)
    assert result.value.dtype == expected.dtype
    assert result.value.shape == expected.shape
    assert result.value.strides == expected.strides
    assert result.value.flags.writeable == expected.flags.writeable
    assert np.shares_memory(result.value, sample.value) == np.shares_memory(expected, sample.value)
    np.testing.assert_array_equal(result.value, expected)
    np.testing.assert_array_equal(sample.value, before)


@pytest.mark.parametrize('operation,args,kwargs', _TRANSFORMS)
def test_numpy_shape_transform_empty(
    sample: ArrayQuantity,
    operation: Callable[..., Any],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    q = type(sample)(np.empty((1, 0, 1, 3), dtype=np.float32), sample.unit)
    expected = operation(q.value, *args, **kwargs)
    result = operation(q, *args, **kwargs)
    assert type(result) is type(q) and result.unit == q.unit
    assert result.value.shape == expected.shape
    assert result.value.dtype == expected.dtype


@pytest.mark.parametrize(
    'operation,args',
    [(np.squeeze, ()), (np.expand_dims, (-1,)), (np.moveaxis, ((), ())), (np.ravel, ())],
)
@pytest.mark.parametrize('value', [2, np.float32(2), np.array(2, dtype=np.int16)])
def test_numpy_shape_transform_scalar(
    sample: ArrayQuantity, operation: Callable[..., Any], args: tuple[object, ...], value: Any
) -> None:
    q = type(sample)(value, sample.unit)
    expected = operation(value, *args)
    result = operation(q, *args)
    assert type(result) is type(q) and result.unit == q.unit
    assert type(result.value) is type(expected)
    assert result.value.shape == expected.shape
    assert result.value.dtype == expected.dtype
    np.testing.assert_array_equal(result.value, expected)


def test_numpy_squeeze_zero_dimensional_result(sample: ArrayQuantity) -> None:
    q = type(sample)(np.array([[2]], dtype=np.int16), sample.unit)
    result = np.squeeze(q)
    assert type(result) is type(q) and result.unit == q.unit
    assert type(result.value) is np.ndarray and result.value.shape == ()
    assert result.value.item() == 2
    assert np.shares_memory(result.value, q.value)


@pytest.mark.parametrize('operation,args,kwargs', _TRANSFORMS)
def test_numpy_shape_transform_shared_write(
    sample: ArrayQuantity,
    operation: Callable[..., Any],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    q = type(sample)(np.arange(6.0).reshape(1, 2, 1, 3), sample.unit)
    result = operation(q, *args, **kwargs)
    assert np.shares_memory(result.value, q.value)
    result.value.flat[0] = 42
    assert q.value.flat[0] == 42
    assert q.unit == sample.unit


@pytest.mark.parametrize('order', ['C', 'F', 'A', 'K'])
@pytest.mark.parametrize('layout', ['C', 'F', 'transpose', 'strided', 'negative'])
def test_numpy_ravel_order(sample: ArrayQuantity, order: str, layout: str) -> None:
    values = np.arange(12, dtype=np.int16).reshape(3, 4)
    if layout == 'F':
        values = np.asfortranarray(values)
    elif layout == 'transpose':
        values = values.T
    elif layout == 'strided':
        values = values[:, ::2]
    elif layout == 'negative':
        values = values[:, ::-1]
    q = type(sample)(values, sample.unit)
    before = values.copy()
    expected = np.ravel(values, order)
    result = np.ravel(q, order)
    assert type(result) is type(q) and result.unit == q.unit
    assert result.value.ndim == 1 and result.value.flags.c_contiguous
    assert np.shares_memory(result.value, values) == np.shares_memory(expected, values)
    np.testing.assert_array_equal(result.value, expected)
    if not np.shares_memory(result.value, values):
        result.value[0] = -1
        np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize(
    'operation,args',
    [
        (np.squeeze, (4,)),
        (np.expand_dims, (-6,)),
        (np.moveaxis, (4, 0)),
        (np.swapaxes, (0, -5)),
        (np.rollaxis, (0, 5)),
    ],
)
def test_numpy_shape_transform_invalid_axis(
    sample: ArrayQuantity, operation: Callable[..., Any], args: tuple[object, ...]
) -> None:
    before = sample.value.copy()
    with pytest.raises(np.exceptions.AxisError):
        operation(sample, *args)
    np.testing.assert_array_equal(sample.value, before)


def test_numpy_squeeze_invalid_axis_size(sample: ArrayQuantity) -> None:
    with pytest.raises(ValueError, match='size not equal to one'):
        np.squeeze(sample, axis=1)


def test_numpy_squeeze_invalid_repeated_axis(sample: ArrayQuantity) -> None:
    with pytest.raises(ValueError, match='duplicate value'):
        np.squeeze(sample, axis=(0, 0))


@pytest.mark.parametrize(
    'operation,args',
    [
        (np.expand_dims, ((0, 0),)),
        (np.moveaxis, ((0, 0), (1, 2))),
        (np.moveaxis, ((0, 1), (2, 2))),
    ],
)
def test_numpy_shape_transform_invalid_repeated_axis(
    sample: ArrayQuantity, operation: Callable[..., Any], args: tuple[object, ...]
) -> None:
    with pytest.raises(ValueError, match='repeated axis'):
        operation(sample, *args)


def test_numpy_moveaxis_invalid_axis_count(sample: ArrayQuantity) -> None:
    with pytest.raises(ValueError, match='same number of elements'):
        np.moveaxis(sample, (0, 1), (2,))


@pytest.mark.parametrize('operation,args,kwargs', _TRANSFORMS)
def test_numpy_shape_transform_invalid_keyword(
    sample: ArrayQuantity,
    operation: Callable[..., Any],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(TypeError, match='unexpected keyword'):
        operation(sample, *args, **kwargs, out=sample)


def test_numpy_ravel_invalid_order(sample: ArrayQuantity) -> None:
    with pytest.raises(ValueError, match='order'):
        np.ravel(sample, order='invalid')


def test_numpy_ravel_invalid_copy(sample: ArrayQuantity) -> None:
    with pytest.raises(TypeError, match='unexpected keyword'):
        np.ravel(sample, copy=True)
