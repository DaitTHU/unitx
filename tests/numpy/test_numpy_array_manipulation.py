"""Array manipulation preserves each operand's unit and native storage rules."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx._numerics import Array
from unitx.factor import Factor, pi

np = pytest.importorskip('numpy')

ArrayQuantity = Quantity[Array] | AffineQuantity[Array] | LogQuantity[Array]

_TRANSFORMS = [
    (np.matrix_transpose, (), {}),
    (np.linalg.matrix_transpose, (), {}),
    (np.permute_dims, (), {'axes': (1, 0)}),
    (np.broadcast_to, ((4, 2, 3),), {'subok': True}),
    (np.flip, (), {}),
    (np.flip, (), {'axis': (0, -1)}),
    (np.fliplr, (), {}),
    (np.flipud, (), {}),
    (np.rot90, (), {'k': -1, 'axes': (1, 0)}),
    (np.rot90, (2,), {}),
    (np.roll, (2,), {}),
    (np.roll, (), {'shift': (1, -2), 'axis': (0, -1)}),
    (np.repeat, (2,), {}),
    (np.repeat, (), {'repeats': [0, 2, 1], 'axis': -1}),
    (np.tile, ((2, 1, 3),), {}),
    (np.copy, (), {}),
]


@pytest.fixture(params=['linear', 'affine', 'logarithmic'])
def sample(request: pytest.FixtureRequest) -> ArrayQuantity:
    values = np.arange(12, dtype=np.int16).reshape(2, 6)[:, ::2]
    if request.param == 'linear':
        return Quantity(values, Factor(2) * pi * Unit('cm'))
    if request.param == 'affine':
        return AffineQuantity(values, 'degC')
    return LogQuantity(values, 'dBm')


def _writeable_flag(value: Any) -> tuple[bool, tuple[tuple[type[Warning], str], ...]]:
    # NumPy may warn when inspecting a broadcast view's writeability. Verify
    # that wrapping retains those warnings instead of filtering them globally.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        writeable = value.flags.writeable
    return writeable, tuple((item.category, str(item.message)) for item in caught)


def _check_result(result: Any, expected: Any, source: ArrayQuantity) -> None:
    assert type(result) is type(source) and result.unit == source.unit
    assert type(result.value) is type(expected)
    assert result.value.dtype == expected.dtype
    assert result.value.shape == expected.shape
    assert result.value.strides == expected.strides
    assert _writeable_flag(result.value) == _writeable_flag(expected)
    assert np.shares_memory(result.value, source.value) == np.shares_memory(expected, source.value)
    np.testing.assert_array_equal(result.value, expected)


@pytest.mark.parametrize('operation,args,kwargs', _TRANSFORMS)
@pytest.mark.parametrize('readonly', [False, True])
def test_numpy_array_transform_storage(
    sample: ArrayQuantity,
    operation: Callable[..., Any],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    readonly: bool,
) -> None:
    sample.value.flags.writeable = not readonly
    before = sample.value.copy()
    expected = operation(sample.value, *args, **kwargs)
    _check_result(operation(sample, *args, **kwargs), expected, sample)
    np.testing.assert_array_equal(sample.value, before)


@pytest.mark.parametrize(
    'operation,args',
    [
        (np.matrix_transpose, ()),
        (np.flip, ()),
        (np.roll, (2,)),
        (np.repeat, (0,)),
        (np.tile, ((2, 1),)),
        (np.copy, ()),
        (np.broadcast_to, ((2, 0, 3),)),
    ],
)
def test_numpy_array_transform_empty(
    sample: ArrayQuantity, operation: Callable[..., Any], args: tuple[object, ...]
) -> None:
    q = type(sample)(np.empty((0, 3), dtype=np.float32), sample.unit)
    _check_result(operation(q, *args), operation(q.value, *args), q)


@pytest.mark.parametrize(
    'operation,args',
    [
        (np.flip, ()),
        (np.roll, (2,)),
        (np.repeat, (3,)),
        (np.tile, ((2, 1),)),
        (np.copy, ()),
        (np.broadcast_to, ((2, 3),)),
        (np.atleast_1d, ()),
        (np.atleast_2d, ()),
        (np.atleast_3d, ()),
    ],
)
@pytest.mark.parametrize('value', [2, np.float32(2), np.array(2, dtype=np.int16)])
def test_numpy_array_transform_scalar(
    sample: ArrayQuantity, operation: Callable[..., Any], args: tuple[object, ...], value: Any
) -> None:
    q = type(sample)(value, sample.unit)
    _check_result(operation(q, *args), operation(value, *args), q)


@pytest.mark.parametrize('operation', [np.atleast_1d, np.atleast_2d, np.atleast_3d])
def test_numpy_atleast_single(sample: ArrayQuantity, operation: Callable[..., Any]) -> None:
    _check_result(operation(sample), operation(sample.value), sample)


@pytest.mark.parametrize(
    'operation', [np.atleast_1d, np.atleast_2d, np.atleast_3d, np.broadcast_arrays]
)
@pytest.mark.parametrize('bare_first', [False, True])
def test_numpy_multiple_arrays_independent_units(
    operation: Callable[..., Any], bare_first: bool
) -> None:
    distance = Quantity(np.array([1, 2, 3]), Factor(2) * pi * Unit('cm'))
    temperature = AffineQuantity(np.array([[10.0], [20]]), 'degC')
    level = LogQuantity(np.array(30.0), 'dBm')
    bare = np.array([4.0, 5, 6])
    arrays = (
        (bare, distance, temperature, level, 2)
        if bare_first
        else (distance, temperature, level, bare, 2)
    )
    values = tuple(
        x.value if isinstance(x, (Quantity, AffineQuantity, LogQuantity)) else x for x in arrays
    )
    expected = operation(*values)
    result = operation(*arrays)
    assert type(result) is type(expected) and len(result) == len(arrays)
    for actual, raw, source in zip(result, expected, arrays):
        if isinstance(source, (Quantity, AffineQuantity, LogQuantity)):
            _check_result(actual, raw, source)
        else:
            assert type(actual) is type(raw) and actual.dtype == raw.dtype
            np.testing.assert_array_equal(actual, raw)


@pytest.mark.parametrize('subok', [False, True])
def test_numpy_broadcast_arrays_single(sample: ArrayQuantity, subok: bool) -> None:
    expected = np.broadcast_arrays(sample.value, subok=subok)
    result = np.broadcast_arrays(sample, subok=subok)
    assert type(result) is type(expected) and len(result) == 1
    _check_result(result[0], expected[0], sample)


def test_numpy_broadcast_to_readonly(sample: ArrayQuantity) -> None:
    result = np.broadcast_to(sample, (4, 2, 3))
    assert np.shares_memory(result.value, sample.value)
    with pytest.raises(ValueError, match='read-only'):
        result[0, 0, 0] = type(sample)(1, sample.unit)


@pytest.mark.parametrize(
    'operation',
    [
        np.flip,
        np.fliplr,
        np.flipud,
        np.rot90,
        np.matrix_transpose,
        np.linalg.matrix_transpose,
        np.atleast_1d,
        np.atleast_2d,
        np.atleast_3d,
    ],
)
def test_numpy_array_transform_shared_write(
    sample: ArrayQuantity, operation: Callable[..., Any]
) -> None:
    result = operation(sample)
    assert np.shares_memory(result.value, sample.value)
    result.value[...] = 42
    np.testing.assert_array_equal(sample.value, np.full(sample.shape, 42))


@pytest.mark.parametrize('order', ['C', 'F', 'A', 'K'])
@pytest.mark.parametrize('layout', ['C', 'F', 'transpose', 'strided', 'negative'])
def test_numpy_copy_order(sample: ArrayQuantity, order: str, layout: str) -> None:
    value = np.arange(12, dtype=np.int16).reshape(3, 4)
    if layout == 'F':
        value = np.asfortranarray(value)
    elif layout == 'transpose':
        value = value.T
    elif layout == 'strided':
        value = value[:, ::2]
    elif layout == 'negative':
        value = value[:, ::-1]
    value.flags.writeable = False
    q = type(sample)(value, sample.unit)
    result = np.copy(q, order=order, subok=True)
    _check_result(result, np.copy(value, order=order, subok=True), q)
    assert result.value.flags.writeable and not np.shares_memory(result.value, value)
    result.value[...] = -1
    assert np.all(value >= 0)


@pytest.mark.parametrize('value', [2, np.float32(2), np.array(2), np.empty((2, 0, 3))])
def test_numpy_array_metadata(sample: ArrayQuantity, value: Any) -> None:
    q = type(sample)(value, sample.unit)
    assert np.shape(q) == np.shape(value)
    assert np.ndim(q) == np.ndim(value)
    assert np.size(q) == np.size(value)
    assert type(np.shape(q)) is tuple
    assert type(np.ndim(q)) is int
    assert type(np.size(q)) is int
    if np.ndim(value):
        assert np.size(q, axis=-1) == np.size(value, axis=-1)
        assert np.size(q, 0) == np.size(value, 0)


@pytest.mark.parametrize(
    'operation,args,error',
    [
        (np.broadcast_to, ((3, 2),), ValueError),
        (np.flip, (3,), np.exceptions.AxisError),
        (np.rot90, (1, (0, 0)), ValueError),
        (np.roll, (1, 3), np.exceptions.AxisError),
        (np.repeat, (-1,), ValueError),
        (np.repeat, ([1, 2], -1), ValueError),
        (np.tile, ((-1, 2),), ValueError),
        (np.copy, ('invalid',), ValueError),
        (np.size, (3,), IndexError),
    ],
)
def test_numpy_array_transform_invalid(
    sample: ArrayQuantity,
    operation: Callable[..., Any],
    args: tuple[object, ...],
    error: type[Exception],
) -> None:
    before = sample.value.copy()
    with pytest.raises(error):
        operation(sample, *args)
    np.testing.assert_array_equal(sample.value, before)


@pytest.mark.parametrize('operation', [np.matrix_transpose, np.linalg.matrix_transpose, np.fliplr])
def test_numpy_matrix_transform_invalid_rank(
    sample: ArrayQuantity, operation: Callable[..., Any]
) -> None:
    q = type(sample)(np.array([1, 2]), sample.unit)
    with pytest.raises(ValueError):
        operation(q)


def test_numpy_broadcast_arrays_invalid_shape(sample: ArrayQuantity) -> None:
    with pytest.raises(ValueError, match='shape mismatch'):
        np.broadcast_arrays(sample, np.ones(4))


@pytest.mark.parametrize(
    'operation', [np.atleast_1d, np.atleast_2d, np.atleast_3d, np.broadcast_arrays]
)
@pytest.mark.parametrize(
    'value', [[1, 2, 3], np.array([object()], dtype=object), np.ma.array([1, 2, 3])]
)
def test_numpy_multiple_arrays_invalid_operand(
    sample: ArrayQuantity, operation: Callable[..., Any], value: Any
) -> None:
    with pytest.raises(TypeError):
        operation(sample, value)


@pytest.mark.parametrize('operation,args,kwargs', _TRANSFORMS)
def test_numpy_array_transform_invalid_keyword(
    sample: ArrayQuantity,
    operation: Callable[..., Any],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(TypeError, match='unexpected keyword'):
        operation(sample, *args, **kwargs, out=sample)


@pytest.mark.parametrize(
    'operation',
    [np.atleast_1d, np.atleast_2d, np.atleast_3d, np.broadcast_arrays, np.shape, np.ndim, np.size],
)
def test_numpy_array_function_invalid_keyword(
    sample: ArrayQuantity, operation: Callable[..., Any]
) -> None:
    with pytest.raises(TypeError, match='unexpected keyword'):
        operation(sample, out=sample)
