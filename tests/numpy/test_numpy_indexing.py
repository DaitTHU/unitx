"""Splitting, indexing and diagonals retain complete units and native storage."""

from collections.abc import Callable
from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx._numerics import Array
from unitx.exceptions import DimensionError

np = pytest.importorskip('numpy')

ArrayQuantity = Quantity[Array] | AffineQuantity[Array] | LogQuantity[Array]


def _check_result(result: Any, expected: Any, source: ArrayQuantity) -> None:
    assert type(result) is type(source) and result.unit == source.unit
    assert type(result.value) is type(expected)
    assert result.value.dtype == expected.dtype
    assert result.value.shape == expected.shape
    assert result.value.strides == expected.strides
    assert result.value.flags.writeable == expected.flags.writeable
    assert np.shares_memory(result.value, source.value) == np.shares_memory(expected, source.value)
    np.testing.assert_array_equal(result.value, expected)


@pytest.mark.parametrize(
    'operation,sections,kwargs',
    [
        (np.split, 2, {}),
        (np.split, [1, 0, 8], {'axis': -1}),
        (np.array_split, 5, {'axis': 1}),
        (np.array_split, [], {}),
        (np.hsplit, 3, {}),
        (np.hsplit, np.array([0, 1, 3]), {}),
        (np.vsplit, 2, {}),
        (np.dsplit, 2, {}),
    ],
)
@pytest.mark.parametrize('readonly', [False, True])
def test_numpy_split_storage(
    array_sample: ArrayQuantity,
    operation: Callable[..., Any],
    sections: Any,
    kwargs: dict[str, object],
    readonly: bool,
) -> None:
    array_sample.value.flags.writeable = not readonly
    expected = operation(array_sample.value, sections, **kwargs)
    result = operation(array_sample, sections, **kwargs)
    assert type(result) is list and len(result) == len(expected)
    for actual, raw in zip(result, expected):
        _check_result(actual, raw, array_sample)


def test_numpy_split_shared_write(array_sample: ArrayQuantity) -> None:
    first, _ = np.split(array_sample, 2, 0)
    assert np.shares_memory(first.value, array_sample.value)
    first[...] = type(array_sample)(42, array_sample.unit)
    np.testing.assert_array_equal(array_sample.value[0], np.full((3, 2), 42))


@pytest.mark.parametrize(
    'operation,args,error',
    [
        (np.split, (4,), ValueError),
        (np.array_split, (0,), ValueError),
        (np.split, (2, 8), IndexError),
        (np.dsplit, (2,), ValueError),
    ],
)
def test_numpy_split_invalid(
    array_sample: ArrayQuantity,
    operation: Callable[..., Any],
    args: tuple[object, ...],
    error: type[Exception],
) -> None:
    q = type(array_sample)(np.ones((2, 3)), array_sample.unit)
    with pytest.raises(error):
        operation(q, *args)


@pytest.mark.parametrize('operation', [np.split, np.array_split, np.hsplit, np.vsplit, np.dsplit])
def test_numpy_split_invalid_sections(
    array_sample: ArrayQuantity, operation: Callable[..., Any]
) -> None:
    with pytest.raises(TypeError):
        operation(array_sample, Quantity(2, Unit()))


@pytest.mark.parametrize('mode', ['raise', 'wrap', 'clip'])
@pytest.mark.parametrize('axis', [None, 0, -1])
def test_numpy_take_storage(array_sample: ArrayQuantity, axis: int | None, mode: str) -> None:
    indices = np.array([[0, -1], [1, 0]])
    expected = np.take(array_sample.value, indices, axis=axis, mode=mode)
    _check_result(np.take(array_sample, indices, axis, None, mode), expected, array_sample)


@pytest.mark.parametrize('value', [np.float32(4), np.array(4), np.empty((0, 3))])
def test_numpy_take_scalar_empty(array_sample: ArrayQuantity, value: Any) -> None:
    q = type(array_sample)(value, array_sample.unit)
    _check_result(np.take(q, []), np.take(value, []), q)


@pytest.mark.parametrize('axis', [None, -1, 0])
def test_numpy_take_along_axis(array_sample: ArrayQuantity, axis: int | None) -> None:
    if axis is None:
        indices = np.array([0, -1, 2])
    elif axis == 0:
        indices = np.array([[[1]], [[0]]])
    else:
        indices = np.array([[[1, 0]]])
    expected = np.take_along_axis(array_sample.value, indices, axis)
    _check_result(np.take_along_axis(array_sample, indices, axis), expected, array_sample)


def test_numpy_take_along_axis_default(array_sample: ArrayQuantity) -> None:
    indices = np.array([[[1, 0]]])
    if np.lib.NumpyVersion(np.__version__) < '2.3.0':
        with pytest.raises(TypeError, match='axis'):
            np.take_along_axis(array_sample, indices)
    else:
        _check_result(
            np.take_along_axis(array_sample, indices),
            np.take_along_axis(array_sample.value, indices),
            array_sample,
        )


@pytest.mark.parametrize('axis', [None, 0, -1])
def test_numpy_compress_storage(array_sample: ArrayQuantity, axis: int | None) -> None:
    # A shorter condition is valid: remaining entries along the axis are dropped.
    condition = [False, True]
    expected = np.compress(condition, array_sample.value, axis=axis)
    _check_result(np.compress(condition, array_sample, axis), expected, array_sample)


@pytest.mark.parametrize('condition', [np.array([], dtype=bool), [], (), True, [True, False]])
def test_numpy_extract_storage(array_sample: ArrayQuantity, condition: Any) -> None:
    _check_result(
        np.extract(condition, array_sample),
        np.extract(condition, array_sample.value),
        array_sample,
    )


@pytest.mark.parametrize('obj', [0, [0], slice(None, None, 2), np.array([True, False]), []])
def test_numpy_delete_storage(array_sample: ArrayQuantity, obj: Any) -> None:
    expected = np.delete(array_sample.value, obj, axis=-1)
    _check_result(np.delete(array_sample, obj, axis=-1), expected, array_sample)


@pytest.mark.parametrize('operation', [np.take, np.take_along_axis, np.delete, np.insert])
@pytest.mark.parametrize(
    'index', [Quantity(0, Unit()), Quantity(np.array([0]), Unit()), np.array([0.5])]
)
def test_numpy_indexing_invalid_index(
    array_sample: ArrayQuantity, operation: Callable[..., Any], index: Any
) -> None:
    before = array_sample.value.copy()
    with pytest.raises(TypeError):
        if operation is np.insert:
            operation(array_sample, index, array_sample)
        elif operation is np.take_along_axis:
            operation(array_sample, index, axis=-1)
        else:
            operation(array_sample, index)
    np.testing.assert_array_equal(array_sample.value, before)


@pytest.mark.parametrize('operation', [np.compress, np.extract])
@pytest.mark.parametrize(
    'condition',
    [Quantity(1, Unit()), [1, 0], np.array([0.0, 1.0]), np.ma.array([True, False])],
)
def test_numpy_indexing_invalid_condition(
    array_sample: ArrayQuantity, operation: Callable[..., Any], condition: Any
) -> None:
    with pytest.raises(TypeError):
        operation(condition, array_sample)


@pytest.mark.parametrize('operation', [np.take, np.compress])
def test_numpy_indexing_output(array_sample: ArrayQuantity, operation: Callable[..., Any]) -> None:
    q = type(array_sample)(np.array([1.0, 2, 3]), array_sample.unit)
    unit = 'm' if isinstance(q, Quantity) else ('K' if isinstance(q, AffineQuantity) else 'dBW')
    target = type(q)(np.full(2, -1.0), unit)
    buffer = target.value
    expected = type(q)(np.array([3.0, 1]), q.unit).to(target.unit).value
    if operation is np.take:
        result = np.take(q, [2, 0], None, target, 'raise')
    else:
        expected = type(q)(np.array([1.0, 3]), q.unit).to(target.unit).value
        result = np.compress([True, False, True], q, None, target)
    assert result is target and target.value is buffer
    np.testing.assert_allclose(buffer, expected)


@pytest.mark.parametrize('operation', [np.take, np.compress])
def test_numpy_indexing_output_alias(
    array_sample: ArrayQuantity, operation: Callable[..., Any]
) -> None:
    q = type(array_sample)(np.array([1.0, 2, 3]), array_sample.unit)
    target = type(q)(q.value[::-1], q.unit)
    if operation is np.take:
        result = np.take(q, [0, 1, 2], out=target)
    else:
        result = np.compress([True, True, True], q, out=target)
    assert result is target
    np.testing.assert_array_equal(target.value, [1, 2, 3])
    np.testing.assert_array_equal(q.value, [3, 2, 1])


def test_numpy_take_output_index_alias() -> None:
    indices = np.array([2, 0, 1])
    target = Quantity(indices, 'm')
    q = Quantity(np.array([10, 20, 30]), 'm')
    assert np.take(q, indices, out=target) is target
    np.testing.assert_array_equal(indices, [30, 10, 20])


@pytest.mark.parametrize('operation', [np.take, np.compress])
@pytest.mark.parametrize('failure', ['dimension', 'shape', 'dtype', 'readonly', 'bare', 'index'])
def test_numpy_indexing_output_invalid(operation: Callable[..., Any], failure: str) -> None:
    q = Quantity(np.array([1.5, 2.5, 3.5]), 'm')
    buffer = np.full(2 if failure == 'shape' else 3, -1, dtype=int if failure == 'dtype' else float)
    target = (
        buffer if failure == 'bare' else Quantity(buffer, 's' if failure == 'dimension' else 'm')
    )
    if failure == 'readonly':
        buffer.flags.writeable = False
    expected = buffer.copy()
    error = (
        DimensionError
        if failure == 'dimension'
        else (
            TypeError
            if failure in ('bare', 'dtype')
            else (IndexError if failure == 'index' else ValueError)
        )
    )
    with pytest.raises(error):
        if operation is np.take:
            np.take(q, [0, 1, 9 if failure == 'index' else 2], out=target)
        else:
            condition = [True, True, True, True] if failure == 'index' else [True, True, True]
            np.compress(condition, q, out=target)
    np.testing.assert_array_equal(buffer, expected)
    np.testing.assert_array_equal(q.value, [1.5, 2.5, 3.5])


@pytest.mark.parametrize('operation', [np.take, np.compress])
def test_numpy_indexing_output_invalid_family(operation: Callable[..., Any]) -> None:
    q = AffineQuantity(np.array([0.0, 10]), 'degC')
    target = Quantity(np.full(2, -1.0), 'K')
    with pytest.raises(DimensionError, match='point/difference'):
        if operation is np.take:
            np.take(q, [0, 1], out=target)
        else:
            np.compress([True, True], q, out=target)
    np.testing.assert_array_equal(target.value, [-1, -1])


@pytest.mark.parametrize(
    'operation,shape,args,kwargs',
    [
        (np.diagonal, (2, 3, 4), (), {'offset': -1, 'axis1': -1, 'axis2': 0}),
        (np.diagonal, (2, 3), (5,), {}),
        (np.diag, (3,), (-1,), {}),
        (np.diag, (2, 3), (1,), {}),
        (np.diagflat, (2, 3), (), {'k': 2}),
        (np.diagflat, (), (), {}),
        (np.diag, (0,), (), {}),
        (np.tril, (2, 3, 4), (), {'k': -1}),
        (np.triu, (2, 3, 4), (2,), {}),
        (np.tril, (3,), (), {}),
        (np.triu, (2, 0), (), {}),
    ],
)
@pytest.mark.parametrize('readonly', [False, True])
def test_numpy_diagonal_storage(
    array_sample: ArrayQuantity,
    operation: Callable[..., Any],
    shape: tuple[int, ...],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    readonly: bool,
) -> None:
    values = np.arange(np.prod(shape, dtype=int), dtype=np.float64).reshape(shape)
    values.flags.writeable = not readonly
    q = type(array_sample)(values, array_sample.unit)
    expected = operation(values, *args, **kwargs)
    _check_result(operation(q, *args, **kwargs), expected, q)


def test_numpy_diagonal_readonly(array_sample: ArrayQuantity) -> None:
    q = type(array_sample)(np.arange(9.0).reshape(3, 3), array_sample.unit)
    result = np.diagonal(q)
    assert np.shares_memory(result.value, q.value)
    with pytest.raises(ValueError, match='read-only'):
        result[0] = type(q)(1, q.unit)


@pytest.mark.parametrize('operation', [np.diag, np.diagflat, np.tril, np.triu])
def test_numpy_diagonal_zero_readings(
    array_sample: ArrayQuantity, operation: Callable[..., Any]
) -> None:
    q = type(array_sample)(np.array([5.0, 10]), array_sample.unit)
    result = operation(q)
    assert result.unit == q.unit
    if operation in (np.diag, np.diagflat, np.tril):
        assert result.value[0, 1] == 0
    else:
        assert result.value[1, 0] == 0


@pytest.mark.parametrize(
    'operation,args,kwargs,error',
    [
        (np.diag, (), {}, ValueError),
        (np.diagonal, (), {'axis1': 0, 'axis2': 0}, ValueError),
        (np.diagonal, (), {'axis2': 4}, IndexError),
        (np.tril, (), {'k': 'invalid'}, TypeError),
    ],
)
def test_numpy_diagonal_invalid(
    array_sample: ArrayQuantity,
    operation: Callable[..., Any],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    before = array_sample.value.copy()
    with pytest.raises(error):
        operation(array_sample, *args, **kwargs)
    np.testing.assert_array_equal(array_sample.value, before)


@pytest.mark.parametrize(
    'operation,args',
    [
        (np.split, (2,)),
        (np.array_split, (2,)),
        (np.hsplit, (3,)),
        (np.vsplit, (2,)),
        (np.dsplit, (2,)),
        (np.take_along_axis, (np.array([[[0, 1]]]), -1)),
        (np.delete, (0,)),
        (np.diagonal, ()),
        (np.diag, ()),
        (np.diagflat, ()),
        (np.tril, ()),
        (np.triu, ()),
    ],
)
def test_numpy_indexing_invalid_keyword(
    array_sample: ArrayQuantity, operation: Callable[..., Any], args: tuple[object, ...]
) -> None:
    with pytest.raises(TypeError, match='unexpected keyword'):
        operation(array_sample, *args, out=array_sample)
