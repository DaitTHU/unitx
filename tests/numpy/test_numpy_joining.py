"""Joining aligns samples before NumPy chooses shape and dtype."""

from collections.abc import Callable
from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx._numerics import Array
from unitx.exceptions import DimensionError
from unitx.factor import Factor
from unitx.si import si

np = pytest.importorskip('numpy')

ArrayQuantity = Quantity[Array] | AffineQuantity[Array] | LogQuantity[Array]
STACKS = [np.hstack, np.vstack, np.dstack, np.column_stack]


@pytest.mark.parametrize('operation', STACKS)
@pytest.mark.parametrize('shape', [(), (3,), (2, 3), (2, 3, 2)])
def test_numpy_stacking_alignment(
    array_sample: ArrayQuantity, operation: Callable[..., Any], shape: tuple[int, ...]
) -> None:
    a = type(array_sample)(np.full(shape, 1.0), array_sample.unit)
    if isinstance(a, Quantity):
        b = Quantity(np.full(shape, 3.0), Factor(2) * a.unit)
        converted = 6.0
    elif isinstance(a, AffineQuantity):
        b = AffineQuantity(np.full(shape, 273.15), 'K')
        converted = 0.0
    else:
        b = LogQuantity(np.full(shape, 0.0), 'dBW')
        converted = 30.0
    result = operation((a, b))
    assert type(result) is type(a) and result.unit == a.unit
    expected = operation((a.value, np.full(shape, converted)))
    assert result.value.dtype == expected.dtype
    np.testing.assert_allclose(result.value, expected, atol=1e-12)
    assert not np.shares_memory(result.value, a.value)
    assert not np.shares_memory(result.value, b.value)


@pytest.mark.parametrize('operation', STACKS)
def test_numpy_stacking_single_array(
    array_sample: ArrayQuantity, operation: Callable[..., Any]
) -> None:
    result = operation(array_sample)
    assert type(result) is type(array_sample) and result.unit == array_sample.unit
    np.testing.assert_array_equal(result.value, operation(array_sample.value))


@pytest.mark.parametrize('operation', STACKS)
@pytest.mark.parametrize('bare_first', [False, True])
def test_numpy_stacking_dimensionless(operation: Callable[..., Any], bare_first: bool) -> None:
    scaled = Quantity(np.array([1, 2]), Unit('m/cm'))
    bare = np.array([3, 4])
    inputs = (bare, scaled) if bare_first else (scaled, bare)
    values = (bare / 100, scaled.value) if bare_first else (scaled.value, bare / 100)
    result = operation(inputs)
    assert result.unit == scaled.unit
    np.testing.assert_allclose(result.value, operation(values))


@pytest.mark.parametrize('operation', [np.hstack, np.vstack])
def test_numpy_stacking_dtype(operation: Callable[..., Any]) -> None:
    a = Quantity(np.array([1, 2], dtype=np.int16), 'm')
    b = Quantity(np.array([100, 200], dtype=np.int16), 'cm')
    result = operation((a, b), dtype=np.float32, casting='same_kind')
    assert result.value.dtype == np.float32
    np.testing.assert_allclose(result.value, operation(([1, 2], [1, 2])))
    with pytest.raises(TypeError, match="rule 'safe'"):
        operation((a, b), dtype=np.int16, casting='safe')


@pytest.mark.parametrize('operation', [np.hstack, np.vstack])
@pytest.mark.parametrize('dtype', [bool, object, str])
def test_numpy_stacking_invalid_dtype(operation: Callable[..., Any], dtype: Any) -> None:
    with pytest.raises(TypeError, match='numeric output dtypes'):
        operation((Quantity(np.array([1]), 'm'),), dtype=dtype)


@pytest.mark.parametrize('operation', STACKS + [np.concatenate, np.stack])
@pytest.mark.parametrize(
    'other',
    [
        Quantity(np.array([1.0]), 's'),
        np.array([1.0]),
        AffineQuantity(np.array([1.0]), 'K'),
    ],
)
def test_numpy_stacking_invalid_dimension(operation: Callable[..., Any], other: Any) -> None:
    a = Quantity(np.array([1.0]), 'm')
    before = a.value.copy()
    with pytest.raises(DimensionError):
        operation((a, other))
    np.testing.assert_array_equal(a.value, before)


@pytest.mark.parametrize('operation', STACKS + [np.concatenate, np.stack])
def test_numpy_stacking_invalid_convention(operation: Callable[..., Any]) -> None:
    power = LogQuantity(np.array([0.0]), si.dB(si.mW))
    amplitude = LogQuantity(np.array([0.0]), si.dB(si.mW, kind='amplitude'))
    with pytest.raises(DimensionError, match='same convention'):
        operation((power, amplitude))


@pytest.mark.parametrize('operation', STACKS)
@pytest.mark.parametrize('other', [[1, 2], np.array([object()]), np.ma.array([1, 2])])
def test_numpy_stacking_invalid_operand(operation: Callable[..., Any], other: Any) -> None:
    with pytest.raises(TypeError):
        operation((Quantity(np.array([1, 2]), 'm'), other))


@pytest.mark.parametrize('operation', STACKS)
def test_numpy_stacking_invalid_keyword(operation: Callable[..., Any]) -> None:
    a = Quantity(np.array([1, 2]), 'm')
    with pytest.raises(TypeError, match='unexpected keyword'):
        operation((a,), out=a)


@pytest.mark.parametrize('operation', [np.dstack, np.column_stack])
def test_numpy_stacking_invalid_options(operation: Callable[..., Any]) -> None:
    with pytest.raises(TypeError, match='unexpected keyword'):
        operation((Quantity(np.array([1]), 'm'),), dtype=np.float64)


@pytest.mark.parametrize('axis', [None, 0, -1])
def test_numpy_append_alignment(array_sample: ArrayQuantity, axis: int | None) -> None:
    result = np.append(array_sample, array_sample, axis=axis)
    assert type(result) is type(array_sample) and result.unit == array_sample.unit
    np.testing.assert_array_equal(
        result.value, np.append(array_sample.value, array_sample.value, axis=axis)
    )
    assert not np.shares_memory(result.value, array_sample.value)


@pytest.mark.parametrize('operation', [np.append, np.insert])
def test_numpy_join_binary_conversion(operation: Callable[..., Any]) -> None:
    a = Quantity(np.array([1.0, 2]), 'm')
    b = Quantity(np.array([300.0]), 'cm')
    args = (a, 1, b) if operation is np.insert else (a, b)
    result = operation(*args)
    assert result.unit == a.unit
    np.testing.assert_allclose(result.value, [1, 3, 2] if operation is np.insert else [1, 2, 3])
    # A bare first input is dimensionless and is aligned to the first wrapper.
    scaled = Quantity(np.array([2.0]), Unit('m/cm'))
    args = (np.array([100.0]), 1, scaled) if operation is np.insert else (np.array([100.0]), scaled)
    result = operation(*args)
    assert result.unit == scaled.unit
    np.testing.assert_allclose(result.value, [1, 2])


@pytest.mark.parametrize('obj', [0, [0], slice(0, 2), np.array([1, 0])])
def test_numpy_insert_shape(array_sample: ArrayQuantity, obj: Any) -> None:
    values = type(array_sample)(np.array(42.0), array_sample.unit)
    result = np.insert(array_sample, obj, values, axis=-1)
    expected = np.insert(array_sample.value, obj, values.value, axis=-1)
    assert type(result) is type(array_sample) and result.unit == array_sample.unit
    assert result.value.dtype == expected.dtype
    np.testing.assert_array_equal(result.value, expected)
    assert not np.shares_memory(result.value, array_sample.value)


def test_numpy_insert_integer_dtype() -> None:
    a = Quantity(np.array([1, 2], dtype=np.int16), 'm')
    result = np.insert(a, 1, Quantity(150, 'cm'))
    assert result.value.dtype == np.int16
    np.testing.assert_array_equal(result.value, [1, 1, 2])


@pytest.mark.parametrize('operation', [np.append, np.insert])
@pytest.mark.parametrize('family', [AffineQuantity, LogQuantity])
def test_numpy_join_binary_coordinate(operation: Callable[..., Any], family: Any) -> None:
    a = family(np.array([0.0]), 'degC' if family is AffineQuantity else 'dBm')
    b = family(
        np.array([273.15 if family is AffineQuantity else 0.0]),
        'K' if family is AffineQuantity else 'dBW',
    )
    args = (a, 0, b) if operation is np.insert else (a, b)
    result = operation(*args)
    expected = (
        [0, 0] if family is AffineQuantity else ([30, 0] if operation is np.insert else [0, 30])
    )
    assert type(result) is family and result.unit == a.unit
    np.testing.assert_allclose(result.value, expected, atol=1e-12)


@pytest.mark.parametrize('operation', [np.append, np.insert])
def test_numpy_join_binary_invalid_dimension(operation: Callable[..., Any]) -> None:
    a = Quantity(np.array([1, 2]), 'm')
    args = (a, 1, Quantity(1, 's')) if operation is np.insert else (a, Quantity(1, 's'))
    with pytest.raises(DimensionError):
        operation(*args)
    np.testing.assert_array_equal(a.value, [1, 2])
