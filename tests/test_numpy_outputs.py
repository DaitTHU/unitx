"""Unit-aware function and reduction outputs, including shared storage."""

import pytest

np = pytest.importorskip('numpy')

from unitx import Quantity, Unit  # noqa: E402
from unitx.exceptions import DimensionError  # noqa: E402
from unitx.factor import Factor  # noqa: E402


@pytest.mark.parametrize(
    'operation',
    [np.sum, np.mean, np.median, np.std, np.var, np.cumsum, np.add.reduce, np.add.accumulate],
)
def test_numpy_function_out_conversion(operation):
    unit = Factor(2) * Unit('m')
    q = Quantity(np.arange(6.0).reshape(2, 3), unit)
    natural_unit = unit**2 if operation is np.var else unit
    target_unit = Unit('cm2' if operation is np.var else 'cm')
    expected = Quantity(operation(q.value, axis=0), natural_unit).value_in(target_unit)
    target = Quantity(np.full(np.shape(expected), -123.0), target_unit)
    buffer = target.value
    assert operation(q, axis=0, out=target) is target
    assert target.value is buffer and target.unit is target_unit
    np.testing.assert_allclose(buffer, expected)
    np.testing.assert_array_equal(q.value, np.arange(6.0).reshape(2, 3))
    ordinary = operation(q, axis=0, out=None)
    assert ordinary.unit == natural_unit
    np.testing.assert_allclose(ordinary.value, operation(q.value, axis=0))


@pytest.mark.parametrize('operation', [np.sum, np.mean, np.median, np.std, np.var, np.add.reduce])
def test_numpy_reduction_out_axes(operation):
    q = Quantity(np.arange(24.0).reshape(2, 3, 4), 'm')
    unit = 'm2' if operation is np.var else 'm'
    expected = operation(q.value, axis=(0, -1), keepdims=True)
    target = Quantity(np.empty((1, 3, 1)), unit)
    assert operation(q, axis=(0, -1), keepdims=True, out=target) is target
    np.testing.assert_allclose(target.value, expected)


def test_numpy_function_out_positional():
    q = Quantity(np.array([1.0, 2.0, 3.0]), 'm')
    target = Quantity(np.array(-123.0), 'cm')
    for operation in (np.sum, np.mean, np.std, np.var):
        actual = Quantity(target.value, 'cm2') if operation is np.var else target
        assert operation(q, None, None, actual) is actual
    assert np.median(q, None, target) is target
    vector = Quantity(np.empty(3), 'cm')
    assert np.cumsum(q, None, None, vector) is vector
    assert np.add.reduce(q, None, None, target) is target
    assert np.add.reduce(q, out=(target,)) is target
    assert np.add.accumulate(q, out=(vector,)) is vector
    assert np.add.reduce(q, out=(None,)).value == 6
    index = np.array(-1, dtype=np.intp)
    assert np.argmax(q, None, index) is index
    assert index == 2


@pytest.mark.parametrize('operation', [np.concatenate, np.stack])
def test_numpy_join_out_conversion(operation):
    q = Quantity(np.arange(3.0), 'm')
    other = Quantity(np.arange(3.0) * 100, 'cm')
    expected = operation((q.value, q.value)) * 1000
    storage = np.full((*expected.shape, 2), -123.0)
    target = Quantity(storage[..., 0], 'mm')
    assert operation((q, other), out=target) is target
    np.testing.assert_array_equal(target.value, expected)
    np.testing.assert_array_equal(storage[..., 1], -123)
    assert operation((q, other), 0, target) is target


def test_numpy_join_out_alias():
    storage = np.arange(6.0)
    q = Quantity(storage, 'm')
    assert np.concatenate((q[3:], q[:3]), out=q) is q
    np.testing.assert_array_equal(storage, [3, 4, 5, 0, 1, 2])


@pytest.mark.parametrize('operation', [np.argmin, np.argmax])
def test_numpy_index_out(operation):
    q = Quantity(np.array([[3, 1, 4], [2, 7, 0]]), 'm')
    storage = np.full((3, 2), -123, dtype=np.intp)
    target = storage[:, 0]
    assert operation(q, axis=0, out=target) is target
    np.testing.assert_array_equal(target, operation(q.value, axis=0))
    np.testing.assert_array_equal(storage[:, 1], -123)
    kept = np.empty((1, 3), dtype=np.intp)
    assert operation(q, axis=0, out=kept, keepdims=True) is kept
    np.testing.assert_array_equal(kept, operation(q.value, axis=0, keepdims=True))


def test_numpy_reduction_out_storage():
    storage = np.arange(6.0).reshape(2, 3)
    q = Quantity(storage, 'm')
    target = Quantity(storage[0], 'cm')
    assert np.sum(q, axis=0, out=target) is target
    np.testing.assert_array_equal(storage, [[300, 500, 700], [3, 4, 5]])
    assert np.cumsum(q, axis=1, out=q) is q
    np.testing.assert_array_equal(storage, [[300, 800, 1500], [3, 7, 12]])
    assert np.add.accumulate(q, axis=0, out=q) is q
    np.testing.assert_array_equal(storage, [[300, 800, 1500], [303, 807, 1512]])
    q.value.flags.writeable = False
    separate = Quantity(np.empty(3), 'm')
    assert np.sum(q, axis=0, out=separate) is separate
    np.testing.assert_array_equal(separate.value, [603, 1607, 3012])


def test_numpy_function_out_empty():
    q = Quantity(np.empty((0, 3)), 'm')
    target = Quantity(np.full(3, -123.0), 'cm')
    assert np.sum(q, axis=0, out=target) is target
    np.testing.assert_array_equal(target.value, 0)
    empty = Quantity(np.empty((0, 3)), 'cm')
    assert np.cumsum(q, axis=0, out=empty) is empty
    assert np.add.accumulate(q, axis=0, out=empty) is empty
    scalar = Quantity(np.array(2.0), 'km/m')
    coherent = Quantity(np.array(-123.0), '')
    assert np.sum(scalar, out=coherent) is coherent
    assert coherent.value == 2000


@pytest.mark.parametrize('operation', [np.sum, np.add.reduce, np.cumsum, np.add.accumulate])
def test_numpy_function_out_invalid_dimension(operation):
    q = Quantity(np.arange(3.0), 'm')
    target = Quantity(np.full(np.shape(operation(q.value)), -123.0), 's')
    with pytest.raises(DimensionError, match='out has an incompatible dimension'):
        operation(q, out=target)
    np.testing.assert_array_equal(target.value, -123)


@pytest.mark.parametrize('shape', [(1,), (2, 3), (0,)])
def test_numpy_reduction_out_invalid_shape(shape):
    q = Quantity(np.arange(3.0), 'm')
    for operation in (np.sum, np.add.reduce, np.argmax):
        buffer = np.full(shape, -123, dtype=np.intp)
        out = buffer if operation is np.argmax else Quantity(buffer, 'm')
        with pytest.raises(ValueError, match='result shape'):
            operation(q, out=out)
        np.testing.assert_array_equal(buffer, -123)


def test_numpy_function_out_invalid_buffer():
    q = Quantity(np.arange(3.0), 'm')
    for target in (np.array(-123.0), Quantity(1, 'm'), Quantity(np.float64(1), 'm')):
        with pytest.raises(TypeError, match='plain numeric ndarray'):
            np.sum(q, out=target)
    target = Quantity(np.array(-123.0), 'm')
    target.value.flags.writeable = False
    with pytest.raises(ValueError, match='read-only'):
        np.sum(q, out=target)
    assert target.value == -123


@pytest.mark.parametrize(
    'target', [np.array(-123.0), np.array(False), Quantity(np.array(-123), '')]
)
def test_numpy_index_out_invalid_buffer(target):
    q = Quantity(np.arange(3.0), 'm')
    with pytest.raises(TypeError, match='Unitless out'):
        np.argmax(q, out=target)
    assert (target.value if isinstance(target, Quantity) else target) == (
        False if isinstance(target, np.ndarray) and target.dtype.kind == 'b' else -123
    )


def test_numpy_function_out_invalid_cast():
    q = Quantity(np.array([1.0, 2.5]), 'cm')
    target = Quantity(np.array(-123), 'm')
    for operation in (np.sum, np.mean, np.median, np.add.reduce):
        with pytest.raises(TypeError):
            operation(q, out=target)
        assert target.value == -123
    narrow = Quantity(np.array(-123, dtype=np.float32), 'm')
    assert np.sum(q, dtype=np.float64, out=narrow) is narrow
    assert narrow.value.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(narrow.value, 0.035)
    with np.errstate(over='raise'):
        with pytest.raises(FloatingPointError):
            np.sum(Quantity(np.array([1e100]), 'm'), out=narrow)
    np.testing.assert_allclose(narrow.value, 0.035)


def test_numpy_join_out_invalid_options():
    q = Quantity(np.array([1.0]), 'cm')
    target = Quantity(np.full(2, -123), 'm')
    with pytest.raises(TypeError):
        np.concatenate((q, q), out=target)
    np.testing.assert_array_equal(target.value, -123)
    assert np.concatenate((q, q), out=target, casting='unsafe') is target
    np.testing.assert_array_equal(target.value, 0)
    for operation in (np.concatenate, np.stack):
        with pytest.raises(TypeError, match='both dtype and out'):
            operation((q, q), out=target, dtype=float)
    with pytest.raises(DimensionError):
        np.concatenate((q, Quantity(np.array([1.0]), 's')), out=target)
    np.testing.assert_array_equal(target.value, 0)


def test_numpy_function_out_invalid_kernel():
    q = Quantity(np.array([np.inf, np.inf]), 'm')
    target = Quantity(np.array(-123.0), 'm2')
    with np.errstate(invalid='raise'):
        with pytest.raises(FloatingPointError):
            np.var(q, out=target)
    assert target.value == -123
    with pytest.raises(TypeError):
        np.median(q, out=target, overwrite_input=True)
    assert target.value == -123


def test_numpy_reduction_out_protocol():
    q = Quantity(np.arange(3.0), 'm')

    class Foreign:
        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            return 'foreign dispatched'

        def __array_function__(self, func, types, args, kwargs):
            return 'foreign dispatched'

    assert np.add.reduce(q, out=Foreign()) == 'foreign dispatched'
    assert np.sum(q, out=Foreign()) == 'foreign dispatched'
    target = Quantity(np.array(-123.0), 'm')
    for options in ({'where': True}, {'initial': 1}, {'casting': 'unsafe'}):
        with pytest.raises(TypeError):
            np.add.reduce(q, out=target, **options)
    assert target.value == -123
