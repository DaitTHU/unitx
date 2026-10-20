"""Fixed-unit ndarray writes, aliasing, and unchanged targets on failure."""

import operator
import subprocess
import sys
from decimal import Decimal
from fractions import Fraction

import pytest

from unitx import Quantity, Unit
from unitx.exceptions import DimensionError
from unitx.factor import PI, Factor

np = pytest.importorskip('numpy')


@pytest.mark.parametrize(
    'key',
    [
        0,
        -1,
        np.int64(1),
        np.array(1),
        slice(None),
        slice(None, None, -1),
        (slice(None), 1),
        (None, Ellipsis),
        Ellipsis,
        (),
        [0, 1],
        np.array([1, 0]),
        np.array([True, False]),
        (np.array([0, 1]), np.array([1, 2])),
        (slice(None), [2, 0]),
        (np.array([True, False]), slice(None)),
        True,
        False,
        np.array(True),
        [],
    ],
)
def test_assignment_indexing(key):
    expected = np.arange(6.0).reshape(2, 3)
    a = expected.copy()
    q = Quantity(a, 'm')
    unit = q.unit
    expected[key] = 2
    q[key] = Quantity(200, 'cm')
    np.testing.assert_array_equal(a, expected)
    assert q.value is a and q.unit is unit and q.dtype == np.dtype('f8')


@pytest.mark.parametrize('unit', [Unit('m'), Factor(2) * Unit('m'), PI * Unit('m')])
@pytest.mark.parametrize('raw', [3, Fraction(3), Decimal('3'), np.float32(3), np.array(3.0)])
def test_assignment_unit_conversion(unit, raw):
    q = Quantity(np.zeros(3), unit)
    q[:] = Quantity(raw, 'm')
    np.testing.assert_allclose(q.value, float(Quantity(raw, 'm').value_in(unit)))


def test_assignment_dimensionless_values():
    q = Quantity(np.zeros(4), 'km/m')
    q[:2] = 2000
    q[2:] = Quantity(np.array([3.0, 4.0]), '')
    np.testing.assert_allclose(q.value, [2, 2, 0.003, 0.004])
    q[:] = np.array([0.0, 1000.0, 2000.0, 3000.0])
    np.testing.assert_allclose(q.value, [0, 1, 2, 3])


@pytest.mark.parametrize('shape', [(), (0,), (2, 0), (3, 2)])
def test_assignment_empty_scalar_and_strided(shape):
    base = np.zeros(shape + (2,), dtype=np.float32)
    a = base[..., 0]
    q = Quantity(a, 'm')
    q[...] = Quantity(250, 'cm')
    np.testing.assert_array_equal(a, np.full(shape, 2.5))
    np.testing.assert_array_equal(base[..., 1], 0)


def test_assignment_broadcasting_and_overlap():
    q = Quantity(np.arange(12.0).reshape(3, 4), 'm')
    expected = q.value.copy()
    expected[:, 1:] = expected[:, :-1].copy()
    q[:, 1:] = q[:, :-1]
    np.testing.assert_array_equal(q.value, expected)
    q[:, ::2] = Quantity(np.array([[100.0, 200.0]]), 'cm')
    np.testing.assert_array_equal(q.value[:, ::2], [[1, 2]] * 3)
    expected = q.value.copy()[::-1]
    q[:] = q[::-1]
    np.testing.assert_array_equal(q.value, expected)


def test_assignment_snapshots_indices_sharing_target_storage():
    a = np.array([2, 0, 1])
    key = a[:]
    q = Quantity(a, '')
    q[key] = np.array([10, 20, 30])
    np.testing.assert_array_equal(a, [20, 30, 10])
    a = np.ones(8, dtype=np.int8)
    mask = a.view(bool)
    q = Quantity(a, '')
    q[mask] = 0
    np.testing.assert_array_equal(a, 0)


@pytest.mark.parametrize(
    'key,value,error',
    [
        (slice(None), 0, DimensionError),
        (slice(None), np.zeros(3), DimensionError),
        (slice(None), Quantity(0, 's'), DimensionError),
        (slice(None), Quantity(np.ones(2, dtype=int), 'm'), ValueError),
        (slice(None), Quantity(1 + 2j, 'm'), TypeError),
        (slice(None), Quantity(1.0, 'cm'), TypeError),
        ([0, 9], Quantity(1, 'm'), IndexError),
        (np.array([True, False]), Quantity(1, 'm'), IndexError),
        (slice(None, None, 0), Quantity(1, 'm'), ValueError),
        (np.array([1.0]), Quantity(1, 'm'), TypeError),
        (slice(None), [1, 2, 3], TypeError),
        (slice(None), True, TypeError),
        (slice(None), Unit('m'), TypeError),
        (slice(None), Factor(2), TypeError),
    ],
)
def test_assignment_failures_preserve_target(key, value, error):
    a = np.array([4, 5, 6], dtype=np.int16)
    q = Quantity(a, 'm')
    with pytest.raises(error):
        q[key] = value
    np.testing.assert_array_equal(a, [4, 5, 6])


def test_assignment_native_boolean_shape_rules():
    q = Quantity(np.ones(3), 'm')
    with pytest.raises(TypeError):
        q[np.ones(3, dtype=bool)] = Quantity(np.zeros((1, 3)), 'm')
    np.testing.assert_array_equal(q.value, 1)


def test_assignment_cast_overflow_is_staged():
    a = np.ones(3, dtype=np.float32)
    q = Quantity(a, 'm')
    with np.errstate(over='raise'), pytest.raises(FloatingPointError):
        q[:] = Quantity(np.array([2.0, 3.0, 1e300]), 'm')
    np.testing.assert_array_equal(a, 1)


@pytest.mark.parametrize('raw', [1.0, np.float64(1), np.int32(1)])
def test_scalar_assignment_is_rejected(raw):
    with pytest.raises(TypeError, match='ndarray'):
        Quantity(raw, 'm')[...] = Quantity(2, 'm')


@pytest.mark.parametrize(
    'operation,rhs,expected',
    [
        (operator.iadd, Quantity(50, 'cm'), [1.5, 2.5, 3.5]),
        (operator.isub, Quantity(50, 'cm'), [0.5, 1.5, 2.5]),
        (operator.imul, 2, [2, 4, 6]),
        (operator.itruediv, 2, [0.5, 1, 1.5]),
        (operator.ipow, 1, [1, 2, 3]),
        (operator.imul, Quantity(2, 'km/m'), [2000, 4000, 6000]),
        (operator.itruediv, Quantity(2, 'km/m'), [0.0005, 0.001, 0.0015]),
        (operator.imul, Factor(2), [2, 4, 6]),
        (operator.itruediv, Factor(2), [0.5, 1, 1.5]),
        (operator.imul, Unit('km/m'), [1000, 2000, 3000]),
        (operator.itruediv, Unit('km/m'), [0.001, 0.002, 0.003]),
    ],
)
def test_inplace_fixed_destination(operation, rhs, expected):
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    q = Quantity(a, 'm')
    alias = q
    unit = q.unit
    result = operation(q, rhs)
    assert result is q is alias and q.value is a and q.unit is unit
    assert q.dtype == np.dtype('f4')
    np.testing.assert_allclose(a, expected)


def test_inplace_dimensionless_power_and_scaled_unit():
    unit = Factor(2) * Unit('km/m')
    q = Quantity(np.array([1.0, 2.0]), unit)
    assert operator.ipow(q, 2) is q
    assert q.unit is unit
    np.testing.assert_allclose(q.value, [2000, 8000])
    q = Quantity(np.ones(2), PI * Unit('m'))
    assert operator.imul(q, PI) is q
    np.testing.assert_allclose(q.value, np.pi)


def test_inplace_matmul():
    a = np.arange(4.0, dtype=np.float32).reshape(2, 2)
    q = Quantity(a, 'm')
    rhs = Quantity(np.array([[2.0, 1.0], [0.0, 3.0]]), 'km/m')
    expected = a @ rhs.value * 1000
    assert operator.imatmul(q, rhs) is q
    assert q.value is a and q.unit == Unit('m')
    np.testing.assert_allclose(a, expected)


def test_inplace_alias_strided_and_overlap():
    base = np.arange(12.0, dtype=np.float64).reshape(3, 4)
    q = Quantity(base[:, ::2], 'm')
    before = base.copy()
    assert operator.iadd(q, q[::-1]) is q
    np.testing.assert_array_equal(base[:, ::2], before[:, ::2] + before[::-1, ::2])
    np.testing.assert_array_equal(base[:, 1::2], before[:, 1::2])
    assert operator.imul(q, np.array([2.0, 3.0])) is q
    np.testing.assert_array_equal(base[:, ::2], (before[:, ::2] + before[::-1, ::2]) * [2, 3])


@pytest.mark.parametrize('shape', [(), (0,), (2, 0)])
def test_inplace_empty_and_zero_dimensional(shape):
    q = Quantity(np.ones(shape), 'm')
    a = q.value
    assert operator.iadd(q, Quantity(100, 'cm')) is q
    assert q.value is a
    np.testing.assert_array_equal(a, np.full(shape, 2.0))


@pytest.mark.parametrize(
    'operation,rhs,error',
    [
        (operator.iadd, Quantity(1, 's'), DimensionError),
        (operator.isub, 0, DimensionError),
        (operator.imul, Unit('m'), DimensionError),
        (operator.itruediv, Quantity(2, 's'), DimensionError),
        (operator.ipow, 2, DimensionError),
        (operator.iadd, Quantity(0.5, 'm'), TypeError),
        (operator.itruediv, 2, TypeError),
        (operator.imul, np.array([[1], [2]]), ValueError),
        (operator.iadd, Quantity(np.ones(3), 'm'), ValueError),
        (operator.imul, np.array(['x']), TypeError),
        (operator.imatmul, np.ones((3, 3)), ValueError),
    ],
)
def test_inplace_failures_preserve_destination(operation, rhs, error):
    a = np.array([2, 4], dtype=np.int32)
    q = Quantity(a, 'm')
    unit = q.unit
    with pytest.raises(error):
        operation(q, rhs)
    assert q.value is a and q.unit is unit
    np.testing.assert_array_equal(a, [2, 4])


def test_readonly_writes_fail():
    a = np.ones(3)
    a.flags.writeable = False
    q = Quantity(a, 'm')
    for operation in (lambda: q.__setitem__(slice(None), q), lambda: operator.iadd(q, q)):
        with pytest.raises(ValueError, match='read-only'):
            operation()
        np.testing.assert_array_equal(a, 1)


def test_inplace_kernel_and_cast_errors_leave_target_unchanged():
    q = Quantity(np.array([1.0, 0.0]), 'm')
    with np.errstate(all='raise'), pytest.raises(FloatingPointError):
        operator.itruediv(q, np.array([2.0, 0.0]))
    np.testing.assert_array_equal(q.value, [1, 0])
    q = Quantity(np.ones(3, dtype=np.float32), 'm')
    with np.errstate(over='raise'), pytest.raises(FloatingPointError):
        operator.imul(q, np.array([2.0, 3.0, 1e300]))
    np.testing.assert_array_equal(q.value, 1)


def test_augmented_subscription_and_repeated_indices():
    q = Quantity(np.arange(5.0), 'm')
    q[1:] += q[:-1]
    np.testing.assert_array_equal(q.value, [0, 1, 3, 5, 7])
    q[[1, 1, 3]] += Quantity(100, 'cm')
    np.testing.assert_array_equal(q.value, [0, 2, 3, 6, 7])
    q[0] += Quantity(200, 'cm')
    assert q.value[0] == 2
    with pytest.raises(DimensionError):
        q[1:] *= Unit('s')
    np.testing.assert_array_equal(q.value, [2, 2, 3, 6, 7])


@pytest.mark.parametrize('value', [1, 1.0, Fraction(1), Decimal('1'), np.float32(1), np.int32(1)])
def test_scalar_augmented_arithmetic_remains_ordinary(value):
    q = Quantity(value, 'm')
    original = q
    q += Quantity(value, 'm')
    assert q is not original and original.value == value
    assert q.value == 2 * value
    q *= Unit('s')
    assert q.unit == Unit('m s')


def test_scalar_augmented_arithmetic_does_not_import_numpy():
    script = """
import sys
from fractions import Fraction
from decimal import Decimal
from unitx import Quantity, Unit
q = Quantity(Decimal('1'), 'm')
q += Quantity(Decimal('2'), 'm')
q -= Quantity(1, 'm')
q *= 2
q /= 2
q **= 2
assert q.value == 4 and q.unit == Unit('m2')
q = Quantity(Fraction(1), 'm')
q *= Unit('s')
assert q.unit == Unit('m s')
try:
    q[0] = q
except TypeError:
    pass
else:
    raise AssertionError('scalar assignment accepted')
assert 'numpy' not in sys.modules
"""
    subprocess.run([sys.executable, '-c', script], check=True)


def test_reused_output_temporary_still_checks_casting_option():
    q = Quantity(np.ones(3), 'm')
    with pytest.raises(ValueError):
        np.add(q, q, out=q, casting='invalid')
    np.testing.assert_array_equal(q.value, 1)


@pytest.mark.parametrize('dtype', [np.int8, np.uint16, np.float32, np.float64, np.complex64])
def test_assignment_and_inplace_keep_destination_dtype(dtype):
    a = np.array([1, 2, 3], dtype=dtype)
    q = Quantity(a, 'm')
    q[:] = Quantity(np.array([4, 5, 6], dtype=dtype), 'm')
    assert operator.imul(q, 2) is q
    assert a.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(a, [8, 10, 12])


def test_inplace_python_augmented_syntax():
    q = Quantity(np.array([[1.0, 2.0], [3.0, 4.0]]), 'm')
    alias = q
    q += Quantity(100, 'cm')
    q -= Quantity(50, 'cm')
    q *= Factor(2)
    q /= 2
    q **= 1
    q @= np.eye(2)
    assert q is alias
    np.testing.assert_array_equal(q.value, [[1.5, 2.5], [3.5, 4.5]])


def test_inplace_unsupported_subclasses_and_object_arrays():
    q = Quantity(np.ones(3), 'm')
    for value in (np.ma.array([1, 2, 3]), np.array([1, 2, 3], dtype=object)):
        with pytest.raises(TypeError):
            operator.imul(q, value)
        np.testing.assert_array_equal(q.value, 1)


def test_inplace_integer_overflow_uses_numpy_rules():
    q = Quantity(np.array([100, 50], dtype=np.int8), 'm')
    expected = np.multiply(q.value.copy(), 2)
    assert operator.imul(q, 2) is q
    np.testing.assert_array_equal(q.value, expected)


def test_inplace_complex_and_matrix_input_output_alias():
    q = Quantity(np.array([1 + 2j, 3 - 4j], dtype=np.complex64), 'm')
    assert operator.itruediv(q, 2j) is q
    np.testing.assert_array_equal(q.value, [1 - 0.5j, -2 - 1.5j])
    q = Quantity(np.arange(4.0).reshape(2, 2), 'km/m')
    expected = q.value.copy() @ q.value.copy() * 1000
    assert operator.imatmul(q, q) is q
    np.testing.assert_allclose(q.value, expected)
