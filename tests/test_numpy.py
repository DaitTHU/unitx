"""NumPy's numeric domain, unit rules, and deliberate failure boundaries."""

import os
import subprocess
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from unitx.exceptions import DimensionError
from unitx.factor import PI, Factor
from unitx.quantity import Quantity
from unitx.unit import ScaledUnit, Unit

np = pytest.importorskip('numpy')


def assert_quantity(q, values, unit):
    assert isinstance(q, Quantity)
    np.testing.assert_allclose(q.value, values)
    assert q.unit == Unit(unit) if isinstance(unit, str) else q.unit == unit
    assert np.asarray(q.value).dtype.kind in 'iufc'


@pytest.mark.parametrize('unit', [Unit('m'), Factor(2) * Unit('m')])
@pytest.mark.parametrize('value', [np.arange(4), np.array(3.0), np.float32(2), np.int64(4)])
def test_construction_preserves_entire_magnitude(unit, value):
    for q in (value * unit, unit * value, value / unit, Quantity(value, unit)):
        assert isinstance(q, Quantity)
        assert q.value is value
    assert (value / unit).unit == unit**-1
    assert_quantity(
        unit / value if np.all(value != 0) else unit / (value + 1),
        1 / (value if np.all(value != 0) else value + 1),
        unit,
    )


@pytest.mark.parametrize(
    'value',
    [
        np.array(['m']),
        np.array([object()]),
        np.array([True]),
        np.array(['2020'], dtype='datetime64[Y]'),
        np.zeros(2, dtype=[('x', 'f8')]),
        np.ma.array([1, 2]),
        np.arange(2).view(np.recarray),
        [1, 2],
        (1, 2),
    ],
)
def test_unsupported_magnitudes(value):
    with pytest.raises(TypeError):
        Quantity(value, 'm')


def test_no_implicit_coercion_or_unit_removal():
    for value in (1, np.arange(3), np.array(1)):
        for unit in ('', 'm', 'km/m'):
            for convert in (np.array, np.asarray):
                with pytest.raises(TypeError, match='Implicit unit removal'):
                    convert(Quantity(value, unit))
    q = Quantity(np.arange(3), 'm')
    np.testing.assert_array_equal(np.asarray(q.value_in('cm')), [0, 100, 200])


def test_views_index_iteration_and_metadata():
    a = np.arange(12).reshape(3, 4)[:, ::2]
    a.flags.writeable = False
    q = Quantity(a, 'm')
    assert q.shape == (3, 2) and q.ndim == 2 and q.size == 6
    assert q.dtype == a.dtype
    assert isinstance(q[0, 0].value, np.integer)
    assert np.shares_memory(q[0].value, a)
    assert not q[0].value.flags.writeable
    assert np.shares_memory(q.T.value, a)
    assert np.shares_memory(np.transpose(q).value, a)
    assert_quantity(q.flatten(), a.flatten(), 'm')
    assert not np.shares_memory(q.flatten().value, a)
    assert len(q) == 3
    assert len(list(q)) == 3
    assert_quantity(list(q)[1], a[1], 'm')
    with pytest.raises(TypeError):
        q[0] = q[1]
    for scalar in (np.array(2), np.int64(2)):
        with pytest.raises(TypeError):
            iter(Quantity(scalar, 'm'))
    with pytest.raises(AttributeError):
        _ = Quantity(1, 'm').shape


def test_scaling_identity_rational_irrational_dtype_and_storage():
    a = np.arange(4, dtype=np.int16)
    q = Quantity(a, 'm')
    assert q.to('m').value is a
    np.testing.assert_array_equal(q.value_in('cm'), np.multiply(a, 100))
    assert q.value_in('cm').dtype == np.multiply(a, 100).dtype
    np.testing.assert_array_equal(q.value_in('km'), np.multiply(a, 0.001))
    assert q.value_in('km').dtype == np.multiply(a, 0.001).dtype
    assert not np.shares_memory(q.value_in('km'), a)
    for dtype in (np.float32, np.float64, np.longdouble, np.complex64):
        value = a.astype(dtype)
        converted = Quantity(value, PI * Unit('m')).value_in('m')
        np.testing.assert_allclose(converted, np.multiply(value, float(PI)))
        assert converted.dtype == np.multiply(value, float(PI)).dtype
    np.testing.assert_array_equal(a, np.arange(4))
    with pytest.raises(DimensionError):
        q.to('s')
    with pytest.raises(OverflowError):
        Quantity(np.array([1], dtype=np.int8), 'km').to('mm')


def test_exact_scale_stays_in_unit_algebra():
    q = Quantity(np.arange(3), 'cm')
    for result in (q * Factor(2), Factor(2) * q, q / PI):
        assert result.value is q.value
        assert isinstance(result.unit, ScaledUnit)
    assert_quantity((q * Factor(2)).to('m'), [0, 0.02, 0.04], 'm')
    assert (q * Factor(2)).unit.scale == Factor(2)


@pytest.mark.parametrize('operation', [np.add, np.subtract])
def test_addition_broadcast_and_scaled_dimensionless(operation):
    a = Quantity(np.array([[1], [2]]), 'm')
    b = Quantity(np.array([0, 100, 200]), 'cm')
    assert_quantity(operation(a, b), operation(a.value, b.value / 100), 'm')
    assert_quantity(operation(b, a), operation(b.value, a.value * 100), 'cm')
    q = Quantity(np.array([1.0, 2.0]), 'km/m')
    assert_quantity(q + 1, [1.001, 2.001], q.unit)
    assert_quantity(1 + q, [1.001, 2.001], q.unit)
    assert_quantity(1 - q, [-0.999, -1.999], q.unit)
    for other in (1, np.array([1]), Quantity(np.array([1]), 's')):
        with pytest.raises(DimensionError):
            operation(a, other)


def test_operators_and_matmul():
    a = Quantity(np.array([1.0, 2.0]), 'm')
    b = Quantity(np.array([2.0, 4.0]), 's')
    assert_quantity(a * b, [2, 8], 'm s')
    assert_quantity(a / b, [0.5, 0.5], 'm/s')
    assert_quantity(2 / a, [2, 1], '1/m')
    assert_quantity(np.array([2, 3]) * a, [2, 6], 'm')
    assert_quantity(a @ b, 10, 'm s')
    assert_quantity(np.array([2, 3]) @ a, 8, 'm')
    assert_quantity(-a, [-1, -2], 'm')
    assert_quantity(+a, [1, 2], 'm')
    assert_quantity(abs(-a), [1, 2], 'm')
    original = a
    a += Quantity(np.array([100, 200]), 'cm')
    assert a is not original
    np.testing.assert_array_equal(original.value, [1, 2])


@pytest.mark.parametrize(
    'operation', [np.equal, np.not_equal, np.less, np.less_equal, np.greater, np.greater_equal]
)
def test_comparison_broadcast_mixed_domain(operation):
    a = Quantity(np.array([[1], [2]]), 'm')
    b = Quantity(np.array([100.0, 200.0]), 'cm')
    np.testing.assert_array_equal(operation(a, b), operation(a.value, b.value * 0.01))
    np.testing.assert_array_equal(operation(Quantity(1, 'm'), b), operation(1, b.value * 0.01))
    result = (
        operation(a, Quantity(np.array([1, 2]), 's'))
        if operation in (np.equal, np.not_equal)
        else None
    )
    if result is not None:
        assert result.shape == (2, 2)
        assert np.all(result == (operation is np.not_equal))
    else:
        with pytest.raises(DimensionError):
            operation(a, Quantity(np.array([1, 2]), 's'))


def test_python_comparisons_and_hash_truth():
    q = Quantity(np.array([1, 2]), 'm')
    np.testing.assert_array_equal(q == Quantity(100, 'cm'), [True, False])
    np.testing.assert_array_equal(Quantity(100, 'cm') != q, [False, True])
    np.testing.assert_array_equal(q < Quantity(150, 'cm'), [True, False])
    for value in (np.array([1, 2]), np.array(0), np.int64(2)):
        with pytest.raises(TypeError, match='unhashable'):
            hash(Quantity(value, 'm'))
    with pytest.raises(ValueError):
        bool(q)
    assert bool(Quantity(np.array(2), 'm'))
    assert type(Quantity(1, 'm') == Quantity(100, 'cm')) is bool


@pytest.mark.parametrize(
    'operation,power', [(np.square, 2), (np.sqrt, Fraction(1, 2)), (np.cbrt, Fraction(1, 3))]
)
def test_exact_root_units(operation, power):
    q = Quantity(np.array([1.0, 8.0]), Factor(2) * Unit('m'))
    assert_quantity(operation(q), operation(q.value), q.unit**power)


def test_power_uniform_exponent_and_scaled_dimensionless():
    q = Quantity(np.array([1.0, 4.0]), 'm2')
    assert_quantity(q ** Fraction(1, 2), [1, 2], 'm')
    assert_quantity(q ** Decimal('.5'), [1, 2], 'm')
    assert_quantity(np.power(q, Quantity(np.array(0.5), '')), [1, 2], 'm')
    assert_quantity(
        np.power(Quantity(np.array([2.0, 3.0]), 'km/m'), np.array([1, 2])), [2000, 9000000], ''
    )
    with pytest.raises(TypeError):
        np.power(q, np.array([1, 2]))
    with pytest.raises(DimensionError):
        np.power(q, Quantity(2, 's'))
    with pytest.raises(TypeError):
        q**np.nan


@pytest.mark.parametrize('operation', [np.sin, np.cos, np.exp, np.log])
def test_dimensionless_functions_normalize_scale(operation):
    q = Quantity(np.array([1.0, 2.0]), 'km/m')
    with np.errstate(over='ignore'):
        assert_quantity(operation(q), operation(q.value * 1000), '')
    with pytest.raises(DimensionError):
        operation(Quantity(q.value, 'm'))
    assert_quantity(
        np.sin(Quantity(np.array([0.0, 90.0, 180.0]), 'deg')),
        np.sin(np.deg2rad([0.0, 90.0, 180.0])),
        '',
    )


@pytest.mark.parametrize('operation', [np.sum, np.mean, np.std, np.median, np.var, np.cumsum])
def test_reductions(operation):
    q = Quantity(np.arange(6.0).reshape(2, 3), Factor(2) * Unit('cm'))
    assert_quantity(
        operation(q, axis=0),
        operation(q.value, axis=0),
        q.unit**2 if operation is np.var else q.unit,
    )
    assert_quantity(operation(q), operation(q.value), q.unit**2 if operation is np.var else q.unit)
    if operation in (np.var, np.std):
        assert_quantity(
            operation(q, axis=1, ddof=1, keepdims=True),
            operation(q.value, axis=1, ddof=1, keepdims=True),
            q.unit**2 if operation is np.var else q.unit,
        )


def test_empty_zero_dim_and_ufunc_methods():
    q = Quantity(np.array([], dtype=float), 'm')
    assert_quantity(np.sum(q), 0, 'm')
    assert (q == Quantity(1, 'm')).shape == (0,)
    q = Quantity(np.array(2.0), 'm')
    assert_quantity(np.add.reduce(q), 2, 'm')
    assert_quantity(np.add.accumulate(Quantity(np.array([1, 2]), 'm')), [1, 3], 'm')
    with pytest.raises(TypeError):
        np.add.at(q, 0, q)
    for method in (np.multiply.reduce, np.multiply.accumulate, np.add.outer):
        with pytest.raises(TypeError):
            method(q, q) if method == np.add.outer else method(q)


def test_reshape_join_predicates_and_positions():
    q = Quantity(np.arange(6.0), 'm')
    reshaped = np.reshape(q, (2, 3))
    assert np.shares_memory(reshaped.value, q.value)
    assert_quantity(np.stack((q, Quantity(q.value * 100, 'cm'))), np.stack((q.value, q.value)), 'm')
    assert_quantity(np.concatenate((q, Quantity(q.value * 100, 'cm'))), np.tile(q.value, 2), 'm')
    with pytest.raises(DimensionError):
        np.stack((q, q.value))
    np.testing.assert_array_equal(np.isfinite(q), np.isfinite(q.value))
    np.testing.assert_array_equal(np.argmin(reshaped, axis=1), [0, 0])
    assert np.argmax(q) == 5


def test_close_symmetric_vs_numpy_and_units():
    a = Quantity(np.array([10.0, 0.0, np.inf, np.nan]), 'm')
    b = Quantity(np.array([900.0, 0.0, np.inf, np.nan]), 'cm')
    np.testing.assert_array_equal(a.isclose(b, rel_tol=0.1), [True, True, True, False])
    np.testing.assert_array_equal(b.isclose(a, rel_tol=0.1), [True, True, True, False])
    np.testing.assert_array_equal(
        np.isclose(a, b, rtol=0.1, atol=Quantity(0, 'm')), [False, True, True, False]
    )
    np.testing.assert_array_equal(
        np.isclose(a, b, 0.1, Quantity(0, 'm'), True), [False, True, True, True]
    )
    assert np.allclose(a, a, atol=Quantity(0, 'm'), equal_nan=True)
    with pytest.raises(DimensionError):
        np.isclose(a, b)  # nonzero bare default atol has no implicit metre unit
    np.testing.assert_array_equal(
        a.isclose(b, rel_tol=Quantity(np.array([0.1]), '')), [True, True, True, False]
    )
    np.testing.assert_array_equal(
        Quantity(np.array([1, 2], dtype=np.uint8), 'm').isclose(
            Quantity(np.array([2, 1], dtype=np.uint8), 'm'), rel_tol=0
        ),
        [False, False],
    )


@pytest.mark.parametrize('tolerance', [np.array([-0.1]), np.array([np.inf]), np.array([np.nan])])
def test_close_bad_tolerances(tolerance):
    q = Quantity(np.arange(3.0), 'm')
    with pytest.raises(ValueError):
        q.isclose(q, rel_tol=tolerance)
    with pytest.raises(ValueError):
        q.isclose(q, abs_tol=Quantity(tolerance, 'm'))
    with pytest.raises(DimensionError):
        q.isclose(q, abs_tol=0.1)
    with pytest.raises(DimensionError):
        q.isclose(q, rel_tol=Quantity(0.1, 'm'))


def test_out_mask_and_unsupported_functions_do_not_mutate():
    q = Quantity(np.arange(3.0), 'm')
    buffer = np.full(3, -123.0)
    target = Quantity(buffer, 'm')
    for out in (buffer, target):
        with pytest.raises(TypeError):
            np.add(q, q, out=out)
        with pytest.raises(TypeError):
            np.sum(q, out=out)
        with pytest.raises(TypeError):
            np.sum(q, None, None, out)
        with pytest.raises(TypeError):
            np.concatenate((q, q), out=out)
    np.testing.assert_array_equal(buffer, [-123] * 3)
    result = np.add(q, q, where=np.array([True, False, True]))
    np.testing.assert_array_equal(result.value[[0, 2]], [0, 4])
    with pytest.raises(TypeError):
        np.add(q, q, where=Quantity(np.array([1, 0, 1]), ''))
    for f in (np.prod, np.linalg.norm, np.sort):
        with pytest.raises(TypeError):
            f(q)
    with pytest.raises(TypeError):
        np.sum(q, initial=1)
    with pytest.raises(TypeError):
        np.median(q, overwrite_input=True)
    with pytest.raises(TypeError):
        q + Decimal(1)
    with pytest.raises(TypeError):
        q * np.ma.array([1, 2, 3])


@pytest.mark.parametrize('numpy_first', [True, False])
def test_import_order(numpy_first):
    lines = ['import numpy as np', 'from unitx import Unit, Quantity']
    if not numpy_first:
        lines.reverse()
    script = '\n'.join(lines) + "\na = np.arange(3)\nq = a * Unit('m')\nassert q.value is a\n"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    subprocess.run([sys.executable, '-c', script], env=env, check=True, timeout=20)


def test_core_does_not_import_numpy():
    src = Path(__file__).resolve().parents[1] / 'src'
    script = f"""
import sys
sys.path.insert(0, {str(src)!r})
from unitx import Quantity, Unit
from decimal import Decimal
assert (Quantity(Decimal('1'), 'm') + Quantity(1, 'm')).value == Decimal('2')
assert (2 * Unit('m')).value == 2
assert 'numpy' not in sys.modules
"""
    subprocess.run([sys.executable, '-S', '-c', script], check=True, timeout=20)


def test_unknown_protocol_inputs_allow_other_dispatch():
    q = Quantity(np.arange(2), 'm')

    class Foreign:
        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            return 'foreign dispatched'

    other = Foreign()
    assert q.__array_ufunc__(np.add, '__call__', q, other) is NotImplemented
    assert np.add(q, other) == 'foreign dispatched'
    assert q + other == 'foreign dispatched'
    assert q * other == 'foreign dispatched'
    assert q.__array_function__(np.sum, (Quantity, Foreign), (q,), {}) is NotImplemented
    assert q.__array_ufunc__(np.multiply, 'reduce', q) is NotImplemented
    with pytest.raises(TypeError, match='Unsupported array backend'):
        Quantity(other, 'm')


def test_tolerance_broadcast_scale_nan_and_empty():
    q = Quantity(np.array([[1.0], [2.0]]), 'm')
    other = Quantity(np.array([100.0, 201.0]), 'cm')
    tol = Quantity(np.array([0.0, 2.0]), 'cm')
    expected = [[True, False], [False, True]]
    np.testing.assert_array_equal(q.isclose(other, rel_tol=0, abs_tol=tol), expected)
    np.testing.assert_array_equal(other.isclose(q, rel_tol=0, abs_tol=tol), expected)
    np.testing.assert_array_equal(np.isclose(q, other, rtol=0, atol=tol), expected)
    assert Quantity(np.array([], dtype=float), 'm').isclose(Quantity(1, 'm')).size == 0
    assert_quantity(Quantity(np.array([4.0]), 'km/m') ** Fraction(1, 2), np.sqrt([4000.0]), '')


def test_numpy_scalars_and_zero_dim_outputs():
    for value in (np.float32(1), np.array(1, dtype=np.float32)):
        q = Quantity(value, 'm')
        assert_quantity(q.to('cm'), np.multiply(value, 100), 'cm')
        assert np.shape((q == Quantity(1, 'm'))) == ()
        assert q.isclose(Quantity(1, 'm'))
    q = Quantity(np.array([1, 2], dtype=np.int64), 'm')
    for operation in (np.sum, np.std, np.cumsum):
        with pytest.raises(TypeError):
            operation(q, dtype=object)
    for convert in (np.array, np.asarray):
        with pytest.raises(TypeError, match='Implicit unit removal'):
            convert(q, dtype=float, copy=True)


def test_scalar_paths_still_do_not_import_numpy_for_rejected_matmul():
    src = Path(__file__).resolve().parents[1] / 'src'
    script = f"""
import sys
sys.path.insert(0, {str(src)!r})
from unitx import Quantity
q = Quantity(1, 'm')
assert q.__matmul__(q) is NotImplemented
assert 'numpy' not in sys.modules
"""
    subprocess.run([sys.executable, '-S', '-c', script], check=True, timeout=20)
