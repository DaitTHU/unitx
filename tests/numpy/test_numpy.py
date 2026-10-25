"""NumPy's numeric domain, unit rules, and deliberate failure boundaries."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from unitx._numerics import Array
from unitx.exceptions import DimensionError
from unitx.factor import Factor, pi
from unitx.quantity import Quantity
from unitx.unit import Unit

from .._notebooks import notebook_source

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike, NDArray


np = pytest.importorskip('numpy')


def assert_quantity(q: Quantity[Array], values: ArrayLike, unit: str | Unit) -> None:
    assert isinstance(q, Quantity)
    np.testing.assert_allclose(q.value, values)
    assert q.unit == Unit(unit) if isinstance(unit, str) else q.unit == unit
    assert np.asarray(q.value).dtype.kind in 'iufc'


@pytest.mark.parametrize('unit', [Unit('m'), Factor(2) * Unit('m')])
@pytest.mark.parametrize('value', [np.arange(4), np.array(3.0), np.float32(2), np.int64(4)])
def test_construction_preserves_entire_magnitude(
    unit: Unit, value: NDArray[np.generic] | np.float32 | np.int64
) -> None:
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
def test_unsupported_magnitudes(value: object) -> None:
    with pytest.raises(TypeError):
        Quantity(value, 'm')


def test_no_implicit_coercion_or_unit_removal() -> None:
    for value in (1, np.arange(3), np.array(1)):
        for unit in ('', 'm', 'km/m'):
            for convert in (np.array, np.asarray):
                with pytest.raises(TypeError, match='Implicit unit removal'):
                    convert(Quantity(value, unit))
    q = Quantity(np.arange(3), 'm')
    np.testing.assert_array_equal(np.asarray(q.value_in('cm')), [0, 100, 200])


def test_views_index_iteration_and_metadata() -> None:
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
    with pytest.raises(ValueError, match='read-only'):
        q[0] = q[1]
    for scalar in (np.array(2), np.int64(2)):
        with pytest.raises(TypeError):
            iter(Quantity(scalar, 'm'))
    with pytest.raises(AttributeError):
        _ = Quantity(1, 'm').shape


@pytest.mark.parametrize('value', [np.array([1], dtype=np.int16), np.array(1), np.float32(1)])
def test_quantity_numpy_result_types(value: NDArray[np.generic] | np.float32) -> None:
    q = Quantity(value, 'm')
    for actual, expected in (
        (q * 2, np.multiply(value, 2)),
        (q.to('cm'), np.multiply(value, 100)),
        (q.to('km'), np.multiply(value, 0.001)),
    ):
        assert type(actual.value) is type(expected)
        assert actual.value.dtype == expected.dtype
        np.testing.assert_array_equal(actual.value, expected)
    assert type(q.T.value) is type(value) and q.T.value.dtype == value.dtype
    assert type(q.flatten().value) is np.ndarray
    assert q.flatten().value.dtype == value.dtype


def test_scaling_identity_rational_irrational_dtype_and_storage() -> None:
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
        converted = Quantity(value, pi * Unit('m')).value_in('m')
        np.testing.assert_allclose(converted, np.multiply(value, float(pi)))
        assert converted.dtype == np.multiply(value, float(pi)).dtype
    np.testing.assert_array_equal(a, np.arange(4))
    with pytest.raises(DimensionError):
        q.to('s')
    with pytest.raises(OverflowError):
        Quantity(np.array([1], dtype=np.int8), 'km').to('mm')


def test_exact_scale_stays_in_unit_algebra() -> None:
    q = Quantity(np.arange(3), 'cm')
    for result in (q * Factor(2), Factor(2) * q, q / pi):
        assert result.value is q.value
        assert isinstance(result.unit, Unit)
    assert_quantity((q * Factor(2)).to('m'), [0, 0.02, 0.04], 'm')
    assert (q * Factor(2)).unit.scale == Factor(2)


@pytest.mark.parametrize('operation', [np.add, np.subtract])
def test_addition_broadcast_and_scaled_dimensionless(operation: np.ufunc) -> None:
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


def test_operators_and_matmul() -> None:
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
    assert a is original
    np.testing.assert_array_equal(original.value, [2, 4])


@pytest.mark.parametrize('value', [Decimal('1.25'), Fraction(5, 4)])
@pytest.mark.parametrize('dtype', [np.int32, np.float32, np.float64, np.complex64])
def test_quantity_numpy_scalar_mixing(value: Decimal | Fraction, dtype: DTypeLike) -> None:
    a = np.array([2, 4], dtype=dtype)
    q = Quantity(a, 'm')
    scalar = Quantity(value, 'm')
    for left, right in ((q, scalar), (scalar, q)):
        assert_quantity(left + right, a + float(value), 'm')
        assert_quantity(left * right, a * float(value), 'm2')
    assert_quantity(q - scalar, a - float(value), 'm')
    assert_quantity(scalar - q, float(value) - a, 'm')
    assert_quantity(q / scalar, a / float(value), '')
    assert_quantity(scalar / q, float(value) / a, '')
    for result in (q * value, value * q):
        assert_quantity(result, a * float(value), 'm')
        assert result.value.dtype == (a * float(value)).dtype
    np.testing.assert_array_equal(q == scalar, a == float(value))
    assert scalar.value is value
    np.testing.assert_array_equal(q.value, a)


def test_quantity_numpy_decimal_context() -> None:
    from unitx.si import SI

    constant = SI.h.value
    with localcontext() as context:
        context.prec = 2
        context.clear_flags()
        q = SI.h * Quantity(np.array([1.0, 2.0]), 'Hz')
        assert_quantity(q, float(constant) * np.array([1.0, 2.0]), 'J s Hz')
        assert not any(context.flags.values())
    assert SI.h.value is constant and isinstance(constant, Decimal)


@pytest.mark.parametrize('exponent', [Fraction(1, 3), Decimal('0.25')])
def test_quantity_numpy_exact_exponent(exponent: Fraction | Decimal) -> None:
    q = Quantity(np.array([64.0]), 'm')
    for power in (exponent, Quantity(exponent * 1000, 'm/km')):
        result = np.power(q, power)
        assert result.unit == Unit('m') ** exponent
        np.testing.assert_allclose(result.value, np.power(q.value, float(exponent)))


def test_quantity_numpy_scalar_output() -> None:
    q = Quantity(np.array([1.0, 2.0]), 'm')
    target = Quantity(np.zeros(2), 'cm')
    assert np.multiply(q, Decimal('0.25'), out=target) is target
    assert_quantity(target, [25, 50], 'cm')
    np.testing.assert_array_equal(q.value, [1, 2])
    np.testing.assert_array_equal(
        q.isclose(Quantity(Decimal('1'), 'm'), rel_tol=Decimal('0.01')), [True, False]
    )


def test_quantity_numpy_blackbody() -> None:
    from unitx.si import SI, si

    lam = si.um * np.linspace(0.01, 3, 100)[:, None]
    T = si.K * np.array([3000, 4000, 5000], dtype=np.float64)
    nu = SI.c / lam
    exponent = SI.h * nu / (SI.kB * T)
    occupation = np.expm1(exponent)
    radiance = (2 * SI.h * SI.c**2 / (lam**5 * occupation)).to('W/m3/sr')
    wavelength = lam.value_in('m')
    h, c, kB = float(SI.h.value), SI.c.value, float(SI.kB.value)
    expected = 2 * h * c**2 / (wavelength**5 * np.expm1(h * c / (wavelength * kB * T.value)))
    assert radiance.shape == (100, 3)
    assert_quantity(radiance, expected, 'W/m3/sr')
    assert np.all(np.isfinite(radiance.value)) and np.all(radiance.value >= 0)
    peaks = lam.value[:, 0][np.argmax(radiance.value, axis=0)]
    np.testing.assert_allclose(peaks, float(SI.b.value_in('um K')) / T.value, atol=0.02)


def test_numpy_blackbody_example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    monkeypatch.setattr(plt, 'show', lambda: None)
    try:
        path = Path(__file__).resolve().parents[2] / 'examples/blackbody.ipynb'
        result = {'__name__': '__main__'}
        exec(compile(notebook_source(path), str(path), 'exec'), result)
        radiance = result['radiance']
        assert isinstance(radiance, Quantity) and radiance.shape == (100, 3)
        lines = plt.gca().get_lines()
        assert len(lines) == 3
        for index, line in enumerate(lines):
            np.testing.assert_array_equal(line.get_ydata(), radiance.value[:, index])
        output = tmp_path / 'blackbody.png'
        plt.savefig(output)
        assert output.stat().st_size > 0
    finally:
        plt.close('all')


@pytest.mark.parametrize(
    'operation', [np.equal, np.not_equal, np.less, np.less_equal, np.greater, np.greater_equal]
)
def test_comparison_broadcast_mixed_domain(operation: np.ufunc) -> None:
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


def test_python_comparisons_and_hash_truth() -> None:
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
def test_exact_root_units(operation: np.ufunc, power: int | Fraction) -> None:
    q = Quantity(np.array([1.0, 8.0]), Factor(2) * Unit('m'))
    assert_quantity(operation(q), operation(q.value), q.unit**power)


def test_power_uniform_exponent_and_scaled_dimensionless() -> None:
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
def test_dimensionless_functions_normalize_scale(operation: np.ufunc) -> None:
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
def test_reductions(operation: Callable[..., object]) -> None:
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


def test_empty_zero_dim_and_ufunc_methods() -> None:
    q = Quantity(np.array([], dtype=float), 'm')
    assert_quantity(np.sum(q), 0, 'm')
    assert (q == Quantity(1, 'm')).shape == (0,)
    q = Quantity(np.array(2.0), 'm')
    assert_quantity(np.add.reduce(q), 2, 'm')
    assert_quantity(np.add.accumulate(Quantity(np.array([1, 2]), 'm')), [1, 3], 'm')
    with pytest.raises(TypeError):
        np.add.at(q, 0, q)
    for method in (np.multiply.accumulate, np.add.outer):
        with pytest.raises(TypeError):
            method(q, q) if method == np.add.outer else method(q)


def test_reshape_join_predicates_and_positions() -> None:
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


def test_close_symmetric_vs_numpy_and_units() -> None:
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
def test_close_bad_tolerances(tolerance: NDArray[np.generic]) -> None:
    q = Quantity(np.arange(3.0), 'm')
    with pytest.raises(ValueError):
        q.isclose(q, rel_tol=tolerance)
    with pytest.raises(ValueError):
        q.isclose(q, abs_tol=Quantity(tolerance, 'm'))
    with pytest.raises(DimensionError):
        q.isclose(q, abs_tol=0.1)
    with pytest.raises(DimensionError):
        q.isclose(q, rel_tol=Quantity(0.1, 'm'))


def test_unsupported_outputs_do_not_mutate() -> None:
    q = Quantity(np.arange(3.0), 'm')
    buffer = np.full(3, -123.0)
    target = Quantity(buffer, 'm')
    with pytest.raises(TypeError):
        np.add(q, q, out=buffer)
    with pytest.raises(TypeError):
        np.sum(q, out=buffer)
    with pytest.raises(ValueError, match='result shape'):
        np.sum(q, None, None, target)
    with pytest.raises(TypeError):
        np.concatenate((q, q), out=buffer)
    with pytest.raises(ValueError, match='result shape'):
        np.concatenate((q, q), out=target)
    np.testing.assert_array_equal(buffer, [-123] * 3)
    result = np.add(q, q, where=np.array([True, False, True]))
    np.testing.assert_array_equal(result.value[[0, 2]], [0, 4])
    with pytest.raises(TypeError):
        np.add(q, q, where=Quantity(np.array([1, 0, 1]), ''))
    with pytest.raises(TypeError):
        np.linalg.norm(q)
    with pytest.raises(DimensionError):
        np.sum(q, initial=1)
    with pytest.raises(TypeError):
        np.median(q, overwrite_input=True)
    with pytest.raises(TypeError):
        q * np.ma.array([1, 2, 3])


@pytest.mark.parametrize('tuple_out', [False, True])
def test_ufunc_out_conversion(tuple_out: bool) -> None:
    q = Quantity(np.array([1.0, 2.0, 3.0]), 'm')
    buffer = np.full(3, -1.0)
    unit = Factor(2) * Unit('cm')
    target = Quantity(buffer, unit)
    result = np.add(q, Quantity(100, 'cm'), out=(target,) if tuple_out else target)
    assert result is target and target.value is buffer and target.unit is unit
    assert_quantity(result, [100, 150, 200], unit)
    np.testing.assert_array_equal(q.value, [1, 2, 3])
    assert_quantity(np.add(q, q, out=(None,)), [2, 4, 6], 'm')


def test_ufunc_out_broadcast() -> None:
    # A zero-dimensional result can broadcast into a larger output buffer.
    target = Quantity(np.empty((2, 3)), 'cm')
    assert np.add(Quantity(1, 'm'), Quantity(2, 'm'), target) is target
    assert_quantity(target, np.full((2, 3), 300), 'cm')
    scalar = Quantity(np.empty(()), 'm')
    assert np.multiply(Quantity(2, 'm'), 3, out=scalar) is scalar
    assert scalar.value.shape == () and scalar.value == 6
    empty = Quantity(np.empty((0, 3)), 'm')
    assert np.negative(Quantity(np.empty((0, 3)), 'm'), out=empty) is empty


def test_ufunc_out_mask() -> None:
    target = Quantity(np.full(3, -123.0), 'cm')
    # Masked-out invalid values are neither evaluated nor scaled for writing.
    q = Quantity(np.array([1.0, 0.0, 3.0]), 'm')
    with np.errstate(divide='raise', invalid='raise'):
        assert (
            np.divide(q, np.array([1.0, 0.0, 1.0]), out=target, where=np.array([True, False, True]))
            is target
        )
    assert_quantity(target, [100, -123, 300], 'cm')
    assert np.add(q, q, out=target, where=False) is target
    assert_quantity(target, [100, -123, 300], 'cm')


@pytest.mark.parametrize('operation', [np.multiply, np.divide, np.sqrt, np.matmul])
def test_ufunc_out_unit_algebra(operation: np.ufunc) -> None:
    a = Quantity(np.array([[1.0, 4.0], [9.0, 16.0]]), 'cm2')
    b = Quantity(np.full((2, 2), 2.0), 's')
    expected = operation(a) if operation is np.sqrt else operation(a, b)
    target = Quantity(np.empty(expected.shape), expected.unit / Factor(2))
    result = operation(a, out=target) if operation is np.sqrt else operation(a, b, out=target)
    assert result is target
    np.testing.assert_allclose(target.value, expected.value * 2)


def test_ufunc_out_dimensionless() -> None:
    target = Quantity(np.zeros(2), 'km/m')
    assert np.exp(Quantity(np.zeros(2), ''), out=target) is target
    assert_quantity(target, [0.001, 0.001], 'km/m')
    with pytest.raises(TypeError):
        np.exp(Quantity(np.zeros(2), ''), out=np.empty(2))
    with pytest.raises(DimensionError, match='out has an incompatible dimension'):
        np.exp(Quantity(np.zeros(2), ''), out=Quantity(np.empty(2), 'm'))


@pytest.mark.parametrize('casting', ['same_kind', 'safe', 'no'])
def test_ufunc_out_casting(casting: str) -> None:
    q = Quantity(np.array([1.0, 2.0], dtype=np.float64), 'm')
    buffer = np.full(2, -123.0, dtype=np.float32)
    target = Quantity(buffer, 'm')
    if casting == 'same_kind':
        assert np.add(q, q, out=target, casting=casting) is target
        assert target.dtype == np.dtype('float32')
        assert_quantity(target, [2, 4], 'm')
    else:
        with pytest.raises(TypeError):
            np.add(q, q, out=target, casting=casting)
        np.testing.assert_array_equal(buffer, [-123, -123])


def test_ufunc_out_integer_casting() -> None:
    q = Quantity(np.array([1, 2], dtype=np.int64), 'cm')
    target = Quantity(np.full(2, -123, dtype=np.int64), 'm')
    with pytest.raises(TypeError):
        np.add(q, q, out=target)
    np.testing.assert_array_equal(target.value, [-123, -123])
    # Explicit unsafe casting follows NumPy's truncation rules.
    assert np.add(q, q, out=target, casting='unsafe') is target
    np.testing.assert_array_equal(target.value, [0, 0])
    floating = Quantity(np.empty(2), 'm')
    assert np.add(q, q, out=floating, dtype=np.float64) is floating
    assert_quantity(floating, [0.02, 0.04], 'm')


def test_ufunc_out_shared_storage() -> None:
    storage = np.arange(6.0)
    source = Quantity(storage[:-1], 'm')
    target = Quantity(storage[1:], 'cm')
    assert np.add(source, source, out=target) is target
    np.testing.assert_array_equal(storage, [0, 0, 200, 400, 600, 800])
    assert np.negative(target, out=target) is target
    np.testing.assert_array_equal(storage, [0, 0, -200, -400, -600, -800])
    a = Quantity(np.arange(4.0).reshape(2, 2), 'm')
    expected = a.value @ a.value
    assert np.matmul(a, a, out=Quantity(a.value, 'm2')).value is a.value
    np.testing.assert_array_equal(a.value, expected)


def test_ufunc_out_strided_buffer() -> None:
    storage = np.full((2, 6), -123.0)
    target = Quantity(storage[:, ::2], 'm')
    q = Quantity(np.arange(6.0).reshape(2, 3), pi * Unit('m'))
    assert np.positive(q, out=target) is target
    np.testing.assert_allclose(target.value, q.value * float(pi))
    np.testing.assert_array_equal(storage[:, 1::2], np.full((2, 3), -123.0))
    readonly = Quantity(np.arange(6.0).reshape(2, 3), 'm')
    readonly.value.flags.writeable = False
    assert np.add(readonly, readonly, out=target) is target
    assert_quantity(target, readonly.value * 2, 'm')


def test_ufunc_out_shared_mask() -> None:
    storage = np.array([True, False, True, False, True, False])
    target = storage[1:]
    mask = storage[:-1]
    expected = np.isfinite(np.array([1.0, 2.0, np.nan, 4.0, np.inf]))
    original, original_mask = target.copy(), mask.copy()
    q = Quantity(np.array([1.0, 2.0, np.nan, 4.0, np.inf]), 'm')
    assert np.isfinite(q, out=target, where=mask) is target
    np.testing.assert_array_equal(target, np.where(original_mask, expected, original))


def test_ufunc_out_invalid_scale() -> None:
    q = Quantity(np.array([1, 2], dtype=np.int8), 'km')
    target = Quantity(np.full(2, -123.0), 'mm')
    with pytest.raises(OverflowError):
        np.positive(q, out=target)
    np.testing.assert_array_equal(target.value, [-123] * 2)
    complex_q = Quantity(np.array([1 + 2j, 3 + 4j]), 'm')
    with pytest.raises(TypeError):
        np.add(complex_q, complex_q, out=target)
    np.testing.assert_array_equal(target.value, [-123] * 2)


def test_ufunc_out_invalid_buffer() -> None:
    q = Quantity(np.arange(3.0), 'm')
    for target in (Quantity(np.full(3, -123.0), 's'), Quantity(np.full(2, -123.0), 'm')):
        with pytest.raises((DimensionError, ValueError)):
            np.add(q, q, out=target)
        assert np.all(target.value == -123)
    readonly = Quantity(np.full(3, -123.0), 'm')
    readonly.value.flags.writeable = False
    with pytest.raises(ValueError, match='read-only'):
        np.add(q, q, out=readonly)
    with pytest.raises(TypeError, match='ndarray buffer'):
        np.add(q, q, out=Quantity(np.float64(1), 'm'))
    with pytest.raises(TypeError, match='ndarray buffer'):
        np.add(q, q, out=Quantity(1.0, 'm'))
    target = Quantity(np.full(3, -123.0), 'm')
    with pytest.raises(TypeError, match='boolean mask'):
        np.add(q, q, out=target, where=np.array([1, 0, 1]))
    with pytest.raises(ValueError):
        np.add(q, q, out=target, where=np.ones(2, dtype=bool))
    np.testing.assert_array_equal(target.value, [-123] * 3)


def test_ufunc_out_cast_error() -> None:
    q = Quantity(np.array([1.0, 1e300]), 'm')
    target = Quantity(np.full(2, -123.0, dtype=np.float32), 'm')
    with np.errstate(over='raise'):
        with pytest.raises(FloatingPointError):
            np.add(q, q, out=target)
    np.testing.assert_array_equal(target.value, [-123] * 2)


@pytest.mark.parametrize('operation', [np.isfinite, np.equal])
def test_ufunc_out_predicates(operation: np.ufunc) -> None:
    q = Quantity(np.array([1.0, np.inf, np.nan]), 'm')
    target = np.array([False, True, True])
    args = (q, q) if operation is np.equal else (q,)
    assert operation(*args, out=target, where=np.array([True, True, False])) is target
    np.testing.assert_array_equal(target, [True, operation is np.equal, True])
    with pytest.raises(TypeError):
        operation(*args, out=np.empty(3))
    with pytest.raises(TypeError):
        operation(*args, out=Quantity(np.empty(3), ''))


def test_ufunc_out_protocol() -> None:
    q = Quantity(np.arange(3.0), 'm')

    class Foreign:
        def __array_ufunc__(
            self, ufunc: np.ufunc, method: str, *inputs: object, **kwargs: object
        ) -> str:
            return 'foreign dispatched'

    assert np.add(q, q, out=Foreign()) == 'foreign dispatched'
    target = Quantity(np.full(3, -123.0), 'm')
    with pytest.raises(TypeError):
        np.multiply.accumulate(q, out=target)
    np.testing.assert_array_equal(target.value, [-123] * 3)


@pytest.mark.parametrize('numpy_first', [True, False])
def test_import_order(numpy_first: bool) -> None:
    lines = ['import numpy as np', 'from unitx import Unit, Quantity']
    if not numpy_first:
        lines.reverse()
    script = '\n'.join(lines) + "\na = np.arange(3)\nq = a * Unit('m')\nassert q.value is a\n"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / 'src'))
    subprocess.run([sys.executable, '-c', script], env=env, check=True, timeout=20)


def test_core_does_not_import_numpy() -> None:
    src = Path(__file__).resolve().parents[2] / 'src'
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


def test_unknown_protocol_inputs_allow_other_dispatch() -> None:
    q = Quantity(np.arange(2), 'm')

    class Foreign:
        def __array_ufunc__(
            self, ufunc: np.ufunc, method: str, *inputs: object, **kwargs: object
        ) -> str:
            return 'foreign dispatched'

    other = Foreign()
    assert q.__array_ufunc__(np.add, '__call__', q, other) is NotImplemented
    assert np.add(q, other) == 'foreign dispatched'
    assert q + other == 'foreign dispatched'
    assert q * other == 'foreign dispatched'
    assert q.__array_function__(np.sum, (Quantity, Foreign), (q,), {}) is NotImplemented
    assert q.__array_ufunc__(np.multiply, 'accumulate', q) is NotImplemented
    with pytest.raises(TypeError, match='Unsupported array backend'):
        Quantity(other, 'm')


def test_tolerance_broadcast_scale_nan_and_empty() -> None:
    q = Quantity(np.array([[1.0], [2.0]]), 'm')
    other = Quantity(np.array([100.0, 201.0]), 'cm')
    tol = Quantity(np.array([0.0, 2.0]), 'cm')
    expected = [[True, False], [False, True]]
    np.testing.assert_array_equal(q.isclose(other, rel_tol=0, abs_tol=tol), expected)
    np.testing.assert_array_equal(other.isclose(q, rel_tol=0, abs_tol=tol), expected)
    np.testing.assert_array_equal(np.isclose(q, other, rtol=0, atol=tol), expected)
    assert Quantity(np.array([], dtype=float), 'm').isclose(Quantity(1, 'm')).size == 0
    assert_quantity(Quantity(np.array([4.0]), 'km/m') ** Fraction(1, 2), np.sqrt([4000.0]), '')


def test_numpy_scalars_and_zero_dim_outputs() -> None:
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


def test_scalar_paths_still_do_not_import_numpy_for_rejected_matmul() -> None:
    src = Path(__file__).resolve().parents[2] / 'src'
    script = f"""
import sys
sys.path.insert(0, {str(src)!r})
from unitx import Quantity
q = Quantity(1, 'm')
assert q.__matmul__(q) is NotImplemented
assert 'numpy' not in sys.modules
"""
    subprocess.run([sys.executable, '-S', '-c', script], check=True, timeout=20)
