"""Array coordinate operations and fixed-buffer safety."""

import pytest

from unitx.affinequantity import AffineQuantity
from unitx.exceptions import DimensionError
from unitx.quantity import Quantity
from unitx.si import si
from unitx.unit import Unit

np = pytest.importorskip('numpy')


def test_array_construction_and_views():
    a = np.arange(6.0).reshape(2, 3)
    for q in (
        a @ si.degC,
        a @ AffineQuantity(0, 'degC').unit,
        AffineQuantity(a, 'degC'),
    ):
        assert isinstance(q, AffineQuantity)
        assert q.value is a
        assert q.to('degC').value is a
        assert np.shares_memory(q[0].value, a)
        assert q.T.unit == AffineQuantity(0, 'degC').unit
        assert np.reshape(q, (3, 2)).unit == AffineQuantity(0, 'degC').unit
        with pytest.raises(TypeError):
            np.asarray(q)


def test_array_conversion_and_statistics():
    q = AffineQuantity(np.array([0.0, 20.0, 30.0]), 'dBm')
    np.testing.assert_allclose(q.value_in('W'), [0.001, 0.1, 1])
    assert np.mean(q).unit == AffineQuantity(0, 'dBm').unit
    assert np.median(q).unit == AffineQuantity(0, 'dBm').unit
    assert np.std(q).unit == si.dB
    assert np.var(q).unit == Unit('dB**2')
    np.testing.assert_allclose(np.std(q).value, np.std(q.value))
    t = AffineQuantity(np.array([0.0, 100.0]), 'degC')
    np.testing.assert_allclose(t.value_in('degF'), [32, 212])
    assert np.std(t).unit == si.K
    assert np.var(t).unit == Unit('K**2')
    joined = np.concatenate([t, AffineQuantity(np.array([32.0]), 'degF')])
    np.testing.assert_allclose(joined.value, [0, 100, 0])


def test_numpy_add_sub_and_comparison():
    q = AffineQuantity(np.array([0.0, 20.0]), 'dBm')
    np.testing.assert_allclose((q + 10 * si.dB).value, [10, 30])
    result = q - AffineQuantity(np.array([-30.0, -10.0]), 'dBW')
    assert result.unit == si.dB
    np.testing.assert_allclose(result.value, [0, 0])
    np.testing.assert_array_equal(q == Quantity(np.array([0.001, 0.1]), 'W'), [False, False])
    np.testing.assert_array_equal(q == q, [True, True])
    assert isinstance(q == Quantity(1, 'V'), np.ndarray)
    assert np.all(np.isclose(q, q, rtol=0, atol=Quantity(1, 'dB')))


@pytest.mark.parametrize(
    'func',
    [
        np.sum,
        np.cumsum,
        np.prod,
        np.exp,
        np.log,
        np.sin,
        np.negative,
        np.abs,
        lambda q: np.add.reduce(q),
        lambda q: np.modf(q),
        lambda q: np.frexp(q),
        lambda q: q * 2,
    ],
)
def test_array_forbidden(func):
    with pytest.raises(TypeError):
        func(AffineQuantity(np.array([1.0, 2.0]), 'degC'))


def test_staged_write_and_fixed_point_unit():
    a = np.array([0.0, 100.0])
    q = AffineQuantity(a, 'degC')
    out = np.add(q, Quantity(1, 'K'), out=q)
    assert out is q and q.value is a
    np.testing.assert_array_equal(a, [1, 101])
    q[:] = AffineQuantity(np.array([32.0, 212.0]), 'degF')
    np.testing.assert_allclose(a, [0, 100])
    before = a.copy()
    for op in (
        lambda: np.subtract(q, q, out=q),
        lambda: q.__setitem__(slice(None), Quantity(1, 'K')),
        lambda: np.std(q, out=q),
    ):
        with pytest.raises((TypeError, DimensionError)):
            op()
        np.testing.assert_array_equal(a, before)
    intq = AffineQuantity(np.array([0, 100]), 'degC')
    before = intq.value.copy()
    with pytest.raises(TypeError):
        np.add(intq, Quantity(0.5, 'K'), out=intq)
    np.testing.assert_array_equal(intq.value, before)


def test_inplace_and_logarithmic_degree():
    q = AffineQuantity(np.array([0.0, 100.0]), 'degC')
    original = q
    q += Quantity(1, 'K')
    assert q is original
    before = q.value.copy()
    with pytest.raises(DimensionError):
        q -= AffineQuantity(np.array([0.0, 0.0]), 'degC')
    np.testing.assert_array_equal(q.value, before)
    log = Quantity(np.array([1.0, 2.0]), 'dB')
    with pytest.raises(DimensionError):
        np.sin(log)
    with pytest.raises(DimensionError):
        np.add(log, 1)
    np.testing.assert_array_equal(np.square(log).value, [1, 4])
    assert np.square(log).unit == Unit('dB**2')
    with pytest.raises(DimensionError):
        np.add(log, log, out=Quantity(np.zeros(2), ''))
    assert np.power(log, 2).unit == Unit('dB**2')


def test_masked_scaling_does_not_evaluate_inactive_coordinates():
    q = AffineQuantity(np.array([0.0, 1e308]), 'degC')
    right = AffineQuantity(np.array([32.0, 1e308]), 'degF')
    target = AffineQuantity(np.array([7.0, 8.0]), 'degF')
    with np.errstate(over='raise', invalid='raise'):
        np.subtract(right, q, where=np.array([True, False]), out=Quantity(np.zeros(2), 'degR'))
        np.add(q, Quantity(np.array([1.0, 1e308]), 'K'), where=np.array([True, False]), out=target)
    np.testing.assert_allclose(target.value, [33.8, 8.0])


def test_predicates_comparison_masks_and_extrema():
    q = AffineQuantity(np.array([0.0, 20.0]), 'dBm')
    assert np.min(q).unit == AffineQuantity(0, 'dBm').unit
    assert np.max(q).value == 20
    assert np.min(q, initial=AffineQuantity(-1, 'dBm')).value == -1
    np.testing.assert_array_equal(Quantity(np.array([0.0, 1.0]), 'dB') == 0, [False, False])
    out = np.array([False, False])
    np.equal(q, q, out=out, where=np.array([True, False]))
    np.testing.assert_array_equal(out, [True, False])
    readonly = q.value.copy()
    readonly.flags.writeable = False
    point = AffineQuantity(readonly, 'dBm')
    assert point.to('dBm').value is readonly
    assert np.isfinite(point).all()
    with pytest.raises(ValueError):
        np.add(point, Quantity(1, 'dB'), out=point)


def test_logarithmic_reduction_and_hash_semantics():
    q = Quantity(np.array([1.0, 2.0]), 'dB')
    with pytest.raises(TypeError):
        hash(q)
    assert np.sum(q).unit == si.dB
    with pytest.raises(DimensionError):
        np.sum(q, initial=1)
    product = np.prod(q)
    assert product.unit == Unit('dB**2')
    with pytest.raises(DimensionError):
        np.prod(
            Quantity(np.ones((2, 2)), 'dB'), axis=1, where=np.array([[True, True], [True, False]])
        )


def test_affine_tolerance_arrays_participate_in_broadcasting():
    left = AffineQuantity(0, 'degC')
    right = AffineQuantity(1, 'degC')
    mask = left.isclose(right, rel_tol=np.zeros(3), abs_tol=Quantity(1, 'K'))
    np.testing.assert_array_equal(mask, [True, True, True])
    with pytest.raises(ValueError):
        AffineQuantity(np.zeros(2), 'degC').isclose(
            right, rel_tol=np.zeros(3), abs_tol=Quantity(1, 'K')
        )
    # Linear complex differences remain supported; absolute coordinates are real.
    np.testing.assert_array_equal(Quantity(np.array([0j]), 'K').value_in('degC'), [0j])
    with pytest.raises(TypeError):
        AffineQuantity(np.array([0j]), 'K')


def test_array_at_matrix_separation():
    values = np.array([0.0, 100.0])
    point = values @ si.degC
    delta = values * si.degC
    assert type(point) is AffineQuantity
    assert type(delta) is Quantity
    assert point.value is values and delta.value is values
    np.testing.assert_array_equal(point.value_in('degF'), [32, 212])
    np.testing.assert_array_equal(delta.value_in('degF'), [0, 180])
    left = Quantity(np.eye(2), 'm')
    right = Quantity(np.eye(2), 's')
    matrix = left @ right
    assert type(matrix) is Quantity
    assert matrix.unit == Unit('m s')
    np.testing.assert_array_equal(matrix.value, np.eye(2))
    with pytest.raises(TypeError):
        left @ si.degC


@pytest.mark.parametrize('value', [np.array(0), np.float64(0), np.arange(4.0)[::2]])
def test_array_at_magnitude_identity(value):
    point = value @ si.K
    assert type(point) is AffineQuantity
    assert point.value is value
    assert point.to('K').value is value


def test_array_at_output_semantics():
    values = np.array([0.0, 100.0])
    target = AffineQuantity(np.zeros(2), 'degF')
    result = np.matmul(values, si.degC, out=target)
    assert result is target
    np.testing.assert_array_equal(target.value, [32, 212])
    delta_target = Quantity(np.zeros(2), 'degF')
    with pytest.raises(DimensionError):
        np.matmul(values, si.degC, out=delta_target)
    np.testing.assert_array_equal(delta_target.value, [0, 0])


def test_descriptor_array_projection():
    values = np.array([1.0, 2.0])
    for delta in (Quantity(values, si.degC), values * si.degC, si.degC * values):
        assert type(delta) is Quantity
        assert delta.value is values
        assert delta.unit == Unit('degC')
    point = values @ si.degC
    assert type(point) is AffineQuantity
    assert point.unit is si.degC
    assert point.value is values
    target = Quantity(np.zeros(2), 'K')
    assert np.multiply(values, si.degC, out=target) is target
    np.testing.assert_array_equal(target.value, values)
    point_target = AffineQuantity(np.zeros(2), 'degC')
    with pytest.raises(DimensionError):
        np.multiply(values, si.degC, out=point_target)
    np.testing.assert_array_equal(point_target.value, [0, 0])
