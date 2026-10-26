"""Useful remaining dispatch paths and explicit compatibility boundaries."""

from typing import Any

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx.exceptions import DimensionError
from unitx.factor import Factor, pi

from .conftest import ArrayQuantity

np = pytest.importorskip('numpy')


def test_block_alignment() -> None:
    q = Quantity(np.arange(4.0).reshape(2, 2), Factor(2) * pi * Unit('cm'))
    result = np.block([[q, q.to('m')], [q.to('m'), q]])
    assert result.unit == q.unit
    np.testing.assert_allclose(result.value, np.block([[q.value, q.value], [q.value, q.value]]))
    with pytest.raises(DimensionError):
        np.block([[q, Quantity(q.value, 's')]])


def test_block_points() -> None:
    p = AffineQuantity(np.ones((2, 2)), 'degC')
    result = np.block([[p, p.to('K')]])
    assert result.unit == p.unit and type(result) is AffineQuantity
    np.testing.assert_allclose(result.value, 1)


@pytest.mark.parametrize('copy', [True, False])
@pytest.mark.parametrize(
    'family,unit', [(Quantity, 'm'), (AffineQuantity, 'degC'), (LogQuantity, 'dBm')]
)
def test_nan_to_num_storage(family: Any, unit: str, copy: bool) -> None:
    q = family(np.array([np.nan, np.inf, -np.inf, 2.0]), unit)
    result = np.nan_to_num(q, copy=copy, nan=family(3.0, unit), posinf=4.0, neginf=-4.0)
    assert type(result) is family and result.unit == q.unit
    np.testing.assert_array_equal(result.value, [3, 4, -4, 2])
    assert (result is q) == (not copy)
    if copy:
        assert not np.shares_memory(result.value, q.value)
        assert np.isnan(q.value[0])


def test_nan_to_num_invalid() -> None:
    q = Quantity(np.array([np.nan, np.inf]), 'm')
    with pytest.raises(DimensionError):
        np.nan_to_num(q, copy=False, nan=Quantity(1.0, 's'))
    np.testing.assert_array_equal(q.value, [np.nan, np.inf])
    q.value.flags.writeable = False
    with pytest.raises(ValueError):
        np.nan_to_num(q, copy=False)


def test_ediff1d_points() -> None:
    q = AffineQuantity(np.array([[1.0, 3.0], [6.0, 10.0]]), 'degC')
    result = np.ediff1d(q, to_begin=Quantity(2.0, 'K'), to_end=Quantity(3.0, 'K'))
    assert type(result) is Quantity and result.unit == q.unit.delta_unit
    np.testing.assert_array_equal(result.value, [2, 2, 3, 4, 3])
    with pytest.raises(TypeError):
        np.ediff1d(q, to_begin=AffineQuantity(0.0, 'K'))
    with pytest.raises(DimensionError):
        np.ediff1d(q, to_begin=0.0)


@pytest.mark.parametrize('func', [np.cumulative_sum, np.cumulative_prod])
@pytest.mark.parametrize('include_initial', [False, True])
def test_cumulative_array_api(func: Any, include_initial: bool) -> None:
    q = Quantity(
        np.array([[100.0, 200.0], [300.0, 400.0]]), 'cm/m' if func is np.cumulative_prod else 'm'
    )
    result = func(q, axis=1, include_initial=include_initial)
    raw = q.value / 100 if func is np.cumulative_prod else q.value
    assert result.unit == (Unit() if func is np.cumulative_prod else q.unit)
    np.testing.assert_array_equal(result.value, func(raw, axis=1, include_initial=include_initial))
    output = Quantity(np.zeros(result.shape), result.unit)
    assert func(q, axis=1, include_initial=include_initial, out=output) is output
    np.testing.assert_array_equal(output.value, result.value)
    with pytest.raises(ValueError):
        func(q)
    with pytest.raises(TypeError):
        func(AffineQuantity(q.value, 'K'), axis=0)
    if func is np.cumulative_prod:
        with pytest.raises(DimensionError):
            func(Quantity(q.value, 'm'), axis=0)


def test_array_equality() -> None:
    q = Quantity(np.array([1.0, np.nan, 3.0]), 'm')
    assert np.array_equal(q, q.to('cm'), equal_nan=True)
    assert not np.array_equal(q, q.to('cm'))
    assert not np.array_equal(q, Quantity(q.value, 's'), equal_nan=True)
    assert np.array_equiv(Quantity(np.ones((2, 3)), 'm'), Quantity(100.0, 'cm'))
    point = AffineQuantity(np.array([10.0, 20.0]), 'degC')
    assert np.array_equal(point, point.to('K'))
    assert not np.array_equal(point, Quantity(point.value, 'K'))


@pytest.mark.parametrize('func', [np.isreal, np.iscomplex, np.isrealobj, np.iscomplexobj])
def test_complex_predicates(func: Any) -> None:
    for q in (
        Quantity(np.array([1.0, 2j, complex(np.nan, 1)]), 'm'),
        AffineQuantity(np.array([1.0, 2.0]), 'K'),
    ):
        np.testing.assert_array_equal(func(q), func(q.value))


def test_memory_predicates() -> None:
    q = Quantity(np.arange(6.0), 'm')
    assert np.shares_memory(q, q[::2])
    assert np.may_share_memory(q, q.value)
    assert not np.shares_memory(q, q.to('cm'))
    assert not np.may_share_memory(q, np.copy(q))


@pytest.mark.parametrize('func', [np.all, np.any])
def test_truth_reductions(func: Any) -> None:
    q = AffineQuantity(np.array([[0.0, 1.0], [1.0, np.nan]]), 'degC')
    mask = np.array([[True, False], [True, True]])
    result = func(q, axis=1, keepdims=True, where=mask)
    np.testing.assert_array_equal(result, func(q.value, axis=1, keepdims=True, where=mask))
    out = np.zeros((2, 1), dtype=bool)
    assert func(q, axis=1, keepdims=True, where=mask, out=out) is out
    np.testing.assert_array_equal(out, result)
    with pytest.raises(TypeError):
        func(q, where=Quantity(np.ones(2), ''))


def test_array_preparation(array_sample: ArrayQuantity) -> None:
    resized = np.resize(array_sample, (4, 3))
    assert type(resized) is type(array_sample) and resized.unit == array_sample.unit
    np.testing.assert_array_equal(resized.value, np.resize(array_sample.value, (4, 3)))
    result = np.astype(array_sample, np.float32)
    assert type(result) is type(array_sample) and result.unit == array_sample.unit
    assert result.value.dtype == np.dtype('f4')
    assert np.astype(array_sample, array_sample.value.dtype, copy=False).value is array_sample.value
    q = Quantity(
        np.array([0.0, 0.0, 1.0, 2.0, 0.0]),
        array_sample.unit.delta_unit
        if isinstance(array_sample, (AffineQuantity, LogQuantity))
        else array_sample.unit,
    )
    trimmed = np.trim_zeros(q)
    assert trimmed.unit == q.unit and np.shares_memory(trimmed.value, q.value)
    np.testing.assert_array_equal(trimmed.value, [1, 2])


def test_binary_float_helpers() -> None:
    q = Quantity(np.array([1.0, 2.0, 3.0], dtype=np.float32), Factor(2) * pi * Unit('cm'))
    mantissa, exponent = np.frexp(q)
    result = np.ldexp(mantissa, exponent)
    assert result.unit == q.unit and result.value.dtype == q.value.dtype
    np.testing.assert_array_equal(result.value, q.value)
    spaced = np.spacing(q)
    assert spaced.unit == q.unit
    np.testing.assert_array_equal(spaced.value, np.spacing(q.value))
    toward = np.nextafter(q, q.to('m'))
    np.testing.assert_array_equal(toward.value, np.nextafter(q.value, q.to('m').value_in(q.unit)))
    target = Quantity(np.empty(3), 'm')
    assert np.ldexp(mantissa, exponent, out=target) is target
    np.testing.assert_allclose(target.value, q.value_in('m'))
    with pytest.raises(DimensionError):
        np.ldexp(q, Quantity(np.ones(3, dtype=int), 's'))


def test_heaviside() -> None:
    q = Quantity(np.array([-1.0, 0.0, 1.0]), 'm')
    result = np.heaviside(q, Quantity(50.0, 'cm/m'))
    assert result.unit == Unit()
    np.testing.assert_array_equal(result.value, [0, 0.5, 1])
    with pytest.raises(DimensionError):
        np.heaviside(q, Quantity(1.0, 's'))


@pytest.mark.parametrize(
    'func,args',
    [
        (np.vander, ()),
        (np.polyfit, (np.arange(3.0), 1)),
        (np.linalg.slogdet, ()),
    ],
)
def test_heterogeneous_unit_outputs_invalid(func: Any, args: tuple[Any, ...]) -> None:
    with pytest.raises(TypeError):
        func(Quantity(np.eye(3) if func is np.linalg.slogdet else np.arange(3.0), 'm'), *args)


@pytest.mark.parametrize(
    'func', [np.linalg.matmul, np.linalg.outer, np.linalg.cross, np.linalg.tensordot]
)
def test_linalg_array_api_products(func: Any) -> None:
    a = Quantity(np.array([1.0, 2.0, 3.0]), 'm')
    b = Quantity(np.array([3.0, 2.0, 1.0]), 's')
    kwargs = {'axes': 1} if func is np.linalg.tensordot else {}
    result = func(a, b, **kwargs)
    assert result.unit == a.unit * b.unit
    np.testing.assert_array_equal(result.value, func(a.value, b.value, **kwargs))
    with pytest.raises(TypeError):
        func(AffineQuantity(a.value, 'K'), b, **kwargs)
    with pytest.raises(TypeError):
        func(a, b, out=Quantity(np.zeros(result.shape), result.unit), **kwargs)


def test_linalg_array_api_diagonals() -> None:
    q = Quantity(np.arange(9.0).reshape(3, 3), 'm')
    result = np.linalg.trace(q, offset=1, dtype=np.float32)
    assert result.unit == q.unit
    np.testing.assert_array_equal(
        result.value, np.linalg.trace(q.value, offset=1, dtype=np.float32)
    )
    p = AffineQuantity(q.value, 'degC')
    diagonal = np.linalg.diagonal(p, offset=-1)
    assert diagonal.unit == p.unit and np.shares_memory(diagonal.value, p.value)
    np.testing.assert_array_equal(diagonal.value, np.linalg.diagonal(p.value, offset=-1))
    with pytest.raises(TypeError):
        np.linalg.trace(p)


def test_choose_select_alignment() -> None:
    q = Quantity(np.array([1.0, 2.0, 3.0]), 'm')
    choices = [q, q.to('cm') * 2]
    selected = np.choose(np.array([0, 1, 0]), choices)
    assert selected.unit == q.unit
    np.testing.assert_allclose(selected.value, [1, 4, 3])
    target = Quantity(np.zeros(3), 'cm')
    assert np.choose(np.array([0, 1, 0]), choices, out=target) is target
    np.testing.assert_allclose(target.value, [100, 400, 300])
    selected = np.select([np.array([True, False, False])], [q], default=Quantity(50.0, 'cm'))
    np.testing.assert_array_equal(selected.value, [1, 0.5, 0.5])
    selected = np.select([np.array([True, False, False])], [q])
    np.testing.assert_array_equal(selected.value, [1, 0, 0])
    with pytest.raises(DimensionError):
        np.choose(np.array([0, 1, 0]), [q, Quantity(q.value, 's')])
    with pytest.raises(TypeError):
        np.select([Quantity(np.ones(3), '')], [q])


def test_select_points() -> None:
    p = AffineQuantity(np.array([10.0, 20.0, 30.0]), 'degC')
    selected = np.select([np.array([True, False, False])], [p], default=AffineQuantity(273.15, 'K'))
    assert type(selected) is AffineQuantity and selected.unit == p.unit
    np.testing.assert_allclose(selected.value, [10, 0, 0], atol=1e-12)


def test_unstack_views(array_sample: ArrayQuantity) -> None:
    result = np.unstack(array_sample, axis=1)
    expected = np.unstack(array_sample.value, axis=1)
    assert type(result) is tuple and len(result) == len(expected)
    for view, raw in zip(result, expected):
        assert type(view) is type(array_sample) and view.unit == array_sample.unit
        assert np.shares_memory(view.value, array_sample.value)
        np.testing.assert_array_equal(view.value, raw)


def test_i0_scaled_inputs() -> None:
    q = Quantity(np.array([0.0, 50.0, 100.0]), 'cm/m')
    result = np.i0(q)
    assert result.unit == Unit()
    np.testing.assert_allclose(result.value, np.i0(q.value / 100))
    with pytest.raises(DimensionError):
        np.i0(Quantity(q.value, 'm'))


@pytest.mark.parametrize('func', [np.isposinf, np.isneginf])
def test_infinity_predicates(func: Any) -> None:
    q = Quantity(np.array([np.inf, -np.inf, np.nan]), 'm')
    target = np.zeros(3, dtype=bool)
    assert func(q, out=target) is target
    np.testing.assert_array_equal(target, func(q.value))


def test_sort_complex_units() -> None:
    q = Quantity(np.array([1 + 2j, 1 + 1j, 0 + 3j]), 'm')
    result = np.sort_complex(q)
    assert result.unit == q.unit
    np.testing.assert_array_equal(result.value, np.sort_complex(q.value))


@pytest.mark.parametrize('name', ['matvec', 'vecmat'])
def test_matrix_vector_ufuncs(name: str) -> None:
    func = getattr(np, name, None)
    if func is None:
        pytest.skip('Matrix-vector ufuncs were added in NumPy 2.2')
    matrix = Quantity(np.array([[1.0 + 1j, 2.0], [3.0, 4.0]]), 'm')
    vector = Quantity(np.array([2.0 + 1j, 1.0]), 's')
    a, b = (matrix, vector) if name == 'matvec' else (vector, matrix)
    result = func(a, b)
    assert result.unit == a.unit * b.unit
    np.testing.assert_array_equal(result.value, func(a.value, b.value))
    target = Quantity(np.empty(result.shape, dtype=complex), result.unit)
    assert func(a, b, out=target) is target
    np.testing.assert_array_equal(target.value, result.value)
