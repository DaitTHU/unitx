"""Differences keep ordinate units; gradients divide by each coordinate unit."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING

import pytest

from unitx._numerics import Magnitude

np = pytest.importorskip('numpy')

from unitx import Quantity, Unit  # noqa: E402
from unitx.exceptions import DimensionError  # noqa: E402
from unitx.factor import Factor  # noqa: E402

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray


@pytest.mark.parametrize('n', [0, 1, 2, 5])
@pytest.mark.parametrize('axis', [0, -1])
def test_numpy_diff_orders(n: int, axis: int) -> None:
    value = np.arange(24.0).reshape(4, 6)[:, ::2]
    value.flags.writeable = False
    q = Quantity(value, Factor(2) * Unit('cm'))
    result = np.diff(q, n=n, axis=axis)
    assert result.unit == q.unit
    np.testing.assert_array_equal(result.value, np.diff(value, n=n, axis=axis))
    if n == 0:
        assert result is q


@pytest.mark.parametrize('dtype', [np.uint8, np.int16, np.float32, np.complex128])
def test_numpy_diff_dtype(dtype: DTypeLike) -> None:
    value = np.array([5, 2, 1, 4], dtype=dtype)
    result = np.diff(Quantity(value, 'm'))
    expected = np.diff(value)
    assert result.value.dtype == expected.dtype
    np.testing.assert_array_equal(result.value, expected)


@pytest.mark.parametrize('shape,axis', [((0,), 0), ((2, 0), -1), ((0, 3), 0)])
def test_numpy_diff_empty(shape: tuple[int, ...], axis: int) -> None:
    q = Quantity(np.empty(shape), 'm')
    result = np.diff(q, axis=axis)
    assert result.unit == q.unit
    assert result.shape == np.diff(q.value, axis=axis).shape


def test_numpy_diff_boundaries() -> None:
    q = Quantity(np.array([[1.0, 3, 6], [2, 4, 8]]), Factor(2) * Unit('m'))
    first = Quantity(np.array([[100.0], [200.0]]), 'cm')
    last = Quantity(20, 'm')
    result = np.diff(q, 1, -1, first, last)
    assert result.unit == q.unit
    np.testing.assert_array_equal(result.value, [[0.5, 2, 3, 4], [1, 2, 4, 2]])
    # Differencing and cumulative summation reconstruct the stored samples.
    restored = q[:, 0:1] + np.cumsum(np.diff(q, prepend=q[:, 0:1]), axis=1)
    np.testing.assert_array_equal(restored.value, q.value)
    np.testing.assert_array_equal(first.value, [[100], [200]])


def test_numpy_diff_dimensionless() -> None:
    q = Quantity(np.array([1.0, 2]), 'km/m')
    result = np.diff(q, prepend=500, append=Quantity(300000, 'cm/m'))
    assert result.unit == q.unit
    np.testing.assert_allclose(result.value, [0.5, 1, 1])
    result = np.diff(np.array([1.0, 2]), prepend=Quantity(50, 'cm/m'))
    assert result.unit == Unit()
    np.testing.assert_array_equal(result.value, [0.5, 1])


def test_numpy_diff_noop() -> None:
    q = Quantity(np.array(2.0), 'm')
    assert np.diff(q, n=0, axis=99, prepend=None) is q


@pytest.mark.parametrize('boundary', [0, Quantity(0, 's')])
@pytest.mark.parametrize('name', ['prepend', 'append'])
def test_numpy_diff_invalid_dimension(boundary: int | Quantity[Magnitude], name: str) -> None:
    q = Quantity(np.arange(3.0), 'm')
    with pytest.raises(DimensionError, match=f'{name} has an incompatible dimension'):
        np.diff(q, **{name: boundary})
    np.testing.assert_array_equal(q.value, [0, 1, 2])


@pytest.mark.parametrize('boundary', [True, [1], np.array([True]), np.array(['1']), None])
def test_numpy_diff_invalid_boundary(
    boundary: bool | list[int] | NDArray[np.generic] | None,
) -> None:
    with pytest.raises(TypeError):
        np.diff(Quantity(np.arange(3.0), 'm'), prepend=boundary)


@pytest.mark.parametrize(
    'options', [{'n': -1}, {'axis': 1}, {'prepend': Quantity(np.ones((2, 2)), 'm')}]
)
def test_numpy_diff_invalid_arguments(options: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        np.diff(Quantity(np.arange(3.0), 'm'), **options)
    with pytest.raises(ValueError, match='at least one dimensional'):
        np.diff(Quantity(np.array(2.0), 'm'))


@pytest.mark.parametrize('edge_order', [1, 2])
@pytest.mark.parametrize('spacing', [2, np.float32(2), np.array(2.0), Fraction(2), Decimal(2)])
def test_numpy_gradient_spacing(
    spacing: int | np.float32 | NDArray[np.generic] | Fraction | Decimal, edge_order: int
) -> None:
    q = Quantity(np.arange(5.0) ** 2, Factor(3) * Unit('cm'))
    result = np.gradient(q, Quantity(spacing, 'ms'), edge_order=edge_order)
    assert result.unit == q.unit / Unit('ms')
    np.testing.assert_allclose(result.value, np.gradient(q.value, 2, edge_order=edge_order))
    other = np.gradient(q, spacing, edge_order=edge_order)
    assert other.unit == q.unit
    np.testing.assert_allclose(other.value, result.value)


@pytest.mark.parametrize('axis', [0, -1, (1,), (1, 0), None])
def test_numpy_gradient_axes(axis: int | tuple[int, ...] | None) -> None:
    values = np.arange(48.0).reshape(6, 8)[::2, ::2]
    values.flags.writeable = False
    q = Quantity(values, 'V')
    result = np.gradient(q, Quantity(np.array(2.0), 's'), axis=axis)
    expected = np.gradient(values, 2, axis=axis)
    if isinstance(expected, tuple):
        assert isinstance(result, tuple)
        for actual, raw in zip(result, expected):
            assert actual.unit == Unit('V/s')
            np.testing.assert_allclose(actual.value, raw)
    else:
        assert result.unit == Unit('V/s')
        np.testing.assert_allclose(result.value, expected)


def test_numpy_gradient_coordinates() -> None:
    time = Quantity(np.array([0.0, 1, 3, 6]), Factor(2) * Unit('s'))
    position = Quantity(np.array([0.0, 1, 4]), 'cm')
    potential = Quantity(time.value[:, None] ** 2 + position.value[None, :] ** 2, 'V')
    dt, dx = np.gradient(potential, time, position, edge_order=2)
    assert dt.unit == potential.unit / time.unit
    assert dx.unit == potential.unit / position.unit
    np.testing.assert_allclose(
        dt.value, np.broadcast_to(2 * time.value[:, None], potential.shape), atol=1e-14
    )
    np.testing.assert_allclose(
        dx.value, np.broadcast_to(2 * position.value, potential.shape), atol=1e-14
    )
    dx2, dt2 = np.gradient(potential, position, time, axis=(1, 0), edge_order=2)
    np.testing.assert_array_equal(dx2.value, dx.value)
    np.testing.assert_array_equal(dt2.value, dt.value)
    assert dx2.unit == dx.unit and dt2.unit == dt.unit


@pytest.mark.parametrize('reverse', [False, True])
def test_numpy_gradient_physical(reverse: bool) -> None:
    time = Quantity(np.array([0.0, 1, 3, 6]), 's')
    position = Quantity(np.array([0.0, 2, 6, 12]), 'm')
    if reverse:
        time, position = time[::-1], position[::-1]
    velocity = np.gradient(position, time, edge_order=2)
    np.testing.assert_allclose(velocity.value, 2)
    converted = np.gradient(position.to('cm'), time.to('ms'), edge_order=2)
    np.testing.assert_allclose(converted.value_in(velocity.unit), velocity.value)
    displacement = np.trapezoid(velocity, time)
    assert displacement.isclose(position[-1] - position[0])


def test_numpy_gradient_mixed() -> None:
    values = np.arange(12.0).reshape(3, 4)
    dt, dx = np.gradient(values, Quantity(2, 's'), np.array([0.0, 1, 3, 6]))
    assert dt.unit == Unit('s^-1') and dx.unit == Unit()
    expected = np.gradient(values, 2, np.array([0.0, 1, 3, 6]))
    np.testing.assert_array_equal(dt.value, expected[0])
    np.testing.assert_array_equal(dx.value, expected[1])
    q = Quantity(values, 'km/m')
    ordinary = np.gradient(q)
    assert all(component.unit == q.unit for component in ordinary)
    scaled = np.gradient(q, Quantity(2, 'km/m'))
    assert all(component.unit == Unit() for component in scaled)
    np.testing.assert_array_equal(scaled[0].value, np.gradient(values, 2)[0])


@pytest.mark.parametrize('dtype', [np.uint8, np.int16, np.float32, np.complex128])
def test_numpy_gradient_dtype(dtype: DTypeLike) -> None:
    q = Quantity(np.array([5, 2, 1, 4], dtype=dtype), 'm')
    result = np.gradient(q)
    expected = np.gradient(q.value)
    assert result.value.dtype == expected.dtype
    np.testing.assert_array_equal(result.value, expected)


def test_numpy_gradient_empty_axes() -> None:
    assert np.gradient(Quantity(np.array(2.0), 'm')) == ()
    assert np.gradient(Quantity(np.empty((0, 3)), 'm'), axis=()) == ()
    result = np.gradient(Quantity(np.empty((0, 3)), 'm'), Quantity(2, 's'), axis=1)
    assert result.shape == (0, 3) and result.unit == Unit('m/s')


def test_numpy_gradient_axis_subset() -> None:
    q = Quantity(np.arange(60.0).reshape(3, 4, 5), 'J')
    dz, dx = np.gradient(q, Quantity(2, 's'), Quantity(3, 'm'), axis=(2, 0))
    assert dz.unit == Unit('J/s') and dx.unit == Unit('J/m')
    expected = np.gradient(q.value, 2, 3, axis=(2, 0))
    np.testing.assert_array_equal(dz.value, expected[0])
    np.testing.assert_array_equal(dx.value, expected[1])


def test_numpy_gradient_invalid_kernel() -> None:
    q = Quantity(np.arange(3.0), 'm')
    with np.errstate(all='raise'):
        with pytest.raises(FloatingPointError):
            np.gradient(q, Quantity(0, 's'))
    np.testing.assert_array_equal(q.value, [0, 1, 2])


@pytest.mark.parametrize('spacing', [True, [0, 1, 2], np.array([True, False]), np.array(['1'])])
def test_numpy_gradient_invalid_spacing(spacing: bool | list[int] | NDArray[np.generic]) -> None:
    with pytest.raises(TypeError):
        np.gradient(Quantity(np.arange(3.0), 'm'), spacing)


@pytest.mark.parametrize('spacing', [np.arange(4.0), np.ones((3, 1))])
def test_numpy_gradient_invalid_coordinates(spacing: NDArray[np.generic]) -> None:
    with pytest.raises(ValueError):
        np.gradient(Quantity(np.arange(3.0), 'm'), Quantity(spacing, 's'))


@pytest.mark.parametrize('options', [{'axis': 2}, {'axis': (0, 0)}, {'edge_order': 3}])
def test_numpy_gradient_invalid_options(options: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        np.gradient(Quantity(np.arange(3.0), 'm'), **options)


def test_numpy_gradient_invalid_size() -> None:
    with pytest.raises(ValueError, match='too small'):
        np.gradient(Quantity(np.array([1.0]), 'm'))
    with pytest.raises(TypeError, match='invalid number'):
        np.gradient(Quantity(np.ones((3, 4)), 'm'), Quantity(np.arange(3.0), 's'))


def test_numpy_differentiation_protocol() -> None:
    class Foreign:
        def __array_function__(
            self,
            func: Callable[..., object],
            types: tuple[type[object], ...],
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> str:
            return 'foreign dispatched'

    q = Quantity(np.arange(3.0), 'm')
    assert np.gradient(q, Foreign()) == 'foreign dispatched'
    assert np.diff(q, prepend=Foreign()) == 'foreign dispatched'
    for operation in (np.diff, np.gradient):
        with pytest.raises(TypeError):
            operation(q, out=q)
        with pytest.raises(TypeError):
            operation(q, where=True)
