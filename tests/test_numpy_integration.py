"""Trapezoidal integrals combine ordinate and coordinate units."""

import pytest

np = pytest.importorskip('numpy')

from unitx import Quantity, Unit  # noqa: E402
from unitx.factor import Factor  # noqa: E402


@pytest.mark.parametrize('y_unit', [Unit('A'), Factor(2) * Unit('mA')])
@pytest.mark.parametrize('x_unit', [Unit('ms'), Factor(3) * Unit('s')])
def test_trapezoid_coordinate_units(y_unit, x_unit):
    current = Quantity(np.array([1.0, 2, 4]), y_unit)
    time = Quantity(np.array([0.0, 2, 5]), x_unit)
    charge = np.trapezoid(current, x=time)
    assert charge.unit == y_unit * x_unit
    assert charge.value == 12
    assert charge.isclose(Quantity(12, y_unit * x_unit))
    np.testing.assert_array_equal(current.value, [1, 2, 4])
    np.testing.assert_array_equal(time.value, [0, 2, 5])


def test_trapezoid_mixed_inputs():
    values = np.array([1.0, 2, 3])
    coordinates = np.array([0.0, 1, 2])
    assert np.trapezoid(Quantity(values, 'V'), coordinates).isclose(Quantity(4, 'V'))
    assert np.trapezoid(values, Quantity(coordinates, 'ms')).isclose(Quantity(4, 'ms'))
    result = np.trapezoid(Quantity(values, 'nm^-1'), Quantity(coordinates, 'nm'))
    assert result.unit == Unit() and result.value == 4


@pytest.mark.parametrize('spacing', [2, np.float32(2), np.array(2.0), Quantity(2, 's')])
def test_trapezoid_spacing(spacing):
    current = Quantity(np.array([1.0, 2, 3]), 'A')
    result = np.trapezoid(current, dx=spacing)
    assert result.value == 8
    assert result.unit == current.unit * (spacing.unit if isinstance(spacing, Quantity) else Unit())


@pytest.mark.parametrize('axis', [0, 1, -1])
@pytest.mark.parametrize('coordinate_rank', [1, 2])
def test_trapezoid_axes(axis, coordinate_rank):
    values = np.arange(12.0).reshape(3, 4)[:, ::-1]
    coordinates = np.arange(values.shape[axis], dtype=float)
    if coordinate_rank == 2:
        coordinates = np.broadcast_to(
            coordinates[:, None] if axis == 0 else coordinates, values.shape
        )
    result = np.trapezoid(Quantity(values, 'W'), Quantity(coordinates, 's'), axis=axis)
    assert result.unit == Unit('W s')
    np.testing.assert_array_equal(result.value, np.trapezoid(values, coordinates, axis=axis))


@pytest.mark.parametrize('size', [0, 1])
def test_trapezoid_empty_intervals(size):
    result = np.trapezoid(Quantity(np.ones(size), 'V'), dx=Quantity(2, 's'))
    assert result.value == 0 and result.unit == Unit('V s')


def test_trapezoid_oriented_coordinates():
    y = Quantity(np.array([1.0, 2, 4]), 'A')
    x = Quantity(np.array([0.0, 2, 5]), 's')
    forward = np.trapezoid(y, x)
    reverse = np.trapezoid(y[::-1], x[::-1])
    assert reverse.isclose(-forward)
    # Explicit coordinates make dx irrelevant, including its unit and type.
    assert np.trapezoid(y, x, dx=object()).isclose(forward)


def test_trapezoid_invalid_spacing():
    with pytest.raises(TypeError, match='dx must be'):
        np.trapezoid(Quantity(np.arange(3.0), 'm'), dx=Quantity(np.ones(2), 's'))


def test_trapezoid_invalid_shape():
    with pytest.raises(ValueError):
        np.trapezoid(Quantity(np.arange(3.0), 'm'), Quantity(np.arange(5.0), 's'))


def test_trapezoid_invalid_options():
    q = Quantity(np.arange(3.0), 'm')
    with pytest.raises(TypeError):
        np.trapezoid(q, out=q)


def test_trapezoid_foreign_protocol():
    class Foreign:
        def __array_function__(self, func, types, args, kwargs):
            return 'foreign dispatched'

    q = Quantity(np.arange(3.0), 'm')
    assert np.trapezoid(q, Foreign()) == 'foreign dispatched'
