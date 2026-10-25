"""Linear interpolation aligns coordinates and retains ordinate units."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING

import pytest

from unitx._numerics import Magnitude

np = pytest.importorskip('numpy')

from unitx import AffineQuantity, LogQuantity, Quantity, Unit  # noqa: E402
from unitx.exceptions import DimensionError  # noqa: E402
from unitx.factor import Factor  # noqa: E402

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray


@pytest.mark.parametrize('sample_unit', [Unit('A'), Unit('cm', scale=Factor(2)), Unit('km/m')])
@pytest.mark.parametrize('dtype', [np.int16, np.float32, np.complex128])
def test_numpy_interp_units(sample_unit: Unit, dtype: DTypeLike) -> None:
    storage = np.arange(8, dtype=dtype)
    values = storage[::2]
    if np.iscomplexobj(values):
        values += 1j
    values.flags.writeable = False
    xp = Quantity(np.array([0.0, 1, 3, 6]), Unit('s', scale=Factor(2)))
    query = np.array([[-500.0, 0, 1000], [4000, 10000, 14000]])
    result = np.interp(Quantity(query, 'ms'), xp, Quantity(values, sample_unit))
    expected = np.interp(query / 2000, xp.value, values)
    assert result.unit is sample_unit and result.shape == query.shape
    assert result.value.dtype == expected.dtype
    np.testing.assert_allclose(result.value, expected)
    np.testing.assert_array_equal(xp.value, [0, 1, 3, 6])
    np.testing.assert_array_equal(query, [[-500, 0, 1000], [4000, 10000, 14000]])


@pytest.mark.parametrize('x', [0.5, Fraction(1, 2), Decimal('0.5'), np.float32(0.5), np.array(0.5)])
def test_numpy_interp_scalar(
    x: float | Fraction | Decimal | np.float32 | NDArray[np.generic],
) -> None:
    result = np.interp(
        Quantity(x, 's'), Quantity(np.array([0, 1]), 's'), Quantity(np.array([0, 2]), 'm')
    )
    assert result.unit == Unit('m') and result.value == 1
    assert isinstance(result.value, np.number)


def test_numpy_interp_mixed() -> None:
    # Each of the three dispatched inputs can independently enter unitx.
    fp = Quantity(np.array([0.0, 2, 6]), 'm')
    assert np.interp(0.5, [0, 1, 3], fp).isclose(Quantity(1, 'm'))
    for x, xp in (
        (Quantity(50, 'cm/m'), [0, 1, 3]),
        (0.5, Quantity(np.array([0.0, 100, 300]), 'cm/m')),
    ):
        result = np.interp(x, xp, (0, 2, 6))
        assert result.unit == Unit() and result.value == 1
    np.testing.assert_array_equal(np.interp([2, 0.5], [0, 1, 3], fp).value, [4, 1])
    np.testing.assert_array_equal(np.interp([[], []], [0, 1, 3], fp).value, np.empty((2, 0)))


@pytest.mark.parametrize('left', [Quantity(-200, 'cm'), Quantity(np.array(-2.0), 'm')])
def test_numpy_interp_boundaries(left: Quantity[Magnitude]) -> None:
    xp = Quantity(np.array([0.0, 1]), 's')
    fp = Quantity(np.array([2.0, 4]), Unit('m', scale=Factor(2)))
    result = np.interp(
        Quantity(np.array([-1.0, 0, 0.5, 1, 2]), 's'), xp, fp, left, Quantity(1000, 'cm')
    )
    assert result.unit is fp.unit
    np.testing.assert_array_equal(result.value, [-1, 2, 3, 4, 5])
    np.testing.assert_array_equal(fp.value, [2, 4])


def test_numpy_interp_dimensionless_boundaries() -> None:
    fp = Quantity(np.array([1.0, 2]), 'km/m')
    result = np.interp([-1, 0, 1, 2], [0, 1], fp, left=500, right=Quantity(300000, 'cm/m'))
    assert result.unit == fp.unit
    np.testing.assert_allclose(result.value, [0.5, 1, 2, 3])


def test_numpy_interp_single_sample() -> None:
    fp = Quantity(np.array([1 + 2j]), 'V')
    result = np.interp([-1, 0, 1], [0], fp, left=Quantity(-1j, 'V'), right=Quantity(300j, 'cV'))
    np.testing.assert_array_equal(result.value, [-1j, 1 + 2j, 3j])
    assert result.unit == fp.unit


def test_numpy_interp_calculus() -> None:
    time = Quantity(np.array([0.0, 1, 3]), 's')
    current = Quantity(np.array([0.0, 2, 6]), 'A')
    grid = Quantity(np.array([0.0, 500, 1000, 2000, 3000]), 'ms')
    interpolated = np.interp(grid, time, current)
    np.testing.assert_allclose(interpolated.value, [0, 1, 2, 4, 6])
    assert np.trapezoid(interpolated, grid).isclose(Quantity(9, 'C'))
    np.testing.assert_allclose(np.gradient(interpolated, grid).value_in('A/s'), 2)
    np.testing.assert_array_equal(np.diff(interpolated).value, [1, 1, 2, 2])


def test_numpy_interp_nonfinite_queries() -> None:
    result = np.interp([np.nan, -np.inf, np.inf], [0, 1], Quantity(np.array([2, 4]), 'm'))
    np.testing.assert_allclose(result.value, [np.nan, 2, 4], equal_nan=True)


@pytest.mark.parametrize('coordinates', [[], 1, np.ones((2, 2))])
def test_numpy_interp_invalid_coordinate_shape(coordinates: object) -> None:
    with pytest.raises(ValueError, match='nonempty one-dimensional'):
        np.interp(0.5, coordinates, Quantity(np.array([0, 1]), 'm'))


@pytest.mark.parametrize(
    'coordinates',
    [
        [1, 0],
        [0, 0],
        [0, 2, 1],
        [0, np.nan],
        [0, np.inf],
        [-np.inf, 0],
        np.array([255, 0], dtype=np.uint8),
        np.array([2**53, 2**53 + 1], dtype=np.int64),
    ],
)
def test_numpy_interp_invalid_coordinate_order(coordinates: object) -> None:
    with pytest.raises(ValueError, match='finite, strictly increasing'):
        np.interp(0.5, coordinates, Quantity(np.array([0, 1]), 'm'))


@pytest.mark.parametrize(
    'x,xp',
    [
        (Quantity(0.5, 's'), np.array([0, 1])),
        (0, Quantity(np.array([0, 1]), 's')),
        (Quantity(0.5, 's'), Quantity(np.array([0, 1]), 'm')),
        (Quantity(0.5, 'dB'), Quantity(np.array([0, 1]), '')),
    ],
)
def test_numpy_interp_invalid_coordinate_dimension(x: object, xp: object) -> None:
    with pytest.raises(DimensionError, match='x and xp must have compatible dimensions'):
        np.interp(x, xp, Quantity(np.array([2, 4]), 'm'))


@pytest.mark.parametrize('name', ['left', 'right'])
@pytest.mark.parametrize('boundary', [0, Quantity(0, 's'), Quantity(0, 'dB')])
def test_numpy_interp_invalid_boundary_dimension(name: str, boundary: object) -> None:
    with pytest.raises(DimensionError, match=f'{name} has an incompatible dimension'):
        np.interp(0.5, [0, 1], Quantity(np.array([2, 4]), 'm'), **{name: boundary})


@pytest.mark.parametrize('name', ['left', 'right'])
@pytest.mark.parametrize('boundary', [True, [1], np.array([1]), Quantity(np.array([1]), 'm')])
def test_numpy_interp_invalid_boundary(name: str, boundary: object) -> None:
    with pytest.raises(TypeError, match='numeric scalar or scalar Quantity'):
        np.interp(0.5, [0, 1], Quantity(np.array([2, 4]), 'm'), **{name: boundary})


@pytest.mark.parametrize('name', ['x', 'xp', 'fp', 'left', 'right'])
@pytest.mark.parametrize('point', [AffineQuantity(1, 'K'), LogQuantity(1, 'dBm')])
def test_numpy_interp_invalid_point(name: str, point: object) -> None:
    inputs = {'x': 0.5, 'xp': np.array([0, 1]), 'fp': Quantity(np.array([2, 4]), 'm'), name: point}
    with pytest.raises(TypeError, match='linear quantities or differences'):
        np.interp(**inputs)


@pytest.mark.parametrize('name', ['x', 'xp', 'fp'])
@pytest.mark.parametrize(
    'value',
    [True, [True], np.array([1], dtype=object), np.ma.array([1]), np.array([1]).view(np.recarray)],
)
def test_numpy_interp_invalid_input(name: str, value: object) -> None:
    inputs = {
        'x': Quantity(0.5, ''),
        'xp': np.array([0, 1]),
        'fp': Quantity(np.array([2, 4]), 'm'),
        name: value,
    }
    with pytest.raises(TypeError):
        np.interp(**inputs)


@pytest.mark.parametrize('name', ['x', 'xp'])
def test_numpy_interp_invalid_complex_coordinate(name: str) -> None:
    inputs = {
        'x': 0.5,
        'xp': np.array([0, 1]),
        'fp': Quantity(np.array([2, 4]), 'm'),
        name: np.array([1j, 2j]),
    }
    with pytest.raises(TypeError):
        np.interp(**inputs)


@pytest.mark.parametrize('values', [np.array([]), np.ones(3), np.ones((2, 2)), np.array(1)])
def test_numpy_interp_invalid_sample_shape(values: NDArray[np.generic]) -> None:
    with pytest.raises(ValueError):
        np.interp(0.5, [0, 1], Quantity(values, 'm'))


@pytest.mark.parametrize('period', [0, 1, Quantity(1, 's')])
def test_numpy_interp_invalid_period(period: object) -> None:
    with pytest.raises(TypeError, match='period is not supported'):
        np.interp(0.5, [0, 1], Quantity(np.array([2, 4]), 'm'), period=period)


@pytest.mark.parametrize('options', [{'out': None}, {'dtype': float}, {'where': True}])
def test_numpy_interp_invalid_options(options: dict[str, object]) -> None:
    with pytest.raises(TypeError):
        np.interp(0.5, [0, 1], Quantity(np.array([2, 4]), 'm'), **options)


def test_numpy_interp_foreign_protocol() -> None:
    class Foreign:
        def __array_function__(
            self,
            func: Callable[..., object],
            types: tuple[type[object], ...],
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> str:
            return 'foreign dispatched'

    assert np.interp(Foreign(), [0, 1], Quantity(np.array([2, 4]), 'm')) == 'foreign dispatched'
