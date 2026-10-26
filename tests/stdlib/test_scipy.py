"""Finite SciPy support, using generic numerical examples."""

from decimal import Decimal
from typing import Any

import pytest

from unitx import AffineQuantity, Quantity
from unitx.exceptions import DimensionError
from unitx.factor import Factor
from unitx.scipy import simpson
from unitx.si import si

np = pytest.importorskip('numpy')
special = pytest.importorskip('scipy.special')
integrate = pytest.importorskip('scipy.integrate')


@pytest.mark.parametrize(
    'argument', [0.2, 0.2 + 0.1j, np.array([0.2, 0.4]), np.array([0.2j, 0.4j])]
)
def test_scipy_jv(argument: Any) -> None:
    order = Quantity(np.array([[0.0], [1.0]]), Factor(2) * si.one)
    q = Quantity(argument, Factor(2) * si.one)
    result = special.jv(order, q)
    assert result.unit == si.one
    np.testing.assert_allclose(result.value, special.jv(order.value * 2, argument * 2))
    scalar = special.jv(0, 2 * si.one)
    np.testing.assert_allclose(scalar.value, special.jv(0, 2))


def test_scipy_jv_outputs() -> None:
    q = Quantity(np.array([0.2, 0.4]), '')
    target = Quantity(np.full(2, 7.0), 'km/m')
    assert special.jv(0, q, out=target, where=np.array([True, False])) is target
    np.testing.assert_allclose(target.value, [special.jv(0, 0.2) / 1000, 7])
    invalid = Quantity(np.full(2, 7), '')
    with pytest.raises(TypeError):
        special.jv(0, q, out=invalid)
    np.testing.assert_array_equal(invalid.value, 7)


@pytest.mark.parametrize(
    'order, argument', [(0, Quantity(1, 'm')), (Quantity(1, 's'), Quantity(1, ''))]
)
def test_scipy_jv_invalid_dimension(order: Any, argument: Any) -> None:
    with pytest.raises(DimensionError):
        special.jv(order, argument)


def test_scipy_jv_invalid_point() -> None:
    with pytest.raises(TypeError):
        special.jv(0, AffineQuantity(1, 'K'))


@pytest.mark.parametrize('count', [2, 5, 6])
@pytest.mark.parametrize('complex_samples', [False, True])
def test_scipy_simpson(count: int, complex_samples: bool) -> None:
    coordinates = np.linspace(0.0, 2.0, count) ** 1.5
    samples = np.stack((coordinates**2, coordinates**3))
    if complex_samples:
        samples = samples * (1 + 2j)
    x, y = Quantity(coordinates, 's'), Quantity(samples, 'W')
    result = simpson(y, x=x, axis=-1)
    assert result.unit == si.W * si.s
    np.testing.assert_allclose(result.value, integrate.simpson(samples, x=coordinates))
    converted = simpson(y.to(si.kW), x=x.to(si.ms))
    np.testing.assert_allclose(converted.value_in(si.J), result.value)
    # dx is not evaluated when explicit x is provided, as in SciPy.
    np.testing.assert_allclose(simpson(y, x=x, dx=Quantity(1, 'm')).value, result.value)


def test_scipy_simpson_spacing() -> None:
    y = Quantity(np.arange(5.0) ** 2, 'W')
    for spacing in (Quantity(Decimal('0.25'), 's'), Quantity(np.array(0.25), 's')):
        result = simpson(y, dx=spacing)
        assert result.unit == si.W * si.s
        np.testing.assert_allclose(result.value, integrate.simpson(y.value, dx=0.25))
    result = simpson(y.value, dx=2 * si.s)
    assert result.unit == si.s
    np.testing.assert_allclose(result.value, integrate.simpson(y.value, dx=2))


def test_scipy_simpson_invalid_samples() -> None:
    with pytest.raises(TypeError, match='requires numeric samples'):
        simpson([1.0, 2.0, 3.0])


@pytest.mark.parametrize('argument', ['y', 'x', 'dx'])
def test_scipy_simpson_invalid_point(argument: str) -> None:
    kwargs = {argument: AffineQuantity(1, 'K')}
    if argument != 'y':
        kwargs['y'] = Quantity(np.ones(3), 'W')
    with pytest.raises(TypeError, match='requires linear quantities'):
        simpson(**kwargs)


def test_scipy_simpson_invalid_spacing() -> None:
    with pytest.raises(TypeError, match='dx must be scalar'):
        simpson(Quantity(np.ones(3), 'W'), dx=np.ones(3))


@pytest.mark.parametrize('kwargs', [dict(dx=1j), dict(x=np.ones(3) * 1j)])
def test_scipy_simpson_invalid_complex_coordinates(kwargs: dict[str, Any]) -> None:
    with pytest.raises(TypeError, match='coordinates must be real'):
        simpson(Quantity(np.ones(3), 'W'), **kwargs)


@pytest.mark.parametrize('argument', ['x', 'dx'])
def test_scipy_simpson_invalid_coordinates(argument: str) -> None:
    with pytest.raises(TypeError, match='coordinates must be real numeric'):
        simpson(Quantity(np.ones(3), 'W'), **{argument: [1.0, 2.0, 3.0]})
