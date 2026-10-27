"""Explicit numerical exits and physical representation changes."""

from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from unitx import AffineQuantity, LogQuantity, Quantity, Unit
from unitx.exceptions import DimensionError
from unitx.factor import pi
from unitx.si import si


@pytest.mark.parametrize('value', [1, 1.25, Fraction(1, 3), Decimal('1.25'), 1 + 2j])
def test_quantity_n_identity(value: int | float | Fraction | Decimal | complex) -> None:
    assert Quantity(value, si.one).n is value


def test_quantity_n_scaled() -> None:
    assert (1 * si.km / si.m).n == 1000
    assert (25 * si.percent).n == Fraction(1, 4)
    assert (180 * si.deg).n == pytest.approx(3.141592653589793)
    assert Quantity(1 + 2j, si.percent).n == 0.01 + 0.02j
    assert Quantity(2, Unit('', scale=pi)).n == pytest.approx(6.283185307179586)
    with localcontext() as context:
        context.prec = 2
        assert Quantity(Decimal('12.345'), si.percent).n == Decimal('0.12')


@pytest.mark.parametrize('unit', [si.m, si.dB, si.dB**2, si.dB / si.m])
def test_quantity_n_invalid_dimension(unit: Unit) -> None:
    with pytest.raises(DimensionError):
        _ = Quantity(1, unit).n


def test_quantity_n_invalid_assignment() -> None:
    with pytest.raises(AttributeError, match='read-only'):
        Quantity(1, si.one).n = 2


def test_quantity_n_point_boundary() -> None:
    assert not hasattr(AffineQuantity, 'n')
    assert not hasattr(LogQuantity, 'n')


def test_quantity_n_numpy() -> None:
    np = pytest.importorskip('numpy')
    values = np.array([25.0, 50.0])
    quantity = Quantity(values, si.percent)
    np.testing.assert_array_equal(quantity.n, [0.25, 0.5])
    np.testing.assert_array_equal(values, [25, 50])
    assert Quantity(values, si.one).n is values
    scalar = Quantity(np.array(25.0), si.percent).n
    assert scalar.shape == ()
    np.testing.assert_allclose(scalar, 0.25)
    with pytest.raises(TypeError):
        np.asarray(quantity)
