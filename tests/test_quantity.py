import pytest

from unitx import Dimension, Quantity, Unit
from unitx.exceptions import DimensionError


def test_quantity_creation_and_properties() -> None:
    q1 = Quantity(12, 'm')
    assert q1.value == 12
    assert q1.unit == Unit('m')
    assert q1.dimension == Dimension(L=1)
    assert q1._base_value == 12
    assert repr(q1) == "Quantity(12, 'm')"
    assert str(q1) == '12 m'
    unit = Unit('km')
    q2 = Quantity(2, unit)
    assert q2.unit is unit
    assert q2.value == 2


def test_conversion_between_scaled_units() -> None:
    assert Quantity(100, 'cm').to('m') == Quantity(1, 'm')
    assert Quantity(2, 'km').to(Unit('m')) == Quantity(2000, 'm')


def test_conversion_rejects_different_dimensions() -> None:
    with pytest.raises(DimensionError):
        Quantity(1, 'm').to('s')


def test_equality_is_unit_independent() -> None:
    assert Quantity(1, 'm') == Quantity(100, 'cm')
    assert Quantity(1, 'm') != Quantity(2, 'm')
    assert Quantity(1, 'm') != Quantity(1, 's')


def test_dimensionless_quantity_compares_with_number() -> None:
    q = Quantity(2, '')
    assert q == 2
    assert q < 3
    assert q <= 2
    assert q > 1
    assert q >= 2


def test_comparison_rejects_different_dimensions() -> None:
    with pytest.raises(DimensionError):
        assert Quantity(1, 'm') < Quantity(1, 's')


def test_addition_and_subtraction_convert_rhs_to_lhs_unit() -> None:
    assert Quantity(1, 'm') + Quantity(20, 'cm') == Quantity(1.2, 'm')
    assert Quantity(1, 'cm') + Quantity(1, 'm') == Quantity(101, 'cm')
    assert Quantity(1, 'm') - Quantity(20, 'cm') == Quantity(0.8, 'm')


def test_addition_and_subtraction_reject_different_dimensions() -> None:
    with pytest.raises(DimensionError):
        Quantity(1, 'm') + Quantity(1, 's')  # type: ignore
    with pytest.raises(DimensionError):
        Quantity(1, 'm') - Quantity(1, 's')  # type: ignore


def test_dimensionless_quantity_supports_number_arithmetic() -> None:
    q = Quantity(2, '')
    assert q + 3 == 3 + q == Quantity(5, '')
    assert q - 3 == Quantity(-1, '')
    assert 3 - q == Quantity(1, '')


def test_unary_operations() -> None:
    q = Quantity(-3, 'm')
    assert +q == q
    assert -q == Quantity(3, 'm')


def test_multiplication_and_division_of_quantities() -> None:
    distance = Quantity(10, 'm')
    duration = Quantity(2, 's')
    assert distance * duration == Quantity(20, 'm*s')
    assert distance / duration == Quantity(5, 'm/s')
    assert 2 * distance == Quantity(20, 'm')
    assert distance * 2 == Quantity(20, 'm')
    assert distance / 2 == Quantity(5, 'm')
    assert 20 / distance == Quantity(2, '1/m')


def test_multiplication_and_division_with_unit() -> None:
    q = Quantity(3, 'm')
    assert q * Unit('s') == Quantity(3, 'm*s')
    assert q / Unit('s') == Quantity(3, 'm/s')


def test_power() -> None:
    area = Quantity(3, 'm') ** 2
    root = Quantity(9, Unit('m') ** 2) ** 0.5
    assert area.value == 9
    assert area.unit == Unit('m') ** 2
    assert root == Quantity(3, 'm')


def test_quantity_hash_is_consistent_with_equality() -> None:
    assert hash(Quantity(1, 'm')) == hash(Quantity(100, 'cm'))
    assert {Quantity(1, 'm'), Quantity(100, 'cm')} == {Quantity(1, 'm')}

