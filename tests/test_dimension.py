import pytest
from fractions import Fraction
from unitx import Dimension, DIMENSIONLESS
from unitx.exceptions import InexactFloatWarning


def test_dimension_creation() -> None:
    d1 = Dimension(T=1, L=2)
    d2 = Dimension(T=1, L=2)
    d3 = Dimension()

    assert d1 == d2
    assert d1 != d3
    assert d3 is DIMENSIONLESS
    assert d1.as_tuple() == (1, 2, 0, 0, 0, 0, 0)
    assert repr(d2) == "Dimension(T=1, L=2)"
    assert str(d3) == "1"
    assert str(d1) == "TL²"


def test_temperature_alias() -> None:
    d = Dimension(Theta=1)

    assert d.Theta == 1
    assert d.Θ == 1
    assert d.temperature == 1
    assert str(d) == "Θ"


def test_dimension_operations() -> None:
    d1 = Dimension(T=1, L=2)
    d2 = Dimension(M=1)

    d_mul = d1 * d2
    d_div = d1 / d2
    d_pow = d1 ** 2
    d_inv = 1 / d1

    assert d_mul == Dimension(T=1, L=2, M=1)
    assert d_div == Dimension(T=1, L=2, M=-1)
    assert d_pow == Dimension(T=2, L=4)
    assert d_inv == Dimension(T=-1, L=-2)


def test_fractional_exponents() -> None:
    d = Dimension(L=1) ** Fraction(1, 2)
    assert d.L == Fraction(1, 2)
    assert str(d) == "L¹ᐟ²"


def test_invalid_operations() -> None:
    with pytest.raises(TypeError):
        Dimension(L=1) * 2  # type: ignore

    with pytest.raises(TypeError):
        Dimension(L=1) / 2  # type: ignore


def test_hash() -> None:
    assert hash(Dimension(L=1)) == hash(Dimension(L=1))
    assert {Dimension(L=1), Dimension(L=1)} == {Dimension(L=1)}


def test_float_power_warns() -> None:
    with pytest.warns(InexactFloatWarning):
        Dimension(L=1) ** (1/3)  # type: ignore


def test_operation_returns_dimensionless_singleton() -> None:
    d = Dimension(L=1)
    assert d / d is DIMENSIONLESS
    assert d ** 0 is DIMENSIONLESS


def test_dimension_is_immutable() -> None:
    d = Dimension(L=1)

    with pytest.raises(AttributeError):
        d._exponents = (1, 2, 3, 4, 5, 6, 7)  # type: ignore
    with pytest.raises(AttributeError):
        d.L = 2  # type: ignore
