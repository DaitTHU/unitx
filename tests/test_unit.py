import pytest

from unitx import Unit, Dimension


def test_unit_creation():
    u0 = Unit()
    assert u0.symbol == "1"
    assert u0.dimension == Dimension()
    assert u0.factor == 1.0
    u = Unit("m/s")
    assert u.symbol == "m/s"
    assert u.dimension == Dimension(L=1, T=-1)
    assert u.factor == 1.0


def test_unit_invalid_symbol():
    with pytest.raises(TypeError):
        Unit(123)  # type: ignore


def test_unit_operations():
    u1 = Unit("m")
    u2 = Unit("s")
    u3 = Unit("kg")

    u_mul = u1 * u2
    u_div = u1 / u2
    u_pow = u1 ** 2

    assert u_mul.symbol == "m⋅s"
    assert u_mul.dimension == Dimension(L=1, T=1)
    assert u_div.symbol == "m/s"
    assert u_div.dimension == Dimension(L=1, T=-1)
    assert u_pow.symbol == "m²"
