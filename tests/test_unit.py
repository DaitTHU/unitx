import pytest

from unitx import Unit, Dimension
from unitx.exceptions import UnitSyntaxError, UnitSymbolError


def test_unit_creation():
    u0 = Unit()
    assert u0.symbol == '1'
    assert u0.dimension == Dimension()
    assert float(u0.factor) == 1.0
    u = Unit('m/s')
    assert u.symbol == 'm/s'
    assert u.dimension == Dimension(L=1, T=-1)
    assert float(u.factor) == 1.0
    assert repr(u) == "Unit('m * s**-1')"


def test_unit_invalid_symbol():
    with pytest.raises(TypeError):
        Unit(123)  # type: ignore
    with pytest.raises(TypeError):
        Unit(None)  # type: ignore
    with pytest.raises(TypeError):
        Unit([])  # type: ignore
    with pytest.raises(TypeError):
        Unit(Unit('m'))  # type: ignore
    with pytest.raises(UnitSymbolError):
        Unit('invalid_unit')
    with pytest.raises(UnitSyntaxError):
        Unit('m/')


def test_unit_operations():
    u1 = Unit('m')
    u2 = Unit('s')

    u_mul = u1 * u2
    u_div = u1 / u2
    u_pow = u1 ** 2
    u_root = u1 ** -0.5

    assert u_mul.symbol == 'm⋅s'
    assert u_mul.dimension == Dimension(L=1, T=1)
    assert u_div.symbol == 'm/s'
    assert u_div.dimension == Dimension(L=1, T=-1)
    assert u_pow.symbol == 'm²'
    assert repr(u_root) == "Unit('m**(-1/2)')"
    assert eval(repr(u_root)) == u_root
