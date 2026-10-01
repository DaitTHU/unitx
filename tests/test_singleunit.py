import pytest

from unitx.singleunit import SingleUnit, UnitSymbolError


def test_singleunit_property() -> None:
    u = SingleUnit('km')

    assert u.prefix == 'k'
    assert u.unit == 'm'
    assert u.symbol == 'km'
    assert u.prefix_name == 'kilo'
    assert u.unit_name == 'meter'
    assert u.name == 'kilometer'
    assert u.prefix_factor == 1e3
    assert u.unit_factor == 1.0
    assert u.factor == 1e3
    assert u.dimension.length == 1


def test_singleunit_creation() -> None:
    u1 = SingleUnit('m')
    u2 = SingleUnit('mm')
    u3 = SingleUnit('dam')
    u4 = SingleUnit('um')
    u5 = SingleUnit(chr(0xB5) + 'm')
    u6 = SingleUnit(chr(0x03BC) + 'm')
    v1 = SingleUnit('K')
    v2 = SingleUnit('kK')
    v3 = SingleUnit('KK')

    assert u1.prefix == ''
    assert u2.prefix_name == 'milli'
    assert u3.prefix_factor == 1e1
    assert u4 == u5 == u6
    assert v1.unit == 'K'
    assert v2.prefix == v3.prefix == 'k'


def test_singleunit_invalid_creation() -> None:
    with pytest.raises(UnitSymbolError):
        SingleUnit(' ')

    with pytest.raises(UnitSymbolError):
        SingleUnit('km/s')  # compound unit

    with pytest.raises(UnitSymbolError):
        SingleUnit('k')  # single prefix without unit

    with pytest.raises(TypeError):
        SingleUnit(123)  # type: ignore


def test_singleunit_deprefix() -> None:
    u1 = SingleUnit('km')
    u2 = u1.deprefix()
    u3 = SingleUnit('m')

    assert u2 == u3
    assert u3.deprefix() is u3


def test_singleunit_hash() -> None:
    u1 = SingleUnit('km')
    u2 = SingleUnit('Km')
    u3 = SingleUnit('m')

    assert hash(u1) == hash(u2)
    assert {u1, u2} == {u1}
    assert u1 != u3
    with pytest.raises(AttributeError):
        u1._prefix = 'M'
