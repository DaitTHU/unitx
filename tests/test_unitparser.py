import pytest

from unitx.unitparser import UnitParser, UnitSyntaxError


def test_simple_units() -> None:
    u0 = UnitParser('').parse()
    u1 = UnitParser('   ').parse()
    u2 = UnitParser('m').parse()
    u3 = UnitParser('µm').parse()
    assert str(u0) == str(u1) == '1'
    assert u0 == u1
    assert str(u2) == 'm'
    assert str(u3) == 'µm'


def test_integer_exponents() -> None:
    u1 = UnitParser('m2').parse()
    u2 = UnitParser('m ^ +20').parse()
    u3 = UnitParser('m**-1').parse()
    assert str(u1) == 'm²'
    assert str(u2) == 'm²⁰'
    assert str(u3) == '1/m'


def test_fractional_exponents() -> None:
    u1 = UnitParser('m^(1/2)').parse()
    u2 = UnitParser('m ** (-3/7)').parse()
    u3 = UnitParser('m **(3/-7)').parse()
    u4 = UnitParser('m **-(3/7)').parse()
    u5 = UnitParser('m ** -(-3/-7)').parse()
    assert str(u1) == 'm¹ᐟ²'
    assert str(u2) == '1/m³ᐟ⁷'
    assert str(u3) == '1/m³ᐟ⁷'
    assert str(u4) == '1/m³ᐟ⁷'
    assert str(u5) == '1/m³ᐟ⁷'


def test_multiplication_and_division() -> None:
    u1 = UnitParser('m s').parse()
    u2 = UnitParser('m/s').parse()
    u3 = UnitParser('m(s)').parse()
    u4 = UnitParser('kg*m/s^2').parse()
    assert str(u1) == 'm⋅s'
    assert str(u2) == 'm/s'
    assert str(u3) == 'm⋅s'
    assert str(u4) == 'kg⋅m/s²'


def test_parentheses() -> None:
    u1 = UnitParser('(m s)^2').parse()
    u2 = UnitParser('(m/s)^2').parse()
    u3 = UnitParser('kg / (m s^2)').parse()
    u4 = UnitParser('(kg m)/(s^2)').parse()
    u5 = UnitParser('((m s)^2 kg)^-1').parse()
    assert str(u1) == 'm²⋅s²'
    assert str(u2) == 'm²/s²'
    assert str(u3) == 'kg/(m⋅s²)'
    assert str(u4) == 'kg⋅m/s²'
    assert str(u5) == '1/(m²⋅s²⋅kg)'


def test_space_around_structures() -> None:
    u1 = UnitParser('  m  ').parse()
    u2 = UnitParser('  kg * m / s^2  ').parse()
    u3 = UnitParser('( m s )').parse()
    assert str(u1) == 'm'
    assert str(u2) == 'kg⋅m/s²'
    assert str(u3) == 'm⋅s'


@pytest.mark.parametrize('unit_str', ['*m', 'm*', 'm/', 'm *', 'm /'])
def test_missing_operands(unit_str: str) -> None:
    with pytest.raises(UnitSyntaxError):
        UnitParser(unit_str).parse()


def test_missing_numerator() -> None:
    u1 = UnitParser('/m').parse()  # it works!
    u2 = UnitParser('/ m(s)').parse()
    assert str(u1) == '1/m'
    assert str(u2) == 's/m'


@pytest.mark.parametrize('unit_str', ['2', 'm 2', 'm + 2', 'm - 2'])
def test_number_as_factor(unit_str: str) -> None:
    with pytest.raises(UnitSyntaxError):
        UnitParser(unit_str).parse()


@pytest.mark.parametrize('unit_str', ['m *2', 'm /2', 'm **', 'm ^()'])
def test_operator_without_factor(unit_str: str) -> None:
    with pytest.raises(UnitSyntaxError):
        UnitParser(unit_str).parse()


@pytest.mark.parametrize('unit_str', ['m^1/2', 'm**-3/7'])
def test_fraction_exponent_without_parentheses(unit_str: str) -> None:
    with pytest.raises(UnitSyntaxError):
        UnitParser(unit_str).parse()


def test_zero_denominator_in_fraction_exponent() -> None:
    with pytest.raises(ZeroDivisionError):
        UnitParser('m^(1/0)').parse()


@pytest.mark.parametrize('unit_str', ['m^(hello)', 'm^(1//2)', 'm**(1/)', 'm^(/2)', 'm**(/2)', 'm**(1/2/3)', 'm^(--1)'])
def test_invalid_exponent(unit_str: str) -> None:
    with pytest.raises(UnitSyntaxError):
        UnitParser(unit_str).parse()


@pytest.mark.parametrize('unit_str', ['(m', 'm)', '(m s', 'm^(2'])
def test_unclosed_parentheses(unit_str: str) -> None:
    with pytest.raises(UnitSyntaxError):
        UnitParser(unit_str).parse()


@pytest.mark.parametrize('unit_str', ['m++2', 'm--2', 'm**+', 'm*/s', 'm^/s', 'm/**s'])
def test_consective_operators(unit_str: str) -> None:
    with pytest.raises(UnitSyntaxError):
        UnitParser(unit_str).parse()
