import warnings
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import permutations

import pytest

from unitx.dimension import DIMENSIONLESS, Dimension
from unitx.exceptions import InexactFloatWarning, UnitSymbolError, UnitSyntaxError
from unitx.unit import Unit


def test_unit_creation() -> None:
    u0 = Unit()
    assert str(u0) == u0.symbol == '1'
    assert u0.dimension is DIMENSIONLESS
    assert u0.factor.value == 1
    assert type(u0.factor.value) is int
    u = Unit('m/s')
    assert str(u) == u.symbol == 'm/s'
    assert u.dimension == Dimension(L=1, T=-1)
    assert u.factor.value == 1
    assert type(u.factor.value) is int
    assert repr(u) == "Unit('m * s**-1')"


@pytest.mark.parametrize(
    ('text', 'canonical'),
    [
        ('kilogram metre / second2', 'kg*m/s2'),
        ('meter / second', 'm/s'),
        ('um/ms', 'µm/ms'),
        ('μm/ms', 'µm/ms'),
    ],
)
def test_unit_aliases(text: str, canonical: str) -> None:
    unit = Unit(text)
    expected = Unit(canonical)
    assert unit == expected
    assert unit.symbol == expected.symbol
    assert unit.factor == expected.factor
    assert unit.dimension == expected.dimension


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('m*s', 'm⋅s'),
        ('m/(s*A)', 'm/(s⋅A)'),
        ('1/(m*s)', '1/(m⋅s)'),
        ('m^(1/2)/s2', 'm¹ᐟ²/s²'),
    ],
)
def test_unit_symbol(text: str, expected: str) -> None:
    unit = Unit(text)
    assert unit.symbol == str(unit) == expected


def test_unit_equality() -> None:
    assert Unit('m/s') == Unit('s^-1*m')
    assert Unit('m') != Unit('mm')
    assert Unit('m') != Unit('s')
    assert Unit('m/m') == Unit()

    # A coherent dimensionless expression can retain distinct formal unit factors.
    unit = Unit('kg*m/(g*km)')
    assert unit.factor.value == 1
    assert unit.is_dimensionless()
    assert unit != Unit()

    for other in ('m', 1, None):
        assert Unit('m') != other


def test_unit_hash() -> None:
    units = [Unit('*'.join(symbols)) for symbols in permutations(('kg', 'm', 's'))]
    first = units[0]
    lookup = {first: 'found'}
    for unit in units:
        assert unit == first
        assert hash(unit) == hash(first)
        assert lookup[unit] == 'found'
    assert len(set(units)) == 1
    assert len({Unit(), Unit('m/m'), Unit('s0')}) == 1


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('g', Fraction(1, 1000)),
        ('kg', 1),
        ('mg', Fraction(1, 1_000_000)),
        ('km/ms', 1_000_000),
        ('g/cm3', 1000),
        ('mg/cm3', 1),
        ('cm2/ms2', 100),
        ('cm^(1/2)', Fraction(1, 10)),
    ],
)
def test_unit_factor(text: str, expected: int | Fraction) -> None:
    value = Unit(text).factor.value
    assert value == expected
    assert type(value) is type(expected)


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('kg*m/s2', Dimension(M=1, L=1, T=-2)),
        ('mg/cm3', Dimension(M=1, L=-3)),
        ('m/(s*A)', Dimension(L=1, T=-1, I=-1)),
        ('K*mol/cd', Dimension(Theta=1, N=1, J=-1)),
    ],
)
def test_unit_dimension(text: str, expected: Dimension) -> None:
    unit = Unit(text)
    assert unit.dimension == expected
    assert not unit.is_dimensionless()


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('', 1),
        (' \t\n', 1),
        ('1', 1),
        ('m/m', 1),
        ('km/m', 1000),
        ('m/km', Fraction(1, 1000)),
        ('kg/g', 1000),
    ],
)
def test_unit_dimensionless(text: str, expected: int | Fraction) -> None:
    unit = Unit(text)
    assert unit.is_dimensionless()
    assert unit.dimension is DIMENSIONLESS
    assert unit.factor.value == expected
    assert type(unit.factor.value) is type(expected)


def test_unit_operations() -> None:
    u1 = Unit('m')
    u2 = Unit('s')

    u_mul = u1 * u2
    u_div = u1 / u2
    u_pow = u1**2
    u_root = u1**-0.5

    assert u_mul.symbol == 'm⋅s'
    assert u_mul.dimension == Dimension(L=1, T=1)
    assert u_div.symbol == 'm/s'
    assert u_div.dimension == Dimension(L=1, T=-1)
    assert u_pow.symbol == 'm²'
    assert u_pow.dimension == Dimension(L=2)
    assert repr(u_root) == "Unit('m**(-1/2)')"
    assert eval(repr(u_root)) == u_root


def test_unit_cancellation() -> None:
    unit = Unit('mg/cm3')
    other = Unit('cm2/s')
    assert unit * other == other * unit
    assert (unit * other) / other == unit
    assert unit / unit == Unit()
    assert unit * Unit() == Unit() * unit == unit
    assert unit / Unit() == unit
    assert unit**0 == Unit()
    assert unit * unit**-1 == Unit()


def test_unit_fractional_power() -> None:
    unit = Unit('mm/s')
    root = unit ** Fraction(1, 3)
    assert root.dimension == Dimension(L=Fraction(1, 3), T=Fraction(-1, 3))
    assert root.factor.value == Fraction(1, 10)
    assert type(root.factor.value) is Fraction
    assert root**3 == unit
    assert (root**3).factor.value == Fraction(1, 1000)
    assert unit ** Fraction(-1, 3) == root**-1
    assert (unit ** Fraction(-1, 3)).factor.value == 10


def test_unit_irrational_factor() -> None:
    root = Unit('mm') ** Fraction(1, 2)
    assert root.dimension == Dimension(L=Fraction(1, 2))
    assert type(root.factor.value) is float
    assert root.factor.value == pytest.approx(0.001**0.5)
    squared = root * root
    assert squared == Unit('mm')
    assert squared.factor.value == Fraction(1, 1000)
    assert type(squared.factor.value) is Fraction


@pytest.mark.parametrize(
    ('exponent', 'expression', 'expected'),
    [
        (Fraction(1, 1_000_001), 'm^(1/1000001)', Fraction(1, 1_000_001)),
        (Decimal('0.1'), 'm^0.1', Fraction(1, 10)),
        (Decimal('0.0000000001'), 'm^0.0000000001', Fraction(1, 10**10)),
    ],
)
def test_unit_exact_exponents(
    exponent: Fraction | Decimal, expression: str, expected: Fraction
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error', InexactFloatWarning)
        unit = Unit('m') ** exponent
        assert unit == Unit(expression)
    assert unit.dimension.L == expected
    assert type(unit.dimension.L) is Fraction


def test_unit_float_power() -> None:
    with pytest.warns(InexactFloatWarning, match='0.1'):
        unit = Unit('m') ** 0.1
    assert unit == Unit('m^0.1')
    assert unit.dimension == Dimension(L=Fraction(1, 10))


def test_unit_inplace_operations() -> None:
    original = Unit('cm/s')
    before = (
        original.symbol,
        repr(original),
        original.factor.value,
        original.dimension,
        hash(original),
    )
    lookup = {original: 'unchanged'}
    product = quotient = power = original
    product *= Unit('s')
    quotient /= original
    power **= Fraction(-1, 2)
    assert product == Unit('cm')
    assert product.factor.value == Fraction(1, 100)
    assert quotient == Unit()
    assert quotient.factor.value == 1
    assert power == Unit('cm^(-1/2)*s^(1/2)')
    assert power.factor.value == 10
    assert power.dimension == Dimension(L=Fraction(-1, 2), T=Fraction(1, 2))
    assert (
        original.symbol,
        repr(original),
        original.factor.value,
        original.dimension,
        hash(original),
    ) == before
    assert lookup[original] == 'unchanged'


def test_unit_conversion_roundtrip() -> None:
    source, target = Unit('g/cm3'), Unit('kg/m3')
    assert source.dimension == target.dimension
    with localcontext() as context:
        context.prec = 2
        converted = (source.factor / target.factor)(Decimal('1.2345'))
    assert converted == Fraction(2469, 2)
    assert type(converted) is Fraction
    restored = (target.factor / source.factor)(converted)
    assert restored == Fraction(2469, 2000)
    assert type(restored) is Fraction


@pytest.mark.parametrize('text', ['', 'µm/ms', 'kg*m/s2', '1/(m*s)', 'cm^(1/3)/ms^(2/3)'])
def test_unit_repr_roundtrip(text: str) -> None:
    unit = Unit(text)
    rebuilt = eval(repr(unit))
    assert rebuilt == unit
    assert rebuilt.symbol == unit.symbol
    assert rebuilt.dimension == unit.dimension
    assert rebuilt.factor == unit.factor
    assert {unit: 'found'}[rebuilt] == 'found'


@pytest.mark.parametrize('value', [123, None, [], True, b'm', Unit('m')])
def test_unit_invalid_type(value: object) -> None:
    with pytest.raises(TypeError, match='Unit symbol must be str'):
        Unit(value)  # pyright: ignore[reportArgumentType]


def test_unit_invalid_symbol() -> None:
    with pytest.raises(UnitSymbolError, match='is not a valid unit symbol'):
        Unit('invalid_unit')


def test_unit_invalid_syntax() -> None:
    with pytest.raises(UnitSyntaxError, match='Expected unit or parenthesized expression'):
        Unit('m/')


@pytest.mark.parametrize('value', [None, 2])
def test_unit_invalid_operands(value: object) -> None:
    unit = Unit('m')
    with pytest.raises(TypeError, match='unsupported operand'):
        unit * value  # pyright: ignore[reportUnusedExpression, reportOperatorIssue]
    with pytest.raises(TypeError, match='unsupported operand'):
        unit / value  # pyright: ignore[reportUnusedExpression, reportOperatorIssue]


@pytest.mark.parametrize('value', [None, '1/2'])
def test_unit_invalid_power(value: object) -> None:
    with pytest.raises(TypeError, match='unsupported operand'):
        Unit('m') ** value  # pyright: ignore[reportUnusedExpression, reportOperatorIssue]


def test_unit_invalid_boolean_power() -> None:
    with pytest.raises(TypeError, match='Cannot convert bool to Fraction'):
        Unit('m') ** True  # pyright: ignore[reportUnusedExpression]
