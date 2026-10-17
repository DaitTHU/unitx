import operator
import pickle
import warnings
from copy import copy, deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import permutations

import pytest

from unitx.dimension import DIMENSIONLESS, Dimension
from unitx.exceptions import InexactFloatWarning, UnitSymbolError, UnitSyntaxError
from unitx.factor import ONE, PI, Factor, SymbolicConstant
from unitx.quantity import Quantity
from unitx.unit import ScaledUnit, Unit


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


@pytest.mark.parametrize('value', [0, -2, 2.0, Fraction(3, 2), Decimal('1.2345'), 2j])
@pytest.mark.parametrize(
    ('unit', 'reciprocal'),
    [
        (Unit('cm'), Unit('cm^-1')),
        (ScaledUnit('cm', scale=Factor(2)), ScaledUnit('cm^-1', scale=Factor(Fraction(1, 2)))),
    ],
)
def test_unit_quantity_construction(value, unit: Unit, reciprocal: Unit) -> None:
    with localcontext() as context:
        context.prec = 2
        product = value * unit
        quotient = value / unit
    assert isinstance(product, Quantity)
    assert isinstance(quotient, Quantity)
    assert product.value is value
    assert quotient.value is value
    assert product.unit is unit
    assert quotient.unit == reciprocal


@pytest.mark.parametrize('operation', [operator.mul, operator.truediv])
@pytest.mark.parametrize('unit', [Unit('cm'), ScaledUnit('cm', scale=PI), Unit('km/m')])
@pytest.mark.parametrize(
    'value',
    [Factor(2), PI, Factor(2) ** Fraction(1, 2)],
)
def test_unit_numeric_scaling(unit: Unit, value, operation) -> None:
    before = (repr(unit), unit.factor, hash(unit))
    factor = value if isinstance(value, Factor) else Factor(value)
    original_scale = unit.scale if isinstance(unit, ScaledUnit) else ONE
    with localcontext() as context:
        context.prec = 2
        result = operation(unit, value)
    assert type(result) is ScaledUnit
    assert result.scale == operation(original_scale, factor)
    assert result.factor == operation(unit.factor, factor)
    assert result.dimension == unit.dimension
    assert result.expression() == unit.expression()
    assert (repr(unit), unit.factor, hash(unit)) == before


def test_unit_scaling_cancellation() -> None:
    unit = Unit('cm')
    scale = 2**54
    scaled = unit * Factor(scale)
    assert scaled.scale.value == scale
    assert (scaled / Unit('s')).factor == Factor(Fraction(scale, 100))
    assert ((scaled ** Fraction(1, 2)) ** 2) == scaled
    assert (PI * unit / PI) == unit
    for result in (
        scaled / Factor(scale),
        unit / Factor(scale) * Factor(scale),
        unit * ONE,
        unit / ONE,
    ):
        assert type(result) is ScaledUnit
        assert result.scale == ONE
        assert result.factor == unit.factor
        assert result == unit
        assert hash(result) == hash(unit)
        assert {unit: 'found'}[result] == 'found'
        assert len({unit, result}) == 1


def test_unit_quantity_grouping() -> None:
    metre = Unit('m')
    left = (3 * metre) * 2
    right = 3 * (metre * 2)
    assert type(left) is type(right) is Quantity
    assert left.value == 6
    assert left.unit is metre
    assert right.value == 6
    assert right.unit is metre
    assert left == right
    assert hash(left) == hash(right)
    half = 3 * (metre / 2)
    assert half.value == Fraction(3, 2)
    assert half.to(metre).value == Fraction(3, 2)
    assert type(half.to(metre).value) is Fraction


@pytest.mark.parametrize('unit', [Unit('cm'), Factor(2) * Unit('cm')])
def test_unit_quantity_dispatch(unit: Unit) -> None:
    quantity = Quantity(Fraction(3, 2), 's')
    for product in (unit * quantity, quantity * unit):
        assert type(product) is Quantity
        assert product.value is quantity.value
        assert product.unit == unit * quantity.unit
    quotient = unit / quantity
    assert type(quotient) is Quantity
    assert quotient.value == Fraction(2, 3)
    assert quotient.unit == unit / quantity.unit


@pytest.mark.parametrize('unit', [Unit('cm'), ScaledUnit('cm', PI)])
@pytest.mark.parametrize(
    'value', [0, -2, 2**60 + 1, Fraction(7, 3), Decimal('1.2345'), 0.1, 1e-300]
)
def test_unit_multiplication_symmetry(unit: Unit, value) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error', InexactFloatWarning)
        left, right = value * unit, unit * value
    assert type(left) is type(right) is Quantity
    assert left.value is right.value is value
    assert left.unit is right.unit is unit
    assert left == right
    assert hash(left) == hash(right)


@pytest.mark.parametrize('unit', [Unit('deg'), ScaledUnit('deg', PI)])
@pytest.mark.parametrize('divisor', [60, 2**60 + 1, -60, Fraction(7, 3), Decimal('1.2345')])
def test_unit_division_exact(unit: Unit, divisor) -> None:
    with localcontext() as context:
        context.prec = 2
        result = unit / divisor
    assert type(result) is Quantity
    assert type(result.value) is Fraction
    assert result.value == 1 / Fraction(divisor)
    assert result.unit is unit
    assert (result * divisor) == 1 * unit


def test_unit_division_arcminute() -> None:
    degree = Unit('deg')
    arcmin = degree / 60
    assert arcmin.value == Fraction(1, 60)
    assert arcmin.unit is degree
    assert arcmin.to('arcmin').value == 1
    assert arcmin.to('arcmin').to(degree).value == Fraction(1, 60)


@pytest.mark.parametrize('value', [0.1, 1e-300, -0.5])
def test_unit_division_float(value: float) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error', InexactFloatWarning)
        result = Unit('m') / value
    assert type(result.value) is float
    assert result.value == 1 / value


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


@pytest.mark.parametrize('value', [None, 2j, object()])
def test_unit_invalid_operands(value: object) -> None:
    unit = Unit('m')
    with pytest.raises(TypeError, match='unsupported operand'):
        unit * value  # pyright: ignore[reportUnusedExpression, reportOperatorIssue]
    with pytest.raises(TypeError, match='unsupported operand'):
        unit / value  # pyright: ignore[reportUnusedExpression, reportOperatorIssue]


@pytest.mark.parametrize('unit', [Unit('m'), ScaledUnit('m', Factor(2))])
@pytest.mark.parametrize('divisor', [0, 0.0, Fraction(0), Decimal(0)])
def test_unit_division_invalid(unit: Unit, divisor) -> None:
    with pytest.raises(ZeroDivisionError):
        unit / divisor


@pytest.mark.parametrize('unit', [Unit('m'), ScaledUnit('m', Factor(2))])
@pytest.mark.parametrize('operation', [operator.mul, operator.truediv])
@pytest.mark.parametrize('value', [True, False])
def test_unit_invalid_boolean(unit: Unit, value: bool, operation) -> None:
    with pytest.raises(TypeError, match='Boolean numeric operands'):
        operation(unit, value)


@pytest.mark.parametrize('value', [None, '1/2'])
def test_unit_invalid_power(value: object) -> None:
    with pytest.raises(TypeError, match='unsupported operand'):
        Unit('m') ** value  # pyright: ignore[reportUnusedExpression, reportOperatorIssue]


def test_unit_invalid_boolean_power() -> None:
    with pytest.raises(TypeError, match='Cannot convert bool to Fraction'):
        Unit('m') ** True  # pyright: ignore[reportUnusedExpression]


def test_scaled_unit_construction() -> None:
    base = Unit('cm')
    unit = ScaledUnit(base, Factor(100))
    assert unit.scale == Factor(100)
    assert unit.factor == ONE
    assert unit.dimension == base.dimension
    assert unit.symbol == base.symbol
    assert ScaledUnit(unit, Factor(Fraction(1, 100))) == base
    assert ScaledUnit(base, ONE) == base
    assert (PI * base).scale == PI


def test_scaled_unit_equality() -> None:
    minute = Factor(60) * Unit('s')
    rebuilt = Factor(30) * (Factor(2) * Unit('s'))
    assert minute == rebuilt
    assert hash(minute) == hash(rebuilt)
    assert {minute: 'minute'}[rebuilt] == 'minute'
    assert minute != Unit('s')
    assert Unit('s') != minute
    assert minute != Factor(30) * Unit('s')
    assert len({Unit('s'), minute, rebuilt}) == 2
    assert Factor(100) * Unit('cm') != Unit('m')
    plain = Unit('s')
    unscaled = ScaledUnit(plain, ONE)
    assert plain == unscaled == plain
    assert hash(unscaled) == hash(plain)
    assert {plain: 'found'}[unscaled] == 'found'
    assert {unscaled: 'found'}[plain] == 'found'
    assert len({plain, unscaled}) == 1


@pytest.mark.parametrize('operation', [operator.mul, operator.truediv])
@pytest.mark.parametrize('left_scale', [None, ONE, Factor(2)])
@pytest.mark.parametrize('right_scale', [None, ONE, Factor(3)])
def test_scaled_unit_promotion(
    operation, left_scale: Factor | None, right_scale: Factor | None
) -> None:
    left = Unit('cm') if left_scale is None else ScaledUnit(Unit('cm'), left_scale)
    right = Unit('s') if right_scale is None else ScaledUnit(Unit('s'), right_scale)
    result = operation(left, right)
    expected_type = Unit if left_scale is None and right_scale is None else ScaledUnit
    assert type(result) is expected_type
    assert result.factor == operation(left.factor, right.factor)
    assert result.dimension == operation(left.dimension, right.dimension)
    assert result.symbol == operation(Unit('cm'), Unit('s')).symbol


def test_scaled_unit_operations() -> None:
    minute = Factor(60) * Unit('s')
    metre = Unit('m')
    for product in (minute * metre, metre * minute):
        assert product.factor == Factor(60)
        assert product.dimension == Dimension(T=1, L=1)
    assert (minute / metre).factor == Factor(60)
    assert (metre / minute).factor == Factor(Fraction(1, 60))
    assert (minute * (Factor(2) * metre)).factor == Factor(120)
    assert (minute / (Factor(2) * metre)).factor == Factor(30)
    assert (minute / Unit('s')).factor == Factor(60)
    assert (Unit('s') / minute).factor == Factor(Fraction(1, 60))
    assert (Factor(2) / minute).factor == Factor(Fraction(1, 30))
    assert (PI / minute).factor == PI / 60


def test_scaled_unit_cancellation() -> None:
    unit = Factor(60) * Unit('s')
    before = (repr(unit), unit.factor, unit.scale, hash(unit))
    assert unit * Unit() == Unit() * unit == unit
    assert unit / Unit() == unit
    for canceled in (unit / unit, unit * unit**-1, unit**0):
        assert type(canceled) is ScaledUnit
        assert canceled.scale == ONE
        assert canceled.factor == ONE
        assert canceled == Unit()
        assert Unit() == canceled
        assert hash(canceled) == hash(Unit())
    assert (unit**2).factor == Factor(3600)
    assert (unit**-1).factor == Factor(Fraction(1, 60))
    assert (unit**2) / unit == unit
    assert (repr(unit), unit.factor, unit.scale, hash(unit)) == before


def test_scaled_unit_float_power() -> None:
    unit = Factor(2) * Unit('cm')
    with pytest.warns(InexactFloatWarning, match='0.1') as recorded:
        root = unit**0.1
    assert len(recorded) == 1
    assert type(root) is ScaledUnit
    assert root**10 == unit


def test_scaled_unit_exact_power() -> None:
    unit = (PI * 2) * Unit('cm')
    with warnings.catch_warnings():
        warnings.simplefilter('error', InexactFloatWarning)
        root = unit ** Decimal('0.1')
        assert root**10 == unit
        assert (root**10).factor == unit.factor
    symbolic_root = (PI * Unit('m')) ** Fraction(1, 2)
    assert (symbolic_root * symbolic_root).factor == PI


@pytest.mark.parametrize(
    ('base', 'scale', 'expected'),
    [
        ('s', Factor(60), '60 s'),
        ('', Factor(2), '2'),
        ('1/s', Factor(2), '2/s'),
        ('1/(m*s)', Factor(2), '2/(m⋅s)'),
        ('cm', ONE, 'cm'),
    ],
)
def test_scaled_unit_str(base: str, scale: Factor, expected: str) -> None:
    assert str(ScaledUnit(Unit(base), scale)) == expected


@pytest.mark.parametrize('text', ['', 's', 'm*s', 'm2', 'm^(1/2)', '1/(m*s)', 'µm/ms'])
@pytest.mark.parametrize('scale', [ONE, Factor(60), Factor(2) ** Fraction(1, 2), PI])
def test_scaled_unit_repr_roundtrip(text: str, scale: Factor | SymbolicConstant) -> None:
    unit = ScaledUnit(Unit(text), scale)
    rebuilt = eval(repr(unit))
    assert rebuilt == unit
    assert rebuilt.factor == unit.factor
    assert rebuilt.scale == unit.scale
    assert rebuilt.symbol == unit.symbol
    assert {unit: 'found'}[rebuilt] == 'found'


@pytest.mark.parametrize('rebuild', [copy, deepcopy])
@pytest.mark.parametrize(
    'unit',
    [
        ScaledUnit('cm', scale=ONE),
        ScaledUnit('cm^(1/2)/s', scale=PI / 180),
        ScaledUnit('', scale=Factor(2) ** Fraction(1, 2)),
        Unit('cm') * Factor(Fraction(3, 2)),
    ],
)
def test_scaled_unit_copy(unit: ScaledUnit, rebuild) -> None:
    before = (repr(unit), unit.factor, hash(unit))
    rebuilt = rebuild(unit)
    assert type(rebuilt) is ScaledUnit
    assert rebuilt == unit
    assert rebuilt.scale == unit.scale
    assert rebuilt.factor == unit.factor
    assert rebuilt.symbol == unit.symbol
    assert {unit: 'found'}[rebuilt] == 'found'
    assert (repr(unit), unit.factor, hash(unit)) == before


@pytest.mark.parametrize('protocol', range(2, pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize(
    'unit', [ScaledUnit('cm^(1/2)/s', scale=PI / 180), Unit('cm') * Factor(Fraction(3, 2))]
)
def test_scaled_unit_pickle(protocol: int, unit: ScaledUnit) -> None:
    rebuilt = pickle.loads(pickle.dumps(unit, protocol=protocol))
    assert type(rebuilt) is ScaledUnit
    assert rebuilt == unit
    assert rebuilt.scale == unit.scale
    assert rebuilt.factor == unit.factor
    assert rebuilt.symbol == unit.symbol
    assert {unit: 'found'}[rebuilt] == 'found'


@pytest.mark.parametrize('base', [None, 1])
def test_scaled_unit_invalid_base(base: object) -> None:
    with pytest.raises(TypeError, match='Base unit must be Unit or str'):
        ScaledUnit(base, ONE)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize('scale', [None, 1, Decimal('1'), '2'])
def test_scaled_unit_invalid_scale(scale: object) -> None:
    with pytest.raises(TypeError, match='Scale must be SymbolicConstant or Factor'):
        ScaledUnit(Unit('m'), scale)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize('value', [None, 2j, object()])
def test_scaled_unit_invalid_operands(value: object) -> None:
    unit = Factor(2) * Unit('m')
    with pytest.raises(TypeError, match='unsupported operand'):
        unit * value  # pyright: ignore[reportUnusedExpression, reportOperatorIssue]
    with pytest.raises(TypeError, match='unsupported operand'):
        unit / value  # pyright: ignore[reportUnusedExpression, reportOperatorIssue]


@pytest.mark.parametrize('value', [None, '1/2'])
def test_scaled_unit_invalid_power(value: object) -> None:
    with pytest.raises(TypeError, match='unsupported operand'):
        (Factor(2) * Unit('m')) ** value  # pyright: ignore[reportUnusedExpression, reportOperatorIssue]


def test_scaled_unit_invalid_boolean_power() -> None:
    with pytest.raises(TypeError, match='Cannot convert bool to Fraction'):
        (Factor(2) * Unit('m')) ** True  # pyright: ignore[reportUnusedExpression]
