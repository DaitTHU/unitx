import pickle
import warnings
from decimal import Decimal
from fractions import Fraction

import pytest

from unitx.dimension import DIMENSIONLESS, Dimension
from unitx.exceptions import UnitSymbolError
from unitx.singleunit import SingleUnit
from unitx.unit import Unit
from unitx.unitregistry import NO_PREFIX, PREFIX_DEFINITIONS, UNIT_DEFINITIONS


@pytest.mark.parametrize(
    ('text', 'prefix', 'unit'),
    [
        ('', '', ''),
        ('m', '', 'm'),
        ('metre', '', 'm'),
        ('mol', '', 'mol'),
        ('cd', '', 'cd'),
        ('K', '', 'K'),
        ('kelvin', '', 'K'),
        ('kg', 'k', 'g'),
        ('kilogram', 'k', 'g'),
        ('mm', 'm', 'm'),
        ('damol', 'da', 'mol'),
        ('dacd', 'da', 'cd'),
        ('decamole', 'da', 'mol'),
        ('decacandela', 'da', 'cd'),
    ],
)
def test_single_unit_creation(text: str, prefix: str, unit: str) -> None:
    value = SingleUnit(text)
    assert value.prefix_symbol == prefix
    assert value.unprefixed_symbol == unit
    assert value.symbol == str(value) == prefix + unit
    assert eval(repr(value)) == value


def test_single_unit_combinations() -> None:
    for prefix_definition in (NO_PREFIX, *PREFIX_DEFINITIONS.values()):
        prefix = prefix_definition.symbol
        for symbol, unit_definition in UNIT_DEFINITIONS.items():
            if unit_definition.reference is not None:
                continue
            if prefix and not unit_definition.allows_prefix(symbol):
                continue
            unit = SingleUnit(prefix + symbol)
            assert unit.prefix_symbol == prefix
            assert unit.unprefixed_symbol == symbol
            assert unit.prefix_name == prefix_definition.name
            assert unit.unprefixed_name == unit_definition.name
            expected_name = {
                ('M', 'Ω'): 'megohm',
                ('k', 'Ω'): 'kilohm',
            }.get((prefix, symbol), prefix_definition.name + unit_definition.name)
            if prefix and not unit_definition.allows_prefix(unit_definition.name):
                # Da permits prefixed symbols but no prefixed names.
                expected_name = unit.symbol
            assert unit.name == expected_name
            assert SingleUnit(unit.name) == unit
            assert unit.dimension == unit_definition.dimension
            assert unit.factor == prefix_definition.factor * unit_definition.factor


@pytest.mark.parametrize('symbol', ['s', 'm', 'g', 'A', 'K', 'mol', 'cd'])
def test_single_unit_aliases(symbol: str) -> None:
    unit = SingleUnit('µ' + symbol)
    aliases = [SingleUnit(prefix + symbol) for prefix in ('u', 'μ', 'µ')]
    assert all(alias.symbol == unit.symbol for alias in aliases)
    assert set(aliases) == {unit}
    lookup = {unit: 'micro'}
    assert all(lookup[alias] == 'micro' for alias in aliases)


@pytest.mark.parametrize('prefix', ('', *PREFIX_DEFINITIONS))
def test_single_unit_name_alias(prefix: str) -> None:
    prefix_name = PREFIX_DEFINITIONS[prefix].name if prefix else NO_PREFIX.name
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        unit = SingleUnit(prefix_name + 'meter')
    canonical = SingleUnit(prefix + 'm')
    assert not recorded
    assert unit == canonical
    assert hash(unit) == hash(canonical)
    assert unit.symbol == prefix + 'm'
    assert unit.name == prefix_name + 'metre'
    assert unit.dimension == Dimension(L=1)
    assert unit.factor.value == canonical.factor.value
    assert type(unit.factor.value) is type(canonical.factor.value)


@pytest.mark.parametrize(
    'text', ['', 'K', 'kelvin', 'kK', 'km', 'ms', 'Ms', 'um', 'us', 'µm', 'μm']
)
def test_single_unit_no_warning(text: str) -> None:
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        SingleUnit(text)
    assert not recorded


def test_single_unit_equality() -> None:
    assert SingleUnit('metre') == SingleUnit('m')
    assert SingleUnit('m') != SingleUnit('mm')
    assert SingleUnit('m').dimension == SingleUnit('mm').dimension
    assert SingleUnit('m') != SingleUnit('s')
    for other in ('m', 1, None):
        assert SingleUnit('m') != other


def test_single_unit_dimensionless() -> None:
    unit = SingleUnit('')
    assert unit.dimension is DIMENSIONLESS
    assert unit.factor.value == 1
    assert unit.name == unit.symbol == ''
    assert not unit.has_prefix()
    assert unit.without_prefix() is unit


@pytest.mark.parametrize(
    ('symbol', 'expected'),
    [
        ('kg', 1),
        ('g', Fraction(1, 1000)),
        ('mg', Fraction(1, 1_000_000)),
        ('qg', Fraction(1, 10**33)),
        ('Qm', 10**30),
        ('dam', 10),
        ('um', Fraction(1, 1_000_000)),
        ('mmol', Fraction(1, 1000)),
    ],
)
def test_single_unit_factor(symbol: str, expected: int | Fraction) -> None:
    value = SingleUnit(symbol).factor.value
    assert value == expected
    assert type(value) is type(expected)


def test_single_unit_mass() -> None:
    unit = SingleUnit('kg')
    assert unit.prefix_factor.value == 1000
    assert unit.unprefixed_factor.value == Fraction(1, 1000)
    assert unit.factor.value == 1
    assert unit.dimension == Dimension(M=1)


@pytest.mark.parametrize('symbol', ['', 'm', 'kg', 'um', 'mmol'])
def test_single_unit_deprefix(symbol: str) -> None:
    unit = SingleUnit(symbol)
    before = (unit.symbol, unit.factor, hash(unit))
    unprefixed = unit.without_prefix()
    assert unit.has_prefix() == (unit.prefix_symbol != '')
    assert unprefixed == SingleUnit(unit.unprefixed_symbol)
    assert unprefixed.prefix_symbol == ''
    assert unprefixed.factor == unit.unprefixed_factor
    assert unprefixed.dimension == unit.dimension
    assert (unit.symbol, unit.factor, hash(unit)) == before
    if not unit.has_prefix():
        assert unprefixed is unit


def test_single_unit_conversion() -> None:
    source = SingleUnit('kg')
    target = source.without_prefix()
    scaled = (source.factor / target.factor)(3)
    assert scaled == 3000
    assert type(scaled) is int
    assert (target.factor / source.factor)(scaled) == 3

    source, target = SingleUnit('km'), SingleUnit('mm')
    scaled = (source.factor / target.factor)(Decimal('1.25'))
    assert scaled == 1_250_000
    restored = (target.factor / source.factor)(scaled)
    assert restored == Fraction(5, 4)
    assert type(restored) is Fraction


@pytest.mark.parametrize('symbol', ['', 'kg', 'um'])
@pytest.mark.parametrize(
    'attribute', ['_prefix', '_unit', 'prefix_symbol', 'unprefixed_symbol', 'factor', 'extra']
)
def test_single_unit_immutable(symbol: str, attribute: str) -> None:
    unit = SingleUnit(symbol)
    before = (unit.symbol, unit.factor, unit.dimension, hash(unit))
    lookup = {unit: 'unchanged'}
    with pytest.raises(AttributeError, match='SingleUnit objects are immutable'):
        setattr(unit, attribute, 's')
    with pytest.raises(AttributeError, match='SingleUnit objects are immutable'):
        delattr(unit, attribute)
    assert (unit.symbol, unit.factor, unit.dimension, hash(unit)) == before
    assert lookup[unit] == 'unchanged'


@pytest.mark.parametrize('symbol', ['', 'kg', 'um'])
def test_single_unit_copy(symbol: str) -> None:
    from copy import copy, deepcopy

    unit = SingleUnit(symbol)
    assert copy(unit) is unit
    assert deepcopy(unit) is unit


@pytest.mark.parametrize('symbol', ['', 'kg', 'um'])
@pytest.mark.parametrize('protocol', range(pickle.HIGHEST_PROTOCOL + 1))
def test_single_unit_pickle(symbol: str, protocol: int) -> None:
    unit = SingleUnit(symbol)
    restored = pickle.loads(pickle.dumps(unit, protocol=protocol))
    assert restored == unit
    assert restored.symbol == unit.symbol
    assert restored.factor == unit.factor
    assert restored.dimension == unit.dimension
    assert {unit: 'unchanged'}[restored] == 'unchanged'
    with pytest.raises(AttributeError, match='SingleUnit objects are immutable'):
        restored._unit = UNIT_DEFINITIONS['s']


@pytest.mark.parametrize('value', [None, True, 1, 1.0, b'm', [], {}])
def test_single_unit_invalid_type(value: object) -> None:
    with pytest.raises(TypeError, match='Unit symbol or name must be str'):
        SingleUnit(value)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    'symbol',
    [
        'unknown',
        'k',
        'mkg',
        'kkg',
        'm/s',
        'm²',
        ' m',
        'm ',
        'Metre',
        'Kilometre',
        'kmetre',
        'kilom',
        'kilokilogram',
        'Ks',
        'Km',
        'Kg',
        'KA',
        'KK',
        'Kmol',
        'Kcd',
        'Kunknown',
        'Kmeter',
        'Km/s',
        'K m',
        'K⋅m',
    ],
)
def test_single_unit_invalid_symbol(symbol: str) -> None:
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        with pytest.raises(UnitSymbolError, match='is not a valid unit symbol'):
            SingleUnit(symbol)
    assert not recorded


@pytest.mark.parametrize(
    ('short', 'long', 'symbol'),
    [('megohm', 'megaohm', 'MΩ'), ('kilohm', 'kiloohm', 'kΩ'), ('gigaohm', 'gigaohm', 'GΩ')],
)
def test_single_unit_name_exception(short: str, long: str, symbol: str) -> None:
    assert Unit(short) == Unit(long) == Unit(symbol)
    assert SingleUnit(symbol).name == short
    assert Unit('[' + short + ']/s') == Unit(symbol + '/s')
    assert Unit(short) != Unit('Ω')
    with warnings.catch_warnings(record=True) as recorded:
        Unit(short)
    assert not recorded


def test_single_unit_distinct_identity() -> None:
    hertz, becquerel = SingleUnit('Hz'), SingleUnit('Bq')
    assert hertz.dimension == becquerel.dimension
    assert hertz.factor == becquerel.factor
    assert hertz != becquerel
    lookup = {hertz: 'frequency', becquerel: 'activity'}
    assert lookup[SingleUnit('hertz')] == 'frequency'
    assert lookup[SingleUnit('becquerel')] == 'activity'
    assert len({hertz, becquerel, SingleUnit('hertz')}) == 2
