import pickle
import warnings
from collections.abc import Iterator
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from unitx._unit_data import (
    PREFIX_NAME_TO_SYMBOL,
    PREFIX_SYMBOL_ALIASES,
    PREFIXES,
    UNIT_NAME_TO_SYMBOL,
    UNIT_SYMBOL_ALIASES,
    UNITS,
    UnitData,
)
from unitx.dimension import Dimension
from unitx.exceptions import (
    InexactFloatWarning,
    UnitRegistryError,
    UnitSymbolError,
    UnitSymbolWarning,
)
from unitx.factor import PI, Factor
from unitx.quantity import Quantity
from unitx.singleunit import SingleUnit
from unitx.unit import ScaledUnit, Unit


@pytest.fixture(autouse=True)
def unit_registry() -> Iterator[tuple[dict[str, UnitData], dict[str, str], dict[str, str]]]:
    """Keep definitions local to each test without replacing shared dictionaries."""
    tables = (UNITS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIASES)
    snapshots = UNITS.copy(), UNIT_NAME_TO_SYMBOL.copy(), UNIT_SYMBOL_ALIASES.copy()
    yield snapshots
    for table, snapshot in zip(tables, snapshots):
        table.clear()
        table.update(snapshot)


def test_unit_define_identity() -> None:
    source = Unit('kg*m/s2')
    before = (source.symbol, source.factor, source.dimension, hash(source))
    unit = source.define('rvN', name='reviewnewton')
    assert unit == Unit('rvN') == Unit('reviewnewton')
    assert unit != source
    assert unit.factor == source.factor
    assert unit.dimension == source.dimension
    assert Quantity(1, unit) == Quantity(1, source)
    assert (source.symbol, source.factor, source.dimension, hash(source)) == before


def test_unit_define_aliases() -> None:
    unit = Unit('m').define(
        'rvlen',
        name='reviewlength',
        symbol_aliases=('rvlength', 'rvspan'),
        name_aliases=('reviewdistance', 'reviewspan'),
    )
    for text in ('rvlen', 'rvlength', 'rvspan', 'reviewlength', 'reviewdistance', 'reviewspan'):
        assert Unit(text) == unit
        assert SingleUnit(text).symbol == 'rvlen'
    for text in ('krvlength', 'kiloreviewdistance'):
        assert Unit(text) == Unit('krvlen')
        assert Unit(text).factor == Factor(1000)


def test_unit_define_empty_aliases() -> None:
    unit = Unit('m').define('rvlen', name='reviewlength', symbol_aliases='', name_aliases='')
    assert Unit('rvlen') == Unit('reviewlength') == unit
    assert Unit().factor == Factor(1)
    assert SingleUnit('').symbol == ''
    assert SingleUnit('').name == ''
    assert SingleUnit('rvlen').name == 'reviewlength'
    assert SingleUnit('krvlen').name == 'kiloreviewlength'
    named = Unit('s').define(
        'rvtime', name='reviewtime', symbol_aliases='rvduration', name_aliases='reviewduration'
    )
    assert Unit('rvduration') == Unit('reviewduration') == named


@pytest.mark.parametrize('name_aliases', ['reviewdistance', ('', 'reviewdistance', 'reviewspan')])
def test_unit_define_primary_name(name_aliases: str | tuple[str, ...]) -> None:
    Unit('m').define('rvlen', name='reviewlength', name_aliases=name_aliases)
    for symbol, name in (('rvlen', 'reviewlength'), ('krvlen', 'kiloreviewlength')):
        unit = SingleUnit(symbol)
        assert unit.name == name
        assert SingleUnit(unit.name) == unit


def test_unit_define_prefix_warning() -> None:
    Unit('m').define('rvlen', name='reviewlength')
    with pytest.warns(UnitSymbolWarning, match=r"'krvlen' \(kiloreviewlength\)") as recorded:
        unit = SingleUnit('Krvlen')
    assert unit.name == 'kiloreviewlength'
    assert unit == SingleUnit('krvlen')
    assert 'kelvin reviewlength' in str(recorded[0].message)


@pytest.mark.parametrize('symbol', ["rv'length", 'rv"length', r'rv\time'])
@pytest.mark.parametrize('prefix', ['', 'k'])
def test_unit_define_escaped_symbol(symbol: str, prefix: str) -> None:
    Unit('m').define(symbol, name='reviewlength')
    for unit in (
        SingleUnit(prefix + symbol),
        Unit(prefix + symbol),
        ScaledUnit(prefix + symbol, scale=Factor(2)),
    ):
        rebuilt = eval(repr(unit))
        assert type(rebuilt) is type(unit)
        assert rebuilt == unit
        assert rebuilt.factor == unit.factor
        assert rebuilt.symbol == unit.symbol
        restored = pickle.loads(pickle.dumps(unit))
        assert type(restored) is type(unit)
        assert restored == unit
        assert restored.factor == unit.factor
        assert {unit: 'found'}[restored] == 'found'


def test_unit_define_prefixes() -> None:
    unit = Unit('m').define('reviewlength', name='reviewdistanceunit')
    assert SingleUnit('reviewlength').name == 'reviewdistanceunit'
    for symbol, name, scale in (
        ('kreviewlength', 'kiloreviewdistanceunit', 1000),
        ('dareviewlength', 'decareviewdistanceunit', 10),
        ('ureviewlength', 'microreviewdistanceunit', Fraction(1, 10**6)),
    ):
        assert Unit(symbol) == Unit(name)
        assert Unit(symbol).factor == unit.factor * scale


def test_unit_define_prefixable() -> None:
    unit = Unit('s').define('rvmin', name='reviewminute', prefixable=False)
    assert Unit('rvmin') == unit
    for text in ('krvmin', 'kiloreviewminute'):
        with pytest.raises(UnitSymbolError, match='is not a valid unit symbol'):
            Unit(text)


def test_scaled_unit_define_exact() -> None:
    source = (PI / 180) * Unit('m')
    unit = source.define('rvangle', name='reviewangle', prefixable=False)
    assert type(unit) is Unit
    assert unit.factor == PI / 180
    assert unit.dimension == Dimension(L=1)
    assert (Quantity(1, unit) / Quantity(1, source)).to('').value == 1
    rooted = (Factor(2) * Unit('m')) ** Fraction(1, 2)
    root_unit = rooted.define('rvroot', name='reviewroot')
    assert (root_unit**2).factor == Factor(2)
    assert (root_unit**2).dimension == Dimension(L=1)


@pytest.mark.parametrize('value', [60, Fraction(60), Decimal('60'), 60.0])
def test_quantity_define_scale(value) -> None:
    source = Quantity(value, 's')
    unit = source.define('rvmin', name='reviewminute', prefixable=False)
    assert unit.factor == Factor(60)
    assert Quantity(1, unit).to('s').value == 60
    assert source.value is value
    assert source.unit == Unit('s')


def test_quantity_define_decimal() -> None:
    with localcontext() as context:
        context.prec = 2
        unit = Quantity(Decimal('0.3048'), 'm').define('rvft', name='reviewfoot')
    assert unit.factor.value == Fraction(381, 1250)
    assert type(unit.factor.value) is Fraction
    assert Quantity(1, unit).to('m').to(unit).value == 1


def test_quantity_define_scaled() -> None:
    unit = Quantity(2, Factor(60) * Unit('s')).define('rvtwo', name='reviewtwominutes')
    assert unit.factor == Factor(120)
    assert unit.dimension == Dimension(T=1)


def test_quantity_define_float() -> None:
    source = Quantity(0.1, 'm')
    with pytest.warns(InexactFloatWarning) as recorded:
        unit = source.define('rvlen', name='reviewlength')
    assert len(recorded) == 1
    assert unit.factor == Factor(Fraction(1, 10))
    assert unit.factor.value == Fraction(1, 10)
    assert Quantity(1, unit).to('m').to(unit).value == 1
    assert source.value == 0.1
    assert source.unit == Unit('m')


def test_quantity_define_warning_error(unit_registry) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error', InexactFloatWarning)
        with pytest.raises(InexactFloatWarning, match='conversion is inexact'):
            Quantity(0.1, 'm').define(
                'rvlen', name='reviewlength', symbol_aliases='rvspan', name_aliases='reviewspan'
            )
    assert (UNITS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIASES) == unit_registry


def test_unit_define_registry_stability() -> None:
    def resolutions() -> dict[str, tuple[SingleUnit, str, Factor, Dimension]]:
        texts = set(UNITS) | set(UNIT_SYMBOL_ALIASES) | set(UNIT_NAME_TO_SYMBOL)
        for text in (*UNITS, *UNIT_SYMBOL_ALIASES):
            symbol = UNIT_SYMBOL_ALIASES.get(text, text)
            if UNITS[symbol].prefixable:
                texts.update(prefix + text for prefix in (*PREFIXES, *PREFIX_SYMBOL_ALIASES))
        for name, symbol in UNIT_NAME_TO_SYMBOL.items():
            if UNITS[symbol].prefixable:
                texts.update(prefix + name for prefix in PREFIX_NAME_TO_SYMBOL)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UnitSymbolWarning)
            units = {text: SingleUnit(text) for text in texts}
        return {
            text: (unit, unit.name, unit.factor, unit.dimension) for text, unit in units.items()
        }

    before = resolutions()
    for source, symbol, name, prefixable in (
        (Unit('m'), 'rvlen', 'reviewlength', True),
        (Factor(60) * Unit('s'), 'rvmin', 'reviewminute', False),
        (Quantity(Decimal('0.3048'), 'm'), 'rvft', 'reviewfoot', True),
    ):
        source.define(
            symbol,
            name=name,
            symbol_aliases=symbol + 'alias',
            name_aliases=name + 'alias',
            prefixable=prefixable,
        )
        after = resolutions()
        assert {text: after[text] for text in before} == before
        before = after

    registered = UNITS.copy(), UNIT_NAME_TO_SYMBOL.copy(), UNIT_SYMBOL_ALIASES.copy()
    for text in ('metre', 'Krvlenalias', 'kiloreviewlengthalias', 'rvmin'):
        with pytest.raises(UnitRegistryError, match='already resolves to a registered unit'):
            Unit('kg').define(text, name='reviewmass')
        assert resolutions() == before
        assert (UNITS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIASES) == registered


@pytest.mark.parametrize('symbol', ['', 'm/s', 'review2', '1', 'm s', '(m)', 'rv\x00'])
def test_unit_define_invalid_symbol(symbol: str) -> None:
    with pytest.raises(ValueError, match='Unit symbol must be a single printable unit token'):
        Unit('m').define(symbol, name='reviewlength')
    assert 'reviewlength' not in UNIT_NAME_TO_SYMBOL
    if symbol:
        assert symbol not in UNITS


@pytest.mark.parametrize('name', ['metre per second', 'review2', 'm/s', '\n'])
def test_unit_define_invalid_name(name: str) -> None:
    with pytest.raises(ValueError, match='Unit name must be a single printable unit token'):
        Unit('m').define('rvlen', name=name)
    assert 'rvlen' not in UNITS


@pytest.mark.parametrize('source', [Unit('m'), Factor(2) * Unit('m'), Quantity(2, 'm')])
def test_unit_define_invalid_empty_name(source, unit_registry) -> None:
    with pytest.raises(ValueError, match='Unit name must be a single printable unit token'):
        source.define('rvlen', name='', symbol_aliases='rvspan', name_aliases=('reviewlength', ''))
    assert (UNITS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIASES) == unit_registry


@pytest.mark.parametrize('source', [Unit('m'), Factor(2) * Unit('m'), Quantity(2, 'm')])
def test_unit_define_invalid_missing_name(source, unit_registry) -> None:
    with pytest.raises(TypeError, match="required keyword-only argument: 'name'"):
        source.define('rvlen', name_aliases='reviewlength')  # pyright: ignore[reportCallIssue]
    assert (UNITS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIASES) == unit_registry


@pytest.mark.parametrize('alias', ['m/s', 'review2', 'rv alias'])
def test_unit_define_invalid_alias(alias: str) -> None:
    with pytest.raises(ValueError, match='Unit symbol alias must be a single printable unit token'):
        Unit('m').define('rvlen', name='reviewlength', symbol_aliases=('rvok', alias))
    assert 'rvlen' not in UNITS
    assert 'rvok' not in UNIT_SYMBOL_ALIASES
    assert 'reviewlength' not in UNIT_NAME_TO_SYMBOL


@pytest.mark.parametrize('symbol', [None, 2, True])
def test_unit_define_invalid_type(symbol: object) -> None:
    with pytest.raises(TypeError, match='Unit symbol must be str'):
        Unit('m').define(symbol, name='reviewlength')  # pyright: ignore[reportArgumentType]
    assert 'reviewlength' not in UNIT_NAME_TO_SYMBOL


@pytest.mark.parametrize('name', [None, 0, True])
def test_unit_define_invalid_name_type(name) -> None:
    with pytest.raises(TypeError, match='Unit name must be str'):
        Unit('m').define('rvlen', name=name)
    assert 'rvlen' not in UNITS


@pytest.mark.parametrize('name_aliases', [(None,), (0, 'reviewlength')])
def test_unit_define_invalid_name_alias_type(name_aliases, unit_registry) -> None:
    with pytest.raises(TypeError, match='Unit name alias must be str'):
        Unit('m').define('rvlen', name='reviewlength', name_aliases=name_aliases)
    assert (UNITS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIASES) == unit_registry


@pytest.mark.parametrize('alias', ['m/s', 'review2', 'rv alias'])
def test_unit_define_invalid_name_alias(alias: str, unit_registry) -> None:
    with pytest.raises(ValueError, match='Unit name alias must be a single printable unit token'):
        Unit('m').define('rvlen', name='reviewlength', name_aliases=('reviewspan', alias))
    assert (UNITS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIASES) == unit_registry


def test_unit_define_invalid_alias_type() -> None:
    with pytest.raises(TypeError, match='Unit symbol alias must be str'):
        Unit('m').define('rvlen', name='reviewlength', symbol_aliases=(1,))  # pyright: ignore[reportArgumentType]


def test_unit_define_invalid_symbol_aliases() -> None:
    with pytest.raises(TypeError, match='Symbol aliases must be str or tuple of str'):
        Unit('m').define('rvlen', name='reviewlength', symbol_aliases=None)  # pyright: ignore[reportArgumentType]


def test_unit_define_invalid_name_aliases() -> None:
    with pytest.raises(TypeError, match='Name aliases must be str or tuple of str'):
        Unit('m').define('rvlen', name='reviewlength', name_aliases=None)  # pyright: ignore[reportArgumentType]


def test_unit_define_invalid_prefixable() -> None:
    with pytest.raises(TypeError, match='Prefixable must be bool'):
        Unit('m').define('rvlen', name='reviewlength', prefixable=1)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize('symbol', ['m', 'metre', 'meter', 'cm', 'kilometre', 'Km'])
def test_unit_define_invalid_conflict(symbol: str) -> None:
    before = Unit('metre'), Unit('cm')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(UnitRegistryError, match='already resolves to a registered unit'):
            Unit('s').define(symbol, name='reviewtime')
    assert (Unit('metre'), Unit('cm')) == before
    assert 'reviewtime' not in UNIT_NAME_TO_SYMBOL


@pytest.mark.parametrize(
    'kwargs', [{'name': 'km'}, {'symbol_aliases': 'meter'}, {'name_aliases': 'ms'}]
)
def test_unit_define_invalid_namespace(kwargs) -> None:
    arguments = {'name': 'reviewlength', **kwargs}
    with pytest.raises(UnitRegistryError, match='already resolves to a registered unit'):
        Unit('m').define('rvlen', **arguments)
    assert 'rvlen' not in UNITS


def test_unit_define_invalid_duplicate() -> None:
    unit = Unit('m').define('rvlen', name='reviewlength', symbol_aliases='rvspan')
    with pytest.raises(UnitRegistryError, match="'rvspan' already resolves"):
        Unit('s').define('rvspan', name='reviewtime')
    assert Unit('rvlen') == Unit('rvspan') == Unit('reviewlength') == unit
    assert 'reviewtime' not in UNIT_NAME_TO_SYMBOL


@pytest.mark.parametrize('use_name', [False, True])
def test_unit_define_invalid_prefixed_conflict(use_name: bool) -> None:
    if use_name:
        existing = Unit('s').define('rvtime', name='krvlen', prefixable=False)
    else:
        existing = Unit('s').define('krvlen', name='reviewtime', prefixable=False)
    with pytest.raises(UnitRegistryError, match="'krvlen' already resolves"):
        Unit('m').define('rvlen', name='reviewlength')
    assert Unit('krvlen') == existing
    assert 'rvlen' not in UNITS


def test_unit_define_invalid_prefixed_name() -> None:
    existing = Unit('s').define('rvtime', name='kiloreviewlength', prefixable=False)
    with pytest.raises(UnitRegistryError, match="'kiloreviewlength' already resolves"):
        Unit('m').define('rvlen', name='reviewlength')
    assert Unit('kiloreviewlength') == existing
    assert 'rvlen' not in UNITS


def test_unit_define_invalid_ambiguity() -> None:
    with pytest.raises(UnitRegistryError, match='conflicting prefix meanings'):
        Unit('m').define('rvlen', name='krvlen')
    assert 'rvlen' not in UNITS
    unit = Unit('m').define('rvlen', name='krvlen', prefixable=False)
    assert Unit('krvlen') == unit


def test_unit_define_duplicate_aliases() -> None:
    unit = Unit('m').define(
        'rvlen', name='rvlen', symbol_aliases=('rvlen', 'rvlen'), name_aliases=('rvlen', '')
    )
    assert Unit('rvlen') == unit
    assert Unit('kilorvlen') == Unit('krvlen')


@pytest.mark.parametrize('value', [0, -1, Fraction(-1), Decimal('-2')])
def test_quantity_define_invalid_scale(value, unit_registry) -> None:
    with pytest.raises(ValueError, match='Factor must be positive'):
        Quantity(value, 'm').define('rvlen', name='reviewlength')
    assert (UNITS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIASES) == unit_registry


@pytest.mark.parametrize('value', [True, False])
def test_quantity_define_invalid_boolean(value: bool, unit_registry) -> None:
    with pytest.raises(TypeError, match='Factor must be a number'):
        Quantity(value, 'm').define('rvlen', name='reviewlength')
    assert (UNITS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIASES) == unit_registry


@pytest.mark.parametrize('value', [float('nan'), Decimal('NaN')])
def test_quantity_define_invalid_nan(value: float | Decimal, unit_registry) -> None:
    with pytest.raises(ValueError, match='cannot convert NaN to integer ratio'):
        Quantity(value, 'm').define('rvlen', name='reviewlength')
    assert (UNITS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIASES) == unit_registry


@pytest.mark.parametrize(
    'value', [float('inf'), float('-inf'), Decimal('Infinity'), Decimal('-Infinity')]
)
def test_quantity_define_invalid_infinity(value: float | Decimal, unit_registry) -> None:
    with pytest.raises(OverflowError, match='cannot convert Infinity to integer ratio'):
        Quantity(value, 'm').define('rvlen', name='reviewlength')
    assert (UNITS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIASES) == unit_registry


def test_quantity_define_invalid_float(unit_registry) -> None:
    with (
        pytest.warns(InexactFloatWarning),
        pytest.raises(ValueError, match='Factor must be positive'),
    ):
        Quantity(1e-10, 'm').define('rvlen', name='reviewlength')
    assert (UNITS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIASES) == unit_registry
