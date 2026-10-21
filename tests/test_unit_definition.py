import pickle
import warnings
from collections.abc import Iterator
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from unitx.dimension import Dimension
from unitx.exceptions import (
    InexactFloatWarning,
    UnitRegistryError,
    UnitSymbolError,
)
from unitx.factor import PI, Factor
from unitx.monomial import Monomial
from unitx.quantity import Quantity
from unitx.singleunit import SingleUnit
from unitx.unit import ScaledUnit, Unit
from unitx.unitparser import UnitParser
from unitx.unitregistry import (
    PREFIX_DEFINITIONS,
    PREFIX_NAME_TO_SYMBOL,
    PREFIX_SYMBOL_ALIAS_TO_SYMBOL,
    UNIT_DEFINITIONS,
    UNIT_NAME_TO_SYMBOL,
    UNIT_SYMBOL_ALIAS_TO_SYMBOL,
    UnitDefinition,
)


@pytest.fixture(autouse=True)
def unit_registry() -> Iterator[tuple[dict[str, UnitDefinition], dict[str, str], dict[str, str]]]:
    """Keep definitions local to each test without replacing shared dictionaries."""
    tables = UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL
    snapshots = tuple(table.copy() for table in tables)
    yield snapshots  # pyright: ignore[reportReturnType]
    for table, snapshot in zip(tables, snapshots):
        table.clear()
        table.update(snapshot)  # pyright: ignore[reportCallIssue, reportArgumentType]


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


@pytest.mark.parametrize('source', [Unit('m'), Factor(2) * Unit('m'), Quantity(2, 'm')])
@pytest.mark.parametrize('keyword', [False, True])
def test_unit_define_name(source, keyword: bool) -> None:
    if keyword:
        unit = source.define('rvlen', name='reviewlength')
    else:
        unit = source.define('rvlen', 'reviewlength')
    assert type(unit) is Unit
    assert unit == Unit('rvlen') == Unit('reviewlength')
    assert SingleUnit('rvlen').name == 'reviewlength'


@pytest.mark.parametrize(
    ('source', 'scale'),
    [(Unit('m'), 1), (Factor(2) * Unit('m'), 2), (Quantity(2, 'm'), 2)],
)
def test_unit_define_aliases(source, scale: int) -> None:
    unit = source.define(
        'rvlen',
        'reviewlength',
        'reviewdistance',
        'reviewspan',
        symbol_aliases=('rvlength', 'rvspan'),
    )
    for text in ('rvlen', 'rvlength', 'rvspan', 'reviewlength', 'reviewdistance', 'reviewspan'):
        assert Unit(text) == unit
        assert SingleUnit(text).symbol == 'rvlen'
    for text in ('krvlength', 'kiloreviewdistance'):
        assert Unit(text) == Unit('krvlen')
        assert Unit(text).factor == Factor(1000 * scale)
    assert Quantity(1, unit).to('m').value == scale


def test_unit_define_empty_aliases() -> None:
    unit = Unit('m').define('rvlen', 'reviewlength', '', symbol_aliases='')
    assert Unit('rvlen') == Unit('reviewlength') == unit
    assert Unit().factor == Factor(1)
    assert SingleUnit('').symbol == ''
    assert SingleUnit('').name == ''
    assert SingleUnit('rvlen').name == 'reviewlength'
    assert SingleUnit('krvlen').name == 'kiloreviewlength'
    named = Unit('s').define('rvtime', 'reviewtime', 'reviewduration', symbol_aliases='rvduration')
    assert Unit('rvduration') == Unit('reviewduration') == named


@pytest.mark.parametrize(
    'name_aliases', [(), ('reviewdistance',), ('', 'reviewdistance', 'reviewspan')]
)
def test_unit_define_primary_name(name_aliases: tuple[str, ...]) -> None:
    Unit('m').define('rvlen', 'reviewlength', *name_aliases)
    for alias in name_aliases:
        if alias:
            assert Unit(alias) == Unit('rvlen')
    for symbol, name in (('rvlen', 'reviewlength'), ('krvlen', 'kiloreviewlength')):
        unit = SingleUnit(symbol)
        assert unit.name == name
        assert SingleUnit(unit.name) == unit


def test_unit_define_invalid_kilo() -> None:
    Unit('m').define('rvlen', name='reviewlength')
    with pytest.raises(UnitSymbolError):
        SingleUnit('Krvlen')
    assert SingleUnit('krvlen').name == 'kiloreviewlength'


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


@pytest.mark.parametrize(
    ('source', 'scale'),
    [(Unit('m'), 1), (Factor(2) * Unit('m'), 2), (Quantity(2, 'm'), 2)],
)
def test_unit_define_multiword(source, scale: int) -> None:
    unit = source.define(
        'rvau', 'review astronomical unit', 'review orbital distance', symbol_aliases='rvastro'
    )
    for text in ('review astronomical unit', 'review orbital distance', 'rvau', 'rvastro'):
        assert Unit(text) == unit
        single = SingleUnit(text)
        assert single.name == 'review astronomical unit'
        assert single.symbol == 'rvau'
        assert eval(repr(single)) == single
        assert pickle.loads(pickle.dumps(single)) == single
    assert Quantity(1, unit).to('m').value == scale
    assert Unit('review astronomical unit') / Unit('s') == Unit('rvau / s')
    assert eval(repr(unit)) == pickle.loads(pickle.dumps(unit)) == unit


@pytest.mark.parametrize(
    ('symbol', 'name'), [('', ''), ('k', 'kilo'), ('da', 'deca'), ('µ', 'micro')]
)
def test_unit_define_multiword_prefixes(symbol: str, name: str) -> None:
    unit = Unit('m').define('rvau', 'review astronomical unit', 'review orbital distance')
    for text in (name + 'review astronomical unit', name + 'review orbital distance'):
        assert Unit(text) == Unit(symbol + 'rvau')
        single = SingleUnit(text)
        assert single.name == name + 'review astronomical unit'
        assert single.without_prefix() == SingleUnit('rvau')
        assert single.factor == Unit(symbol + 'rvau').factor
    assert (
        Quantity(1, name + 'review astronomical unit').to(unit).value
        == Unit(symbol + 'rvau').factor.value
    )


def test_unit_define_multiword_prefixable() -> None:
    unit = Unit('m').define(
        'rvau', 'review astronomical unit', 'review orbital distance', prefixable=False
    )
    assert Unit('review astronomical unit') == unit
    for text in ('krvau', 'kiloreview astronomical unit', 'kiloreview orbital distance'):
        for constructor in (Unit, SingleUnit):
            with pytest.raises(UnitSymbolError, match='is not a valid unit symbol'):
                constructor(text)


@pytest.mark.parametrize('space', [' ', '  ', '\t', '\n', '\xa0', '\u2003'])
@pytest.mark.parametrize('name', ['review astronomical unit', 'metre per reviewsecond'])
def test_unit_parser_multiword_whitespace(space: str, name: str) -> None:
    Unit('m').define('rvau', name)
    parser = UnitParser(space + space.join(name.split(' ')) + space)
    result = parser.parse()
    assert result == Monomial({SingleUnit('rvau'): 1})
    assert parser.parse() is result
    parser.next_token()
    assert parser.token.kind == 'EOF'


def test_unit_parser_multiword_retry() -> None:
    parser = UnitParser('review astronomical unit')
    with pytest.raises(UnitSymbolError, match='is not a valid unit symbol'):
        parser.parse()
    unit = Unit('m').define('rvau', 'review astronomical unit')
    result = parser.parse()
    assert result == Monomial({SingleUnit('rvau'): 1})
    assert Unit('review astronomical unit') == unit
    assert parser.parse() is result


def test_unit_define_multiword_precedence() -> None:
    unit = Unit('m').define('rvau', 'review astronomical unit')
    for word in ('review', 'astronomical', 'unit'):
        Unit('s').define(word, word + 'name', prefixable=False)
    assert Unit('review astronomical unit') == unit
    assert Unit('review * astronomical * unit').dimension == Dimension(T=3)
    assert Unit('kiloreview astronomical unit') == Unit('krvau')


@pytest.mark.parametrize(
    'text',
    [
        'review astronomical unit / arcsec',
        'm / review astronomical unit',
        'review astronomical unit s',
        'review astronomical unit * s',
        '(review astronomical unit)',
        'review astronomical unit2',
        'review astronomical unit^2',
        '/review astronomical unit',
    ],
)
def test_unit_parser_invalid_multiword_factor(text: str) -> None:
    Unit('m').define('rvau', 'review astronomical unit')
    with pytest.raises(UnitSymbolError, match="'review' is not a valid unit symbol"):
        Unit(text)


@pytest.mark.parametrize('name', ['m s', 'metre second', 'kilometre second', 'km s'])
@pytest.mark.parametrize('use_alias', [False, True])
def test_unit_define_invalid_product(name: str, use_alias: bool, unit_registry) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(UnitRegistryError, match='already parses as a unit product'):
            if use_alias:
                Unit('m').define('rvau', 'review astronomical unit', name)
            else:
                Unit('m').define('rvau', name)
    assert (UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL) == unit_registry


@pytest.mark.parametrize('use_alias', [False, True])
def test_unit_define_invalid_prefixed_product(use_alias: bool) -> None:
    Unit('s').define('rvtime', 'kiloreview', prefixable=False)
    before = UNIT_DEFINITIONS.copy(), UNIT_NAME_TO_SYMBOL.copy(), UNIT_SYMBOL_ALIAS_TO_SYMBOL.copy()
    with pytest.raises(UnitRegistryError, match="'kiloreview second' already parses"):
        if use_alias:
            Unit('m').define('rvau', 'review astronomical unit', 'review second')
        else:
            Unit('m').define('rvau', 'review second')
    assert (UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL) == before


def test_unit_define_invalid_multiword_conflict() -> None:
    unit = Unit('s').define('rvtime', 'kiloreview astronomical unit', prefixable=False)
    before = UNIT_DEFINITIONS.copy(), UNIT_NAME_TO_SYMBOL.copy(), UNIT_SYMBOL_ALIAS_TO_SYMBOL.copy()
    with pytest.raises(UnitRegistryError, match="'kiloreview astronomical unit' already resolves"):
        Unit('m').define('rvau', 'review astronomical unit')
    assert Unit('kiloreview astronomical unit') == unit
    assert (UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL) == before


def test_unit_define_invalid_multiword_prefix(unit_registry) -> None:
    with pytest.raises(UnitRegistryError, match='conflicting prefix meanings'):
        Unit('m').define('rvau', 'review astronomical unit', 'kiloreview astronomical unit')
    assert (UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL) == unit_registry


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
                'rvlen', 'reviewlength', 'reviewspan', symbol_aliases='rvspan'
            )
    assert (UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL) == unit_registry


def test_unit_define_registry_stability() -> None:
    def resolutions() -> dict[str, tuple[SingleUnit, str, Factor, Dimension]]:
        texts = set(UNIT_DEFINITIONS) | set(UNIT_SYMBOL_ALIAS_TO_SYMBOL) | set(UNIT_NAME_TO_SYMBOL)
        for text in (*UNIT_DEFINITIONS, *UNIT_SYMBOL_ALIAS_TO_SYMBOL):
            symbol = UNIT_SYMBOL_ALIAS_TO_SYMBOL.get(text, text)
            if UNIT_DEFINITIONS[symbol].allows_prefix(text):
                texts.update(
                    prefix + text
                    for prefix in (*PREFIX_DEFINITIONS, *PREFIX_SYMBOL_ALIAS_TO_SYMBOL)
                )
        for name, symbol in UNIT_NAME_TO_SYMBOL.items():
            if UNIT_DEFINITIONS[symbol].allows_prefix(name):
                texts.update(prefix + name for prefix in PREFIX_NAME_TO_SYMBOL)
        units = {text: SingleUnit(text) for text in texts}
        return {
            text: (unit, unit.name, unit.factor, unit.dimension) for text, unit in units.items()
        }

    before = resolutions()
    for source, symbol, name, prefixable in (
        (Unit('m'), 'rvlen', 'reviewlength', True),
        (Factor(60) * Unit('s'), 'rvmin', 'reviewminute', False),
        (Quantity(Decimal('0.3048'), 'm'), 'rvft', 'reviewfoot', True),
        (Unit('m'), 'rvau', 'review astronomical unit', True),
    ):
        source.define(
            symbol,
            name,
            name + 'alias',
            symbol_aliases=symbol + 'alias',
            prefixable=prefixable,
        )
        after = resolutions()
        assert {text: after[text] for text in before} == before
        before = after

    registered = (
        UNIT_DEFINITIONS.copy(),
        UNIT_NAME_TO_SYMBOL.copy(),
        UNIT_SYMBOL_ALIAS_TO_SYMBOL.copy(),
    )
    for text in ('metre', 'krvlenalias', 'kiloreviewlengthalias', 'rvmin'):
        with pytest.raises(UnitRegistryError, match='already resolves to a registered unit'):
            Unit('kg').define(text, name='reviewmass')
        assert resolutions() == before
        assert (UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL) == registered


@pytest.mark.parametrize('symbol', ['', 'm/s', 'review2', '1', 'm s', '(m)', 'rv\x00'])
def test_unit_define_invalid_symbol(symbol: str) -> None:
    with pytest.raises(ValueError, match='Unit symbol must be a single printable unit token'):
        Unit('m').define(symbol, name='reviewlength')
    assert 'reviewlength' not in UNIT_NAME_TO_SYMBOL
    if symbol:
        assert symbol not in UNIT_DEFINITIONS


@pytest.mark.parametrize(
    'name',
    [
        'review2',
        'm/s',
        '\n',
        ' review',
        'review ',
        'review  distance',
        'review\tdistance',
        'rv\xa0unit',
    ],
)
def test_unit_define_invalid_name(name: str) -> None:
    with pytest.raises(ValueError, match='Unit name must consist of printable unit tokens'):
        Unit('m').define('rvlen', name=name)
    assert 'rvlen' not in UNIT_DEFINITIONS


@pytest.mark.parametrize('source', [Unit('m'), Factor(2) * Unit('m'), Quantity(2, 'm')])
def test_unit_define_invalid_empty_name(source, unit_registry) -> None:
    with pytest.raises(ValueError, match='Unit name must consist of printable unit tokens'):
        source.define('rvlen', '', 'reviewlength', '', symbol_aliases='rvspan')
    assert (UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL) == unit_registry


@pytest.mark.parametrize('source', [Unit('m'), Factor(2) * Unit('m'), Quantity(2, 'm')])
def test_unit_define_invalid_missing_name(source: Unit | Quantity[int], unit_registry) -> None:
    with pytest.raises(TypeError, match="required positional argument: 'name'"):
        source.define('rvlen')  # pyright: ignore[reportCallIssue]
    assert (UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL) == unit_registry


@pytest.mark.parametrize('alias', ['m/s', 'review2', 'rv alias'])
def test_unit_define_invalid_alias(alias: str) -> None:
    with pytest.raises(ValueError, match='Unit symbol alias must be a single printable unit token'):
        Unit('m').define('rvlen', name='reviewlength', symbol_aliases=('rvok', alias))
    assert 'rvlen' not in UNIT_DEFINITIONS
    assert 'rvok' not in UNIT_SYMBOL_ALIAS_TO_SYMBOL
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
    assert 'rvlen' not in UNIT_DEFINITIONS


@pytest.mark.parametrize('source', [Unit('m'), Factor(2) * Unit('m'), Quantity(2, 'm')])
@pytest.mark.parametrize('name_aliases', [(None,), (0, 'reviewlength'), (('reviewdistance',),)])
def test_unit_define_invalid_name_alias_type(source, name_aliases, unit_registry) -> None:
    with pytest.raises(TypeError, match='Unit name alias must be str'):
        source.define('rvlen', 'reviewlength', *name_aliases)
    assert (UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL) == unit_registry


@pytest.mark.parametrize('source', [Unit('m'), Factor(2) * Unit('m'), Quantity(2, 'm')])
@pytest.mark.parametrize('alias', ['m/s', 'review2', 'rv  alias'])
def test_unit_define_invalid_name_alias(source, alias: str, unit_registry) -> None:
    with pytest.raises(ValueError, match='Unit name alias must consist of printable unit tokens'):
        source.define('rvlen', 'reviewlength', 'reviewspan', alias)
    assert (UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL) == unit_registry


def test_unit_define_invalid_alias_type() -> None:
    with pytest.raises(TypeError, match='Unit symbol alias must be str'):
        Unit('m').define('rvlen', name='reviewlength', symbol_aliases=(1,))  # pyright: ignore[reportArgumentType]


def test_unit_define_invalid_symbol_aliases() -> None:
    with pytest.raises(TypeError, match='Symbol aliases must be str or tuple of str'):
        Unit('m').define('rvlen', name='reviewlength', symbol_aliases=None)  # pyright: ignore[reportArgumentType]


def test_unit_define_invalid_prefixable() -> None:
    with pytest.raises(TypeError, match='Prefixable must be bool'):
        Unit('m').define('rvlen', name='reviewlength', prefixable=1)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize('symbol', ['m', 'metre', 'meter', 'cm', 'kilometre', 'km'])
def test_unit_define_invalid_conflict(symbol: str) -> None:
    before = Unit('metre'), Unit('cm')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(UnitRegistryError, match='already resolves to a registered unit'):
            Unit('s').define(symbol, name='reviewtime')
    assert (Unit('metre'), Unit('cm')) == before
    assert 'reviewtime' not in UNIT_NAME_TO_SYMBOL


@pytest.mark.parametrize(
    ('name', 'name_aliases', 'symbol_aliases'),
    [('km', (), ()), ('reviewlength', (), 'meter'), ('reviewlength', ('ms',), ())],
)
def test_unit_define_invalid_namespace(
    name: str, name_aliases: tuple[str, ...], symbol_aliases: str | tuple[str, ...]
) -> None:
    with pytest.raises(UnitRegistryError, match='already resolves to a registered unit'):
        Unit('m').define('rvlen', name, *name_aliases, symbol_aliases=symbol_aliases)
    assert 'rvlen' not in UNIT_DEFINITIONS


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
    assert 'rvlen' not in UNIT_DEFINITIONS


def test_unit_define_invalid_prefixed_name() -> None:
    existing = Unit('s').define('rvtime', name='kiloreviewlength', prefixable=False)
    with pytest.raises(UnitRegistryError, match="'kiloreviewlength' already resolves"):
        Unit('m').define('rvlen', name='reviewlength')
    assert Unit('kiloreviewlength') == existing
    assert 'rvlen' not in UNIT_DEFINITIONS


def test_unit_define_invalid_ambiguity() -> None:
    with pytest.raises(UnitRegistryError, match='conflicting prefix meanings'):
        Unit('m').define('rvlen', name='krvlen')
    assert 'rvlen' not in UNIT_DEFINITIONS
    unit = Unit('m').define('rvlen', name='krvlen', prefixable=False)
    assert Unit('krvlen') == unit


def test_unit_define_duplicate_aliases() -> None:
    unit = Unit('m').define('rvlen', 'rvlen', 'rvlen', '', symbol_aliases=('rvlen', 'rvlen'))
    assert Unit('rvlen') == unit
    assert Unit('kilorvlen') == Unit('krvlen')


@pytest.mark.parametrize('value', [0, -1, Fraction(-1), Decimal('-2')])
def test_quantity_define_invalid_scale(value, unit_registry) -> None:
    with pytest.raises(ValueError, match='Factor must be positive'):
        Quantity(value, 'm').define('rvlen', name='reviewlength')
    assert (UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL) == unit_registry


@pytest.mark.parametrize('value', [True, False])
def test_quantity_define_invalid_boolean(value: bool, unit_registry) -> None:
    with pytest.raises(TypeError, match='Factor must be a number'):
        Quantity(value, 'm').define('rvlen', name='reviewlength')
    assert (UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL) == unit_registry


@pytest.mark.parametrize('value', [float('nan'), Decimal('NaN')])
def test_quantity_define_invalid_nan(value: float | Decimal, unit_registry) -> None:
    with pytest.raises(ValueError, match='cannot convert NaN to integer ratio'):
        Quantity(value, 'm').define('rvlen', name='reviewlength')
    assert (UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL) == unit_registry


@pytest.mark.parametrize(
    'value', [float('inf'), float('-inf'), Decimal('Infinity'), Decimal('-Infinity')]
)
def test_quantity_define_invalid_infinity(value: float | Decimal, unit_registry) -> None:
    with pytest.raises(OverflowError, match='cannot convert Infinity to integer ratio'):
        Quantity(value, 'm').define('rvlen', name='reviewlength')
    assert (UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL) == unit_registry


def test_quantity_define_invalid_float(unit_registry) -> None:
    with (
        pytest.warns(InexactFloatWarning),
        pytest.raises(ValueError, match='Factor must be positive'),
    ):
        Quantity(1e-10, 'm').define('rvlen', name='reviewlength')
    assert (UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL) == unit_registry
