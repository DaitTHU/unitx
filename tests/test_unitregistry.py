"""Registration policies, spelling fallbacks, and failure invariants."""

import copy
import pickle
from fractions import Fraction

import pytest

from unitx.exceptions import UnitRegistryError, UnitSymbolError
from unitx.factor import PI, Factor
from unitx.quantity import Quantity
from unitx.singleunit import SingleUnit
from unitx.unit import ScaledUnit, Unit
from unitx.unitparser import UnitParser
from unitx.unitregistry import UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL


@pytest.fixture(autouse=True)
def unit_registry():
    tables = UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL
    snapshots = tuple(table.copy() for table in tables)
    yield snapshots
    for table, snapshot in zip(tables, snapshots):
        table.clear()
        table.update(snapshot)


@pytest.mark.parametrize('source', [Unit('m'), PI * Unit('m'), Quantity(Fraction(3, 7), 'm')])
def test_unit_define_binding_policy(source) -> None:
    unit = source.define(
        'rvbind',
        'review binding',
        'review distance',
        symbol_aliases=('rvallowed', 'rvblocked'),
        prefixable=('rvallowed', 'review distance'),
    )
    assert type(unit) is Unit
    for text in ('rvbind', 'rvallowed', 'rvblocked', 'review binding', 'review distance'):
        assert Unit(text) == unit
    for prefix, name in (('k', 'kilo'), ('µ', 'micro')):
        single = SingleUnit(prefix + 'rvallowed')
        assert single.unprefixed_symbol == 'rvbind'
        assert single.symbol == prefix + 'rvallowed'
        assert single.name == name + 'review distance'
        assert single.without_prefix() == SingleUnit('rvbind')
        assert Unit(name + 'review distance') == Unit(prefix + 'rvallowed')
        assert Unit(prefix + 'rvallowed').factor == single.prefix_factor * unit.factor
    assert Unit('mrvallowed') == Unit('millireview distance')
    for text in ('krvbind', 'krvblocked', 'kiloreview binding'):
        with pytest.raises(UnitSymbolError):
            Unit(text)


@pytest.mark.parametrize('prefix', ['µ', 'μ', 'u'])
def test_unit_define_prefix_alias(prefix: str) -> None:
    unit = Unit('m').define(
        'rvselect',
        'review selected',
        prefixable=('rvselect', 'review selected'),
    )
    assert Unit(prefix + 'rvselect') == Unit('microreview selected')
    assert Unit(prefix + 'rvselect').factor == unit.factor / 10**6
    assert Unit('krvselect') == Unit('kiloreview selected')
    assert Unit('mrvselect') == Unit('millireview selected')
    for text in ('uµrvselect', 'micromicroreview selected'):
        with pytest.raises(UnitSymbolError):
            Unit(text)


@pytest.mark.parametrize('name', ['review name-only', 'reviewnameonly'])
def test_single_unit_name_fallback(name: str) -> None:
    Unit('m').define(
        'rvnameonly',
        name,
        prefixable=(name,),
    )
    single = SingleUnit('kilo' + name)
    assert single.symbol == single.name == 'kilo' + name
    unit = Unit(single.name)
    assert unit.expression() == '[kilo' + name + ']'
    assert Unit('s') / unit**2 == Unit('s / [' + single.name + ']**2')
    for value in (single, unit, Factor(7) * unit, Quantity(Fraction(3, 2), unit)):
        assert eval(repr(value), globals() | {'ScaledUnit': ScaledUnit}) == value
        assert copy.copy(value) == copy.deepcopy(value) == value
        restored = pickle.loads(pickle.dumps(value))
        assert restored == value
        assert hash(restored) == hash(value)
        assert {value: 'found'}[restored] == 'found'
    with pytest.raises(UnitSymbolError):
        Unit('krvnameonly')


def test_single_unit_symbol_fallback() -> None:
    Unit('s').define('rvonlysymbol', 'review symbol', prefixable=('rvonlysymbol',))
    single = SingleUnit('krvonlysymbol')
    assert single.name == single.symbol == 'krvonlysymbol'
    assert SingleUnit(single.name) == single
    assert Unit('krvonlysymbol').expression() == 'krvonlysymbol'
    with pytest.raises(UnitSymbolError):
        Unit('[kiloreview symbol]')


@pytest.mark.parametrize('policy', [False, ()])
def test_unit_define_prefix_disabled(policy) -> None:
    unit = Unit('m').define('rvpolicy', 'review policy', prefixable=policy)
    assert Unit('rvpolicy') == Unit('review policy') == unit
    for text in ('mrvpolicy', 'millireview policy'):
        with pytest.raises(UnitSymbolError):
            Unit(text)


def test_unit_define_duplicate_spellings() -> None:
    Unit('m').define('rvpolicy', 'review policy', prefixable=('rvpolicy', 'rvpolicy'))
    assert Unit('mrvpolicy').factor == Factor(Fraction(1, 1000))
    with pytest.raises(UnitSymbolError):
        Unit('millireview policy')


@pytest.mark.parametrize('source', [Unit('m'), PI * Unit('m'), Quantity(Fraction(3, 7), 'm')])
@pytest.mark.parametrize('spelling', ['rvselect', 'rvselected', 'review selected', 'review alias'])
def test_unit_define_string_policy(source, spelling: str) -> None:
    unit = source.define(
        'rvselect',
        'review selected',
        'review alias',
        symbol_aliases='rvselected',
        prefixable=spelling,
    )
    for text in ('rvselect', 'rvselected', 'review selected', 'review alias'):
        assert Unit(text) == unit
        prefixed = ('kilo' if ' ' in text else 'k') + text
        if text == spelling:
            assert Unit(prefixed).factor == 1000 * unit.factor
            assert Unit(Unit(prefixed).expression()) == Unit(prefixed)
        else:
            with pytest.raises(UnitSymbolError):
                Unit(prefixed)


@pytest.mark.parametrize('policy', [None, [], ['rvpolicy'], {}, {'rvpolicy': True}, 1])
def test_unit_define_invalid_policy(policy, unit_registry) -> None:
    with pytest.raises(TypeError, match='Prefixable must be bool, str or tuple of spellings'):
        Unit('m').define('rvpolicy', 'review policy', prefixable=policy)
    assert (UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL) == unit_registry


@pytest.mark.parametrize('text', [1, None, True, ('rvpolicy',)])
def test_unit_define_invalid_policy_type(text, unit_registry) -> None:
    with pytest.raises(TypeError, match='Prefixable spellings must be str'):
        Unit('m').define('rvpolicy', 'review policy', prefixable=('rvpolicy', text))
    assert (UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL) == unit_registry


@pytest.mark.parametrize('text', ['unknown', '', 'mrvpolicy', 'm', 'µ'])
@pytest.mark.parametrize('as_tuple', [False, True])
def test_unit_define_invalid_policy_spelling(text: str, as_tuple: bool, unit_registry) -> None:
    with pytest.raises(ValueError, match='is not a registered spelling'):
        Unit('m').define(
            'rvpolicy', 'review policy', prefixable=('rvpolicy', text) if as_tuple else text
        )
    assert (UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL) == unit_registry


@pytest.mark.parametrize('first', [True, False])
def test_unit_define_policy_conflict(first: bool) -> None:
    if first:
        Unit('m').define('rvconf', 'review collision', prefixable=('rvconf',))
        before = tuple(
            table.copy()
            for table in (UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL)
        )
        with pytest.raises(UnitRegistryError):
            Unit('s').define('mrvconf', 'review occupied', prefixable=False)
    else:
        Unit('s').define('mrvconf', 'review occupied', prefixable=False)
        before = tuple(
            table.copy()
            for table in (UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL)
        )
        with pytest.raises(UnitRegistryError):
            Unit('m').define('rvconf', 'review collision', prefixable=('rvconf',))
    assert (UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL) == before


@pytest.mark.parametrize('text', ['rv[unit', 'rv]unit', '[rvunit]', 'rv²', 'rv⁻¹'])
@pytest.mark.parametrize('kind', ['symbol', 'name', 'symbol_alias', 'name_alias'])
def test_unit_define_invalid_reserved(text: str, kind: str, unit_registry) -> None:
    with pytest.raises(ValueError):
        Unit('m').define(
            text if kind == 'symbol' else 'rvreserved',
            text if kind == 'name' else 'review reserved',
            *(text,) if kind == 'name_alias' else (),
            symbol_aliases=text if kind == 'symbol_alias' else (),
        )
    assert (UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL) == unit_registry


@pytest.mark.parametrize('name', ['review-name', 'review-name measure', 'review-name-measure'])
def test_unit_define_hyphenated_name(name: str) -> None:
    unit = Unit('m').define('rvhyphen', name, 'review alternative-name')
    assert SingleUnit(name).symbol == 'rvhyphen'
    assert Unit(name) == unit
    assert Unit('[review alternative-name]') == unit
    assert Unit('[kilo' + name + '] / s') == Unit('krvhyphen/s')
    assert Unit('kilo' + name) == Unit('krvhyphen')
    assert Unit('[' + name + ']**(1/3)') ** 3 == unit
    with pytest.raises(UnitSymbolError):
        Unit(name + ' / s')


@pytest.mark.parametrize(
    'name', ['-review', 'review-', 'review--name', 'review -name', 'review- name']
)
def test_unit_define_invalid_hyphen(name: str, unit_registry) -> None:
    with pytest.raises(ValueError):
        Unit('m').define('rvhyphen', name)
    assert (UNIT_DEFINITIONS, UNIT_NAME_TO_SYMBOL, UNIT_SYMBOL_ALIAS_TO_SYMBOL) == unit_registry


def test_unit_parser_reference_retry() -> None:
    parser = UnitParser('[review pending] / s')
    with pytest.raises(UnitSymbolError):
        parser.parse()
    unit = Unit('m').define('rvpending', 'review pending')
    result = parser.parse()
    assert result is parser.parse()
    assert result == UnitParser('rvpending/s').parse()
    assert Unit('[review pending]') == unit
