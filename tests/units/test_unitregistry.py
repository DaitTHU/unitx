"""Registration policies, spelling fallbacks, and failure invariants."""

from __future__ import annotations

import copy
import pickle
from collections.abc import Iterator
from fractions import Fraction

import pytest

from unitx._numerics import Magnitude
from unitx._registry.prefixes import NO_PREFIX, PREFIX_DEFINITIONS, PrefixDefinition
from unitx.dimension import Dimension
from unitx.exceptions import UnitRegistryError, UnitSymbolError
from unitx.factor import Factor, pi
from unitx.quantity import Quantity
from unitx.singleunit import SingleUnit
from unitx.unit import Unit
from unitx.unitdefinition import UnitDefinition
from unitx.unitparser import UnitParser
from unitx.unitregistry import default_registry

type RegistrySnapshot = tuple[dict[str, UnitDefinition], dict[str, str], dict[str, str]]


@pytest.fixture(autouse=True)
def unit_registry() -> Iterator[RegistrySnapshot]:
    tables = (
        default_registry._bindings.definitions,
        default_registry._bindings.names,
        default_registry._bindings.aliases,
    )
    snapshots = tuple(table.copy() for table in tables)
    yield snapshots
    for table, snapshot in zip(tables, snapshots):
        table.clear()
        table.update(snapshot)


@pytest.mark.parametrize('source', [Unit('m'), pi * Unit('m'), Quantity(Fraction(3, 7), 'm')])
def test_unit_define_binding_policy(source: Unit | Quantity[Magnitude]) -> None:
    unit = default_registry.define(
        source,
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
    unit = default_registry.define(
        Unit('m'),
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
    default_registry.define(
        Unit('m'),
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
        assert eval(repr(value), globals() | {'Unit': Unit}) == value
        assert copy.copy(value) == copy.deepcopy(value) == value
        restored = pickle.loads(pickle.dumps(value))
        assert restored == value
        assert hash(restored) == hash(value)
        assert {value: 'found'}[restored] == 'found'
    with pytest.raises(UnitSymbolError):
        Unit('krvnameonly')


def test_single_unit_symbol_fallback() -> None:
    default_registry.define(
        Unit('s'), 'rvonlysymbol', 'review symbol', prefixable=('rvonlysymbol',)
    )
    single = SingleUnit('krvonlysymbol')
    assert single.name == single.symbol == 'krvonlysymbol'
    assert SingleUnit(single.name) == single
    assert Unit('krvonlysymbol').expression() == 'krvonlysymbol'
    with pytest.raises(UnitSymbolError):
        Unit('[kiloreview symbol]')


@pytest.mark.parametrize('allowed', ['rvallowed', 'review permitted'])
def test_single_unit_empty_aliases(allowed: str) -> None:
    default_registry.define(
        Unit('m'),
        'rvempty',
        'review empty',
        '',
        'review permitted',
        symbol_aliases=('', 'rvallowed'),
        prefixable=allowed,
    )
    spelling = 'krvallowed' if allowed == 'rvallowed' else 'kiloreview permitted'
    single = SingleUnit(spelling)
    assert single.symbol == single.name == spelling
    assert single.without_prefix() == SingleUnit('rvempty')
    assert Unit(Unit(spelling).expression()) == Unit(spelling)
    assert pickle.loads(pickle.dumps(single)) == single


@pytest.mark.parametrize('policy', [False, ()])
def test_unit_define_prefix_disabled(policy: bool | tuple[object, ...]) -> None:
    unit = default_registry.define(Unit('m'), 'rvpolicy', 'review policy', prefixable=policy)
    assert Unit('rvpolicy') == Unit('review policy') == unit
    for text in ('mrvpolicy', 'millireview policy'):
        with pytest.raises(UnitSymbolError):
            Unit(text)


def test_unit_define_duplicate_spellings() -> None:
    default_registry.define(
        Unit('m'), 'rvpolicy', 'review policy', prefixable=('rvpolicy', 'rvpolicy')
    )
    assert Unit('mrvpolicy').factor == Factor(Fraction(1, 1000))
    with pytest.raises(UnitSymbolError):
        Unit('millireview policy')


@pytest.mark.parametrize('source', [Unit('m'), pi * Unit('m'), Quantity(Fraction(3, 7), 'm')])
@pytest.mark.parametrize('spelling', ['rvselect', 'rvselected', 'review selected', 'review alias'])
def test_unit_define_string_policy(source: Unit | Quantity[Magnitude], spelling: str) -> None:
    unit = default_registry.define(
        source,
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


@pytest.mark.parametrize('policy', [[], ['rvpolicy'], 1])
def test_unit_define_invalid_policy(
    policy: list[object] | list[str] | dict[str, object] | int,
    unit_registry: RegistrySnapshot,
) -> None:
    with pytest.raises(TypeError, match='Prefixable must be bool, str or tuple of spellings'):
        default_registry.define(Unit('m'), 'rvpolicy', 'review policy', prefixable=policy)
    assert (
        default_registry._bindings.definitions,
        default_registry._bindings.names,
        default_registry._bindings.aliases,
    ) == unit_registry


@pytest.mark.parametrize('text', [1, None, True, ('rvpolicy',)])
def test_unit_define_invalid_policy_type(
    text: int | None | bool | tuple[str, ...],
    unit_registry: RegistrySnapshot,
) -> None:
    with pytest.raises(TypeError, match='Prefixable spellings must be str'):
        default_registry.define(
            Unit('m'), 'rvpolicy', 'review policy', prefixable=('rvpolicy', text)
        )
    assert (
        default_registry._bindings.definitions,
        default_registry._bindings.names,
        default_registry._bindings.aliases,
    ) == unit_registry


@pytest.mark.parametrize('text', ['unknown', '', 'mrvpolicy', 'm', 'µ'])
@pytest.mark.parametrize('as_tuple', [False, True])
def test_unit_define_invalid_policy_spelling(
    text: str,
    as_tuple: bool,
    unit_registry: RegistrySnapshot,
) -> None:
    with pytest.raises(ValueError, match='is not a registered spelling'):
        default_registry.define(
            Unit('m'),
            'rvpolicy',
            'review policy',
            prefixable=('rvpolicy', text) if as_tuple else text,
        )
    assert (
        default_registry._bindings.definitions,
        default_registry._bindings.names,
        default_registry._bindings.aliases,
    ) == unit_registry


@pytest.mark.parametrize('first', [True, False])
def test_unit_define_policy_conflict(first: bool) -> None:
    if first:
        default_registry.define(Unit('m'), 'rvconf', 'review collision', prefixable=('rvconf',))
        before = tuple(
            table.copy()
            for table in (
                default_registry._bindings.definitions,
                default_registry._bindings.names,
                default_registry._bindings.aliases,
            )
        )
        with pytest.raises(UnitRegistryError):
            default_registry.define(Unit('s'), 'mrvconf', 'review occupied', prefixable=False)
    else:
        default_registry.define(Unit('s'), 'mrvconf', 'review occupied', prefixable=False)
        before = tuple(
            table.copy()
            for table in (
                default_registry._bindings.definitions,
                default_registry._bindings.names,
                default_registry._bindings.aliases,
            )
        )
        with pytest.raises(UnitRegistryError):
            default_registry.define(Unit('m'), 'rvconf', 'review collision', prefixable=('rvconf',))
    assert (
        default_registry._bindings.definitions,
        default_registry._bindings.names,
        default_registry._bindings.aliases,
    ) == before


@pytest.mark.parametrize('text', ['rv[unit', 'rv]unit', '[rvunit]', 'rv²', 'rv⁻¹'])
@pytest.mark.parametrize('kind', ['symbol', 'name', 'symbol_alias', 'name_alias'])
def test_unit_define_invalid_reserved(
    text: str,
    kind: str,
    unit_registry: RegistrySnapshot,
) -> None:
    with pytest.raises(ValueError):
        default_registry.define(
            Unit('m'),
            text if kind == 'symbol' else 'rvreserved',
            text if kind == 'name' else 'review reserved',
            *(text,) if kind == 'name_alias' else (),
            symbol_aliases=text if kind == 'symbol_alias' else (),
        )
    assert (
        default_registry._bindings.definitions,
        default_registry._bindings.names,
        default_registry._bindings.aliases,
    ) == unit_registry


@pytest.mark.parametrize('name', ['review-name', 'review-name measure', 'review-name-measure'])
def test_unit_define_hyphenated_name(name: str) -> None:
    unit = default_registry.define(Unit('m'), 'rvhyphen', name, 'review alternative-name')
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
def test_unit_define_invalid_hyphen(name: str, unit_registry: RegistrySnapshot) -> None:
    with pytest.raises(ValueError):
        default_registry.define(Unit('m'), 'rvhyphen', name)
    assert (
        default_registry._bindings.definitions,
        default_registry._bindings.names,
        default_registry._bindings.aliases,
    ) == unit_registry


def test_unit_parser_reference_retry() -> None:
    parser = UnitParser('[review pending] / s')
    with pytest.raises(UnitSymbolError):
        parser.parse()
    unit = default_registry.define(Unit('m'), 'rvpending', 'review pending')
    result = parser.parse()
    assert result is parser.parse()
    assert result == UnitParser('rvpending/s').parse()
    assert Unit('[review pending]') == unit


@pytest.mark.parametrize(
    ('definition', 'attribute'),
    [
        *(
            (PREFIX_DEFINITIONS['k'], field)
            for field in ('symbol', 'symbol_aliases', 'names', 'factor')
        ),
        *(
            (default_registry._bindings.definitions['m'], field)
            for field in ('symbols', 'names', 'factor', 'dimension', 'prefixable')
        ),
    ],
)
def test_unit_definition_immutable(
    definition: PrefixDefinition | UnitDefinition, attribute: str
) -> None:
    single = SingleUnit('km')
    before = single.symbol, single.name, single.factor, single.dimension, hash(single)
    with pytest.raises(AttributeError):
        setattr(definition, attribute, None)
    with pytest.raises(AttributeError):
        delattr(definition, attribute)
    assert (single.symbol, single.name, single.factor, single.dimension, hash(single)) == before
    assert {single: 'distance'}[SingleUnit('kilometre')] == 'distance'


def test_single_unit_registration_stability() -> None:
    default_registry.define(Unit('m'), 'rvstable', 'review stable', prefixable='review stable')
    single = SingleUnit('kiloreview stable')
    lookup = {single: 'original'}
    before = single.symbol, single.name, single.factor, single.dimension, hash(single)
    default_registry.define(Unit('s'), 'rvadditional', 'review additional')
    with pytest.raises(UnitRegistryError):
        default_registry.define(Unit('s'), 'rvstable', 'review replacement')
    assert (single.symbol, single.name, single.factor, single.dimension, hash(single)) == before
    assert lookup[SingleUnit(single.symbol)] == 'original'
    assert pickle.loads(pickle.dumps(single)) == single
    assert single.without_prefix() == SingleUnit('rvstable')
    assert Unit(Unit(single.symbol).expression()) == Unit('kiloreview stable')
    assert SingleUnit('rvadditional').dimension == Unit('s').dimension


@pytest.mark.parametrize(
    ('aliases', 'expected'),
    [
        ('', ('',)),
        ('rvalias', ('rvalias',)),
        ((), ()),
        (('rvfirst', 'rvsecond'), ('rvfirst', 'rvsecond')),
    ],
)
def test_prefix_definition_aliases(
    aliases: str | tuple[str, ...], expected: tuple[str, ...]
) -> None:
    definition = PrefixDefinition(
        'rvprefix', 'review prefix', factor=Factor(10), symbol_aliases=aliases
    )
    assert definition.symbol_aliases == expected


@pytest.mark.parametrize(
    ('aliases', 'expected'),
    [
        ('', ('',)),
        ('rvalias', ('rvalias',)),
        ((), ()),
        (('rvfirst', 'rvsecond'), ('rvfirst', 'rvsecond')),
    ],
)
def test_unit_definition_aliases(aliases: str | tuple[str, ...], expected: tuple[str, ...]) -> None:
    definition = UnitDefinition(
        'rvunit', 'review unit', factor=Factor(1), dimension=Dimension(L=1), symbol_aliases=aliases
    )
    assert definition.symbols == ('rvunit', *expected)


def test_unit_definition_defaults() -> None:
    first, second = UnitDefinition('span'), UnitDefinition('span')
    assert first.symbol == first.name == 'span'
    assert first.factor == Factor(1)
    assert first.dimension == Dimension()
    assert first.dimension == first.dimension.physical
    assert not hasattr(first, 'coordinate')
    assert first != second and len({first, second}) == 2


@pytest.mark.parametrize('symbol', ['', 'bad token', 'm2', 'm/s'])
def test_unit_definition_invalid_symbol(symbol: str) -> None:
    with pytest.raises(ValueError, match='Unit symbol must be a single printable unit token'):
        UnitDefinition(symbol)


@pytest.mark.parametrize('factor', [None, 0, 2.0, '2'])
def test_unit_definition_invalid_factor(factor: None | int | float | str) -> None:
    with pytest.raises(TypeError, match='Unit definition factor must be a Factor'):
        UnitDefinition('span', factor=factor)


@pytest.mark.parametrize('dimension', [None, 'L', (1,)])
def test_unit_definition_invalid_dimension(dimension: None | str | tuple[int, ...]) -> None:
    with pytest.raises(TypeError, match='Unit definition dimension must be a Dimension'):
        UnitDefinition('span', dimension=dimension)


def test_unit_definition_invalid_prefix_conflict() -> None:
    with pytest.raises(UnitRegistryError, match='conflicting prefix meanings'):
        UnitDefinition('span', symbol_aliases='kspan')


def test_unit_definition_independent_spellings() -> None:
    first = UnitDefinition('m', dimension=Dimension(T=1))
    second = UnitDefinition('m', dimension=Dimension(L=1))
    assert first is not second and first.dimension != second.dimension
    assert Unit('m').dimension == Dimension(L=1)


@pytest.mark.parametrize('symbol', ['Km', 'Kg', 'Krvlen'])
def test_unit_define_uppercase_symbol(symbol: str) -> None:
    # These spellings no longer have a built-in prefix meaning.
    with pytest.raises(UnitSymbolError):
        SingleUnit(symbol)
    custom = default_registry.define(Unit('s'), symbol, 'review uppercase', prefixable=False)
    assert Unit(symbol) == custom
    assert custom.dimension == Unit('s').dimension
    assert SingleUnit(symbol).prefix_symbol == ''
    assert Unit('km') == Unit('kilometre')
    assert Unit('K m') == Unit('K') * Unit('m')


@pytest.mark.parametrize(
    ('text', 'prefix', 'symbol'),
    [
        ('', '', ''),
        ('m', '', 'm'),
        ('meter', '', 'm'),
        ('kg', 'k', 'g'),
        ('kilogram', 'k', 'g'),
        ('um', 'µ', 'm'),
        ('μm', 'µ', 'm'),
        ('micrometre', 'µ', 'm'),
        ('u', '', 'Da'),
        ('megaohm', 'M', 'Ω'),
        ('megohm', 'M', 'Ω'),
        ('kilohm', 'k', 'Ω'),
    ],
)
def test_unit_resolver_definitions(text: str, prefix: str, symbol: str) -> None:
    resolved_prefix, resolved_unit = default_registry._resolve(text)
    assert resolved_prefix is (PREFIX_DEFINITIONS[prefix] if prefix else NO_PREFIX)
    assert resolved_unit is default_registry._bindings.definitions[symbol]


@pytest.mark.parametrize(
    ('text', 'prefix', 'symbol'),
    [
        ('meter', '', 'm'),
        ('kilogram', 'k', 'g'),
        ('micrometre', 'µ', 'm'),
        ('megaohm', 'M', 'Ω'),
        ('megohm', 'M', 'Ω'),
        ('kilohm', 'k', 'Ω'),
    ],
)
def test_unit_name_definitions(text: str, prefix: str, symbol: str) -> None:
    resolved_prefix, resolved_unit = default_registry._resolve_name(text)
    assert resolved_prefix is (PREFIX_DEFINITIONS[prefix] if prefix else NO_PREFIX)
    assert resolved_unit is default_registry._bindings.definitions[symbol]


def test_unit_resolver_permissions() -> None:
    default_registry.define(
        Unit('m'),
        'rvrecord',
        'review record',
        'review permitted',
        symbol_aliases='rvallowed',
        prefixable=('rvallowed', 'review permitted'),
    )
    prefix, definition = default_registry._resolve('urvallowed')
    name_prefix, name_definition = default_registry._resolve_name('microreview permitted')
    assert prefix is name_prefix is PREFIX_DEFINITIONS['µ']
    assert definition is name_definition is default_registry._bindings.definitions['rvrecord']
    for text in ('urvrecord', 'microreview record'):
        with pytest.raises(UnitSymbolError):
            default_registry._resolve(text)
    with pytest.raises(UnitSymbolError):
        default_registry._resolve_name('urvallowed')


@pytest.mark.parametrize('spelling', ['megaohm', 'megohm', 'kiloohm', 'kilohm'])
def test_unit_define_invalid_contraction(spelling: str, unit_registry: RegistrySnapshot) -> None:
    with pytest.raises(UnitRegistryError, match='already resolves to a registered unit'):
        default_registry.define(Unit('m'), spelling, 'review reserved')
    assert (
        default_registry._bindings.definitions,
        default_registry._bindings.names,
        default_registry._bindings.aliases,
    ) == unit_registry
