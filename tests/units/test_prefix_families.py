"""Per-definition prefix families compose with spelling permissions."""

import copy
import pickle

import pytest

from unitx import Registry, Unit
from unitx.exceptions import UnitRegistryError, UnitSymbolError
from unitx.serialization import from_dict, to_dict
from unitx.si import si
from unitx.unitdefinition import PrefixFamily, PrefixSelection, UnitDefinition


@pytest.mark.parametrize('family', list(PrefixFamily))
def test_unit_definition_prefix_families(family: PrefixFamily) -> None:
    registry = Registry()
    unit = registry.define(
        si.m,
        'span',
        'survey span',
        symbol_aliases='sp',
        prefixable={'sp': family, 'survey span': family},
    )
    for prefix, name, scale, enabled in (
        ('k', 'kilo', 1000, family is not PrefixFamily.BINARY),
        ('Ki', 'kibi', 1024, family is not PrefixFamily.DECIMAL),
    ):
        if enabled:
            parsed = registry.parse(prefix + 'sp')
            assert parsed == registry.parse('[' + name + 'survey span]')
            assert parsed.factor / unit.factor == scale
            assert registry.parse(parsed.expression()) == parsed
            assert parsed.expression() == prefix + 'sp'
            assert pickle.loads(pickle.dumps(parsed)) == parsed
        else:
            with pytest.raises(UnitSymbolError):
                registry.parse(prefix + 'sp')
        with pytest.raises(UnitSymbolError):
            registry.parse(prefix + 'span')


def test_registry_binary_collision_unchanged() -> None:
    registry = Registry()
    occupied = registry.define(si.s, 'Kispan', prefixable=False)
    decimal = registry.define(si.m, 'span')
    assert registry.parse('Kispan') == occupied
    assert registry.parse('kspan').factor == 1000 * decimal.factor
    other = Registry((occupied,))
    with pytest.raises(UnitRegistryError):
        other.define(si.m, 'span', prefixable=PrefixFamily.ALL)
    assert other.parse('Kispan') == occupied
    with pytest.raises(UnitSymbolError):
        other.parse('span')


def test_registry_binary_name_collision_unchanged() -> None:
    registry = Registry()
    occupied = registry.define(si.s, 'old', 'kibisurvey span', prefixable=False)
    with pytest.raises(UnitRegistryError):
        registry.define(si.m, 'span', 'survey span', prefixable=PrefixFamily.BINARY)
    assert registry.parse('kibisurvey span') == occupied
    with pytest.raises(UnitSymbolError):
        registry.parse('span')


def test_unit_definition_prefix_family_inclusion() -> None:
    source = Registry()
    with source.defining():
        unit = (2 * si.m).define('span', prefixable=PrefixFamily.BINARY)
    target = Registry((unit,))
    assert target.parse('Kispan') == source.parse('Kispan')
    with pytest.raises(UnitSymbolError):
        target.parse('kspan')
    assert (1023 * unit).compact().unit == unit
    assert (1024 * unit).compact().unit == target.parse('Kispan')


def test_unit_definition_prefix_mapping() -> None:
    registry = Registry()
    policy = {'span': True, 'sp': PrefixFamily.BINARY, 'survey span': PrefixFamily.ALL}
    unit = registry.define(si.m, 'span', 'survey span', symbol_aliases='sp', prefixable=policy)
    policy.clear()
    decimal = registry.parse('kspan')
    binary = registry.parse('Kisp')
    assert decimal == registry.parse('[kilosurvey span]')
    assert binary == registry.parse('[kibisurvey span]')
    assert decimal.expression() == 'kspan'
    assert binary.expression() == 'Kisp'
    assert binary.factor == 1024 * unit.factor
    for text in ('Kispan', 'ksp'):
        with pytest.raises(UnitSymbolError):
            registry.parse(text)
    for value in (decimal, binary):
        assert registry.parse(value.expression()) == value
        assert pickle.loads(pickle.dumps(value)) == value


def test_unit_definition_prefix_name_fallback() -> None:
    registry = Registry()
    registry.define(
        si.m,
        'span',
        'survey span',
        prefixable={'span': True, 'survey span': PrefixFamily.BINARY},
    )
    assert registry.parse('[kibisurvey span]').expression() == '[kibisurvey span]'
    assert registry.parse('kspan').expression() == 'kspan'


def test_unit_definition_prefix_policy_readonly() -> None:
    policy = {'span': PrefixFamily.ALL}
    definition = UnitDefinition('span', prefixable=policy)
    policy.clear()
    assert definition.allows_prefix('span')
    with pytest.raises(TypeError):
        definition.prefixable['span'] = True  # pyright: ignore[reportIndexIssue]


@pytest.mark.parametrize('family', list(PrefixFamily))
def test_registry_prefix_family_default(family: PrefixFamily) -> None:
    registry = Registry(prefixable=family)
    with registry.defining():
        unit = (2 * si.m).define('span')
    for text, scale, enabled in (
        ('kspan', 1000, family is not PrefixFamily.BINARY),
        ('Kispan', 1024, family is not PrefixFamily.DECIMAL),
    ):
        if enabled:
            assert registry.parse(text).factor == scale * unit.factor
        else:
            with pytest.raises(UnitSymbolError):
                registry.parse(text)


@pytest.mark.parametrize('rule', [1, None, []])
def test_unit_definition_prefix_mapping_value_invalid(rule: object) -> None:
    registry = Registry()
    with pytest.raises(TypeError, match='Prefix selection must be'):
        registry.define(si.m, 'span', prefixable={'span': rule})  # pyright: ignore[reportArgumentType]
    with pytest.raises(UnitSymbolError):
        registry.parse('span')


@pytest.mark.parametrize('policy', [None, 2, ['binary']])
def test_unit_definition_prefix_policy_type_invalid(policy: object) -> None:
    with pytest.raises(TypeError, match='Prefix selection must be'):
        UnitDefinition('span', prefixable=policy)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize('selection', ['kMG', ('k', 'M', 'G'), {'k', 'M', 'G'}, frozenset('kMG')])
def test_unit_definition_prefix_subset(selection: PrefixSelection) -> None:
    registry = Registry(prefixable=selection)
    unit = registry.define(si.m, 'span', 'survey span')
    for symbol, name, scale in [('k', 'kilo', 1000), ('M', 'mega', 10**6), ('G', 'giga', 10**9)]:
        prefixed = registry.parse(symbol + 'span')
        assert prefixed == registry.parse('[' + name + 'survey span]')
        assert 1 * prefixed == scale * unit
        for restored in (
            copy.copy(prefixed),
            copy.deepcopy(prefixed),
            pickle.loads(pickle.dumps(prefixed)),
            from_dict(to_dict(prefixed), registry=registry),
            eval(repr(prefixed), {'Unit': Unit}),
        ):
            assert restored == prefixed
            assert hash(restored) == hash(prefixed)
        base = next(iter(prefixed._monomial.components()))[0]
        assert base.without_prefix() == next(iter(unit._monomial.components()))[0]
    for text in ('mspan', 'Tspan', 'Kispan', '[millisurvey span]', 'kkspan'):
        with pytest.raises(UnitSymbolError):
            registry.parse(text)
    assert (10**12 * unit).compact().unit == registry.parse('Gspan')
    assert (0.001 * unit).compact().unit == unit


def test_unit_definition_prefix_set_normalization() -> None:
    decimal = frozenset(
        (
            'Q',
            'R',
            'Y',
            'Z',
            'E',
            'P',
            'T',
            'G',
            'M',
            'k',
            'h',
            'da',
            'd',
            'c',
            'm',
            'µ',
            'n',
            'p',
            'f',
            'a',
            'z',
            'y',
            'r',
            'q',
        )
    )
    assert UnitDefinition('span').prefixable == decimal
    assert UnitDefinition('span', prefixable=PrefixFamily.DECIMAL).prefixable == decimal
    assert UnitDefinition('span', prefixable='daKiM').prefixable == {'da', 'Ki', 'M'}
    assert UnitDefinition('span', prefixable=('d', 'a')).prefixable == {'d', 'a'}
    assert UnitDefinition('span', prefixable='uµμ').prefixable == {'µ'}
    assert not UnitDefinition('span').allows_prefix('unknown')


def test_unit_definition_prefix_subset_mapping() -> None:
    symbols = {'k', 'Ki'}
    policy: dict[str, PrefixSelection] = {'sp': symbols, 'survey span': 'M', 'span': False}
    registry = Registry()
    unit = registry.define(si.m, 'span', 'survey span', symbol_aliases='sp', prefixable=policy)
    symbols.clear()
    policy.clear()
    assert registry.parse('ksp').expression() == 'ksp'
    assert registry.parse('Kisp').expression() == 'Kisp'
    assert registry.parse('[megasurvey span]').expression() == '[megasurvey span]'
    assert (10**6 * unit).compact().unit == registry.parse('[megasurvey span]')
    for text in ('kspan', 'Msp', '[kilosurvey span]'):
        with pytest.raises(UnitSymbolError):
            registry.parse(text)


@pytest.mark.parametrize('selection', ['unknown', ('kilo',), ('',), ('kMG',), {'bad'}])
def test_unit_definition_prefix_symbols_invalid(selection: PrefixSelection) -> None:
    registry = Registry()
    with pytest.raises(ValueError, match='Unknown prefix symbol'):
        registry.define(si.m, 'span', prefixable=selection)
    with pytest.raises(UnitSymbolError):
        registry.parse('span')


@pytest.mark.parametrize('selection', [False, '', (), set(), frozenset(), {}])
def test_unit_definition_empty_prefix_selection(selection: object) -> None:
    registry = Registry()
    unit = registry.define(si.m, 'span', prefixable=selection)  # pyright: ignore[reportArgumentType]
    assert registry.parse('span') == unit
    with pytest.raises(UnitSymbolError):
        registry.parse('kspan')


def test_registry_prefix_subset_collision_unchanged() -> None:
    registry = Registry()
    occupied = registry.define(si.s, 'mspan', prefixable=False)
    unit = registry.define(si.m, 'span', prefixable='kMG')
    assert registry.parse('mspan') == occupied
    assert registry.parse('kspan').factor == 1000 * unit.factor
    other = Registry((occupied,))
    with pytest.raises(UnitRegistryError):
        other.define(si.m, 'span', prefixable='km')
    assert other.parse('mspan') == occupied
    with pytest.raises(UnitSymbolError):
        other.parse('kspan')


@pytest.mark.parametrize(
    'policy',
    [
        True,
        PrefixFamily.DECIMAL,
        ('span', 'survey span'),
        (('span', True), ('survey span', PrefixFamily.DECIMAL)),
    ],
)
def test_unit_definition_previous_prefix_metadata(policy: object) -> None:
    from unitx._registry.reconstruction import _definition_state, _matches_state

    state = _definition_state(UnitDefinition('span', 'survey span'))
    assert _matches_state(state, state._replace(prefixable=policy))
    assert not _matches_state(state, state._replace(prefixable=False))
    assert not _matches_state(state, state._replace(prefixable=frozenset('kMG')))


def test_quantity_compact_subset_without_kilo() -> None:
    registry = Registry()
    unit = registry.define(si.m, 'span', prefixable=('M', 'Ki'))
    assert (1023 * unit).compact().unit == registry.parse('Kispan')
