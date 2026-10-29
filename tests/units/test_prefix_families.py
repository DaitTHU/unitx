"""Per-definition prefix families compose with spelling permissions."""

import pickle

import pytest

from unitx import Registry
from unitx.exceptions import UnitRegistryError, UnitSymbolError
from unitx.si import si
from unitx.unitdefinition import PrefixFamily, UnitDefinition


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


@pytest.mark.parametrize('rule', ['binary', False, 1, None])
def test_unit_definition_prefix_mapping_value_invalid(rule: object) -> None:
    registry = Registry()
    with pytest.raises(TypeError, match='Prefix mapping values'):
        registry.define(si.m, 'span', prefixable={'span': rule})  # pyright: ignore[reportArgumentType]
    with pytest.raises(UnitSymbolError):
        registry.parse('span')


@pytest.mark.parametrize('policy', [None, 2, ['binary']])
def test_unit_definition_prefix_policy_type_invalid(policy: object) -> None:
    with pytest.raises(TypeError, match='Prefixable must be'):
        UnitDefinition('span', prefixable=policy)  # pyright: ignore[reportArgumentType]
