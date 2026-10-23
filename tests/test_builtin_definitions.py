from fractions import Fraction

import pytest

from unitx._registry.prefixes import (
    NO_PREFIX,
    PREFIX_DEFINITIONS,
    PREFIX_NAME_TO_SYMBOL,
    PREFIX_SYMBOL_ALIAS_TO_SYMBOL,
)
from unitx.dimension import DIMENSIONLESS, Dimension
from unitx.unitregistry import default_registry


@pytest.mark.parametrize(
    ('symbol', 'name', 'exponent'),
    [
        ('Q', 'quetta', 30),
        ('R', 'ronna', 27),
        ('Y', 'yotta', 24),
        ('Z', 'zetta', 21),
        ('E', 'exa', 18),
        ('P', 'peta', 15),
        ('T', 'tera', 12),
        ('G', 'giga', 9),
        ('M', 'mega', 6),
        ('k', 'kilo', 3),
        ('h', 'hecto', 2),
        ('da', 'deca', 1),
        ('d', 'deci', -1),
        ('c', 'centi', -2),
        ('m', 'milli', -3),
        ('µ', 'micro', -6),
        ('n', 'nano', -9),
        ('p', 'pico', -12),
        ('f', 'femto', -15),
        ('a', 'atto', -18),
        ('z', 'zepto', -21),
        ('y', 'yocto', -24),
        ('r', 'ronto', -27),
        ('q', 'quecto', -30),
    ],
)
def test_prefix_definition_values(symbol: str, name: str, exponent: int) -> None:
    definition = PREFIX_DEFINITIONS[symbol]
    assert definition.name == name
    assert definition.factor.value == Fraction(10) ** exponent
    assert type(definition.factor.value) is (int if exponent >= 0 else Fraction)
    if name:
        assert PREFIX_NAME_TO_SYMBOL[name] == symbol
    else:
        assert name not in PREFIX_NAME_TO_SYMBOL


def test_no_prefix_identity() -> None:
    assert NO_PREFIX.symbol == NO_PREFIX.name == ''
    assert NO_PREFIX.symbol_aliases == ()
    assert NO_PREFIX.factor.value == 1
    assert type(NO_PREFIX.factor.value) is int
    assert '' not in PREFIX_DEFINITIONS
    assert '' not in PREFIX_NAME_TO_SYMBOL
    assert '' not in PREFIX_SYMBOL_ALIAS_TO_SYMBOL


def test_prefix_definition_aliases() -> None:
    assert PREFIX_SYMBOL_ALIAS_TO_SYMBOL['u'] == PREFIX_SYMBOL_ALIAS_TO_SYMBOL['μ'] == 'µ'


def test_unit_definition_name_alias() -> None:
    assert (
        default_registry._bindings.names['meter']
        == default_registry._bindings.names['metre']
        == 'm'
    )
    assert default_registry._bindings.definitions['m'].name == 'metre'


@pytest.mark.parametrize(
    ('symbol', 'name', 'dimension', 'factor', 'prefixable'),
    [
        ('', '', DIMENSIONLESS, 1, False),
        ('s', 'second', Dimension(T=1), 1, True),
        ('m', 'metre', Dimension(L=1), 1, True),
        ('g', 'gram', Dimension(M=1), Fraction(1, 1000), True),
        ('A', 'ampere', Dimension(I=1), 1, True),
        ('K', 'kelvin', Dimension(Theta=1), 1, True),
        ('mol', 'mole', Dimension(N=1), 1, True),
        ('cd', 'candela', Dimension(J=1), 1, True),
    ],
)
def test_unit_definition_values(
    symbol: str, name: str, dimension: Dimension, factor: int | Fraction, prefixable: bool
) -> None:
    definition = default_registry._bindings.definitions[symbol]
    assert definition.name == name
    assert definition.dimension == dimension
    assert definition.factor.value == factor
    assert type(definition.factor.value) is type(factor)
    assert definition.prefixable is prefixable
    if name:
        assert default_registry._bindings.names[name] == symbol
    else:
        assert name not in default_registry._bindings.names
