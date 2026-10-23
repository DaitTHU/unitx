"""Immutable SI prefix specifications, independent of unit definitions and registries."""

from itertools import chain

from ._immutable import _Immutable
from .factor import Factor, one, ten


class PrefixDefinition(_Immutable):
    """Immutable canonical symbol, exact scale, and accepted prefix names."""

    __slots__ = ('symbol', 'symbol_aliases', 'names', 'factor')

    symbol: str
    symbol_aliases: tuple[str, ...]
    names: tuple[str, ...]
    factor: Factor

    def __init__(
        self,
        symbol: str,
        name: str,
        *name_aliases: str,
        factor: Factor,
        symbol_aliases: str | tuple[str, ...] = (),
    ) -> None:
        if isinstance(symbol_aliases, str):
            symbol_aliases = (symbol_aliases,)
        object.__setattr__(self, 'symbol', symbol)
        object.__setattr__(self, 'symbol_aliases', symbol_aliases)
        object.__setattr__(self, 'names', (name, *name_aliases))
        object.__setattr__(self, 'factor', factor)

    @property
    def name(self) -> str:
        return self.names[0]


NO_PREFIX = PrefixDefinition('', '', factor=one)
"""Identity prefix for unprefixed units."""


_PREFIXED_NAME_EXCEPTIONS = {'megaohm': 'megohm', 'kiloohm': 'kilohm'}
_PREFIXED_NAME_EXPANSIONS = {
    contracted: expanded for expanded, contracted in _PREFIXED_NAME_EXCEPTIONS.items()
}


_PREFIX_ENTRIES: tuple[PrefixDefinition, ...] = (
    # Decimal multiples.
    PrefixDefinition('Q', 'quetta', factor=ten**30),
    PrefixDefinition('R', 'ronna', factor=ten**27),
    PrefixDefinition('Y', 'yotta', factor=ten**24),
    PrefixDefinition('Z', 'zetta', factor=ten**21),
    PrefixDefinition('E', 'exa', factor=ten**18),
    PrefixDefinition('P', 'peta', factor=ten**15),
    PrefixDefinition('T', 'tera', factor=ten**12),
    PrefixDefinition('G', 'giga', factor=ten**9),
    PrefixDefinition('M', 'mega', factor=ten**6),
    PrefixDefinition('k', 'kilo', factor=ten**3),
    PrefixDefinition('h', 'hecto', factor=ten**2),
    PrefixDefinition('da', 'deca', factor=ten**1),
    # Decimal submultiples.
    PrefixDefinition('d', 'deci', factor=ten**-1),
    PrefixDefinition('c', 'centi', factor=ten**-2),
    PrefixDefinition('m', 'milli', factor=ten**-3),
    PrefixDefinition('µ', 'micro', factor=ten**-6, symbol_aliases=('μ', 'u')),
    PrefixDefinition('n', 'nano', factor=ten**-9),
    PrefixDefinition('p', 'pico', factor=ten**-12),
    PrefixDefinition('f', 'femto', factor=ten**-15),
    PrefixDefinition('a', 'atto', factor=ten**-18),
    PrefixDefinition('z', 'zepto', factor=ten**-21),
    PrefixDefinition('y', 'yocto', factor=ten**-24),
    PrefixDefinition('r', 'ronto', factor=ten**-27),
    PrefixDefinition('q', 'quecto', factor=ten**-30),
)

PREFIX_DEFINITIONS: dict[str, PrefixDefinition] = {
    definition.symbol: definition for definition in _PREFIX_ENTRIES
}
"""Canonical prefix symbol -> prefix definition."""

PREFIX_NAME_TO_SYMBOL: dict[str, str] = {
    name: definition.symbol for definition in _PREFIX_ENTRIES for name in definition.names if name
}
"""Prefix name or name alias -> canonical prefix symbol."""

_PREFIX_NAME_LENGTHS = tuple(sorted({len(name) for name in PREFIX_NAME_TO_SYMBOL}))

PREFIX_SYMBOL_ALIAS_TO_SYMBOL: dict[str, str] = {
    alias: definition.symbol
    for definition in _PREFIX_ENTRIES
    for alias in definition.symbol_aliases
}
"""Prefix symbol alias -> canonical prefix symbol."""

_PREFIX_SYMBOL_LENGTHS = tuple(
    sorted({len(symbol) for symbol in chain(PREFIX_DEFINITIONS, PREFIX_SYMBOL_ALIAS_TO_SYMBOL)})
)

assert PREFIX_SYMBOL_ALIAS_TO_SYMBOL['u'] == chr(0xB5)
assert PREFIX_SYMBOL_ALIAS_TO_SYMBOL[chr(0x3BC)] == chr(0xB5)


def _format_prefixed_name(prefix: PrefixDefinition, name: str) -> str:
    prefixed_name = prefix.name + name
    return _PREFIXED_NAME_EXCEPTIONS.get(prefixed_name, prefixed_name)
