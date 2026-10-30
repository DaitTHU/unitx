"""Immutable decimal and binary prefixes, independent of unit registries."""

from itertools import chain
from typing import Literal

from .._immutable import _Immutable
from ..factor import Factor, one, ten, two


class PrefixDefinition(_Immutable):
    """Immutable symbol, names, exact scale and decimal/binary family."""

    __slots__ = ('symbol', 'symbol_aliases', 'names', 'factor', 'family')

    symbol: str
    symbol_aliases: tuple[str, ...]
    names: tuple[str, ...]
    factor: Factor
    family: Literal['decimal', 'binary']

    def __init__(
        self,
        symbol: str,
        name: str,
        *name_aliases: str,
        factor: Factor,
        symbol_aliases: str | tuple[str, ...] = (),
        family: Literal['decimal', 'binary'] = 'decimal',
    ) -> None:
        if isinstance(symbol_aliases, str):
            symbol_aliases = (symbol_aliases,)
        object.__setattr__(self, 'symbol', symbol)
        object.__setattr__(self, 'symbol_aliases', symbol_aliases)
        object.__setattr__(self, 'names', (name, *name_aliases))
        object.__setattr__(self, 'factor', factor)
        object.__setattr__(self, 'family', family)

    @property
    def name(self) -> str:
        return self.names[0]


NO_PREFIX = PrefixDefinition('', '', factor=one)
"""Identity prefix for unprefixed units."""


_PREFIXED_NAME_EXCEPTIONS = {'megaohm': 'megohm', 'kiloohm': 'kilohm'}
_PREFIXED_NAME_EXPANSIONS = {
    contracted: expanded for expanded, contracted in _PREFIXED_NAME_EXCEPTIONS.items()
}


DECIMAL_PREFIXES: tuple[PrefixDefinition, ...] = (
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

BINARY_PREFIXES: tuple[PrefixDefinition, ...] = (
    # IEC 80000-13 binary multiples, also listed in the SI Brochure.
    PrefixDefinition('Ki', 'kibi', factor=two**10, family='binary'),
    PrefixDefinition('Mi', 'mebi', factor=two**20, family='binary'),
    PrefixDefinition('Gi', 'gibi', factor=two**30, family='binary'),
    PrefixDefinition('Ti', 'tebi', factor=two**40, family='binary'),
    PrefixDefinition('Pi', 'pebi', factor=two**50, family='binary'),
    PrefixDefinition('Ei', 'exbi', factor=two**60, family='binary'),
    PrefixDefinition('Zi', 'zebi', factor=two**70, family='binary'),
    PrefixDefinition('Yi', 'yobi', factor=two**80, family='binary'),
    PrefixDefinition('Ri', 'robi', factor=two**90, family='binary'),
    PrefixDefinition('Qi', 'quebi', factor=two**100, family='binary'),
)

ALL_PREFIXES = (*DECIMAL_PREFIXES, *BINARY_PREFIXES)

PREFIX_DEFINITIONS: dict[str, PrefixDefinition] = {
    definition.symbol: definition for definition in ALL_PREFIXES
}
"""Canonical prefix symbol -> prefix definition."""

PREFIX_NAME_TO_SYMBOL: dict[str, str] = {
    name: definition.symbol for definition in ALL_PREFIXES for name in definition.names if name
}
"""Prefix name or name alias -> canonical prefix symbol."""

_PREFIX_NAME_LENGTHS = tuple(sorted({len(name) for name in PREFIX_NAME_TO_SYMBOL}))

PREFIX_SYMBOL_ALIAS_TO_SYMBOL: dict[str, str] = {
    alias: definition.symbol for definition in ALL_PREFIXES for alias in definition.symbol_aliases
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
