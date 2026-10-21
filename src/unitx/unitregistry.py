"""Unit definitions, spelling resolution, and process-local registration.

``PrefixPolicy`` is a bool or tuple of registered unit spellings. The
``prefixable`` keyword on Unit.define() enables all prefixes for every spelling
with True, disables them with False, or enables only the listed spellings.
"""

from __future__ import annotations

import re
import warnings
from itertools import chain

from ._immutable import _Immutable
from ._unicode import SUPERSCRIPTS
from .dimension import DIMENSIONLESS, Dimension
from .exceptions import (
    _WARNING_SKIP_PREFIXES,
    UnitRegistryError,
    UnitSymbolError,
    UnitSymbolWarning,
)
from .factor import ONE, TEN, Factor

__all__ = ['PrefixPolicy']

type PrefixPolicy = bool | str | tuple[str, ...]

_UNIT_SYMBOL_PATTERN = r'[^\s\d' + re.escape('*^()/+-[]' + SUPERSCRIPTS) + ']+'
_UNIT_SYMBOL_RE = re.compile(_UNIT_SYMBOL_PATTERN)
_UNIT_NAME_RE = re.compile(f'{_UNIT_SYMBOL_PATTERN}(?:[ -]{_UNIT_SYMBOL_PATTERN})*')
_PREFIXED_NAME_EXCEPTIONS = {'megaohm': 'megohm', 'kiloohm': 'kilohm'}
_PREFIXED_NAME_EXPANSIONS = {
    contracted: expanded for expanded, contracted in _PREFIXED_NAME_EXCEPTIONS.items()
}


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


NO_PREFIX = PrefixDefinition('', '', factor=ONE)
"""Identity prefix for unprefixed units."""

_PREFIX_ENTRIES: tuple[PrefixDefinition, ...] = (
    # Decimal multiples.
    PrefixDefinition('Q', 'quetta', factor=TEN**30),
    PrefixDefinition('R', 'ronna', factor=TEN**27),
    PrefixDefinition('Y', 'yotta', factor=TEN**24),
    PrefixDefinition('Z', 'zetta', factor=TEN**21),
    PrefixDefinition('E', 'exa', factor=TEN**18),
    PrefixDefinition('P', 'peta', factor=TEN**15),
    PrefixDefinition('T', 'tera', factor=TEN**12),
    PrefixDefinition('G', 'giga', factor=TEN**9),
    PrefixDefinition('M', 'mega', factor=TEN**6),
    PrefixDefinition('k', 'kilo', factor=TEN**3, symbol_aliases=('K',)),
    PrefixDefinition('h', 'hecto', factor=TEN**2),
    PrefixDefinition('da', 'deca', factor=TEN**1),
    # Decimal submultiples.
    PrefixDefinition('d', 'deci', factor=TEN**-1),
    PrefixDefinition('c', 'centi', factor=TEN**-2),
    PrefixDefinition('m', 'milli', factor=TEN**-3),
    PrefixDefinition('µ', 'micro', factor=TEN**-6, symbol_aliases=('μ', 'u')),
    PrefixDefinition('n', 'nano', factor=TEN**-9),
    PrefixDefinition('p', 'pico', factor=TEN**-12),
    PrefixDefinition('f', 'femto', factor=TEN**-15),
    PrefixDefinition('a', 'atto', factor=TEN**-18),
    PrefixDefinition('z', 'zepto', factor=TEN**-21),
    PrefixDefinition('y', 'yocto', factor=TEN**-24),
    PrefixDefinition('r', 'ronto', factor=TEN**-27),
    PrefixDefinition('q', 'quecto', factor=TEN**-30),
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


class UnitDefinition(_Immutable):
    """Immutable unit metadata and spelling policies, distinct from a formal Unit."""

    __slots__ = ('symbols', 'names', 'factor', 'dimension', 'prefixable')

    symbols: tuple[str, ...]
    names: tuple[str, ...]
    factor: Factor
    dimension: Dimension
    prefixable: bool | tuple[str, ...]

    def __init__(
        self,
        symbol: str,
        name: str,
        *name_aliases: str,
        factor: Factor,
        dimension: Dimension,
        symbol_aliases: str | tuple[str, ...] = (),
        prefixable: PrefixPolicy = True,
    ) -> None:
        if isinstance(symbol_aliases, str):
            symbol_aliases = (symbol_aliases,)
        object.__setattr__(self, 'symbols', (symbol, *symbol_aliases))
        object.__setattr__(self, 'names', (name, *name_aliases))
        object.__setattr__(self, 'factor', factor)
        object.__setattr__(self, 'dimension', dimension)
        object.__setattr__(
            self, 'prefixable', (prefixable,) if isinstance(prefixable, str) else prefixable
        )

    @property
    def symbol(self) -> str:
        return self.symbols[0]

    def allows_prefix(self, spelling: str) -> bool:
        return self.prefixable if isinstance(self.prefixable, bool) else spelling in self.prefixable

    @property
    def name(self) -> str:
        return self.names[0]


_UNIT_ENTRIES: tuple[UnitDefinition, ...] = (
    UnitDefinition('', '', factor=ONE, dimension=DIMENSIONLESS, prefixable=False),
    UnitDefinition('s', 'second', factor=ONE, dimension=Dimension(T=1)),
    UnitDefinition('m', 'metre', 'meter', factor=ONE, dimension=Dimension(L=1)),
    UnitDefinition('g', 'gram', factor=TEN**-3, dimension=Dimension(M=1)),
    UnitDefinition('A', 'ampere', factor=ONE, dimension=Dimension(I=1)),
    UnitDefinition('K', 'kelvin', factor=ONE, dimension=Dimension(Theta=1)),
    UnitDefinition('mol', 'mole', factor=ONE, dimension=Dimension(N=1)),
    UnitDefinition('cd', 'candela', factor=ONE, dimension=Dimension(J=1)),
)

UNIT_DEFINITIONS: dict[str, UnitDefinition] = {
    definition.symbol: definition for definition in _UNIT_ENTRIES
}
"""Canonical unit symbol -> unit definition."""

UNIT_NAME_TO_SYMBOL: dict[str, str] = {
    name: definition.symbol for definition in _UNIT_ENTRIES for name in definition.names if name
}
"""Unit name or name alias -> canonical unit symbol."""

UNIT_SYMBOL_ALIAS_TO_SYMBOL: dict[str, str] = {
    alias: definition.symbol
    for definition in _UNIT_ENTRIES
    for alias in definition.symbols[1:]
    if alias
}
"""Unit symbol alias -> canonical unit symbol."""


def _format_prefixed_name(prefix: PrefixDefinition, name: str) -> str:
    prefixed_name = prefix.name + name
    return _PREFIXED_NAME_EXCEPTIONS.get(prefixed_name, prefixed_name)


def _resolve_unit_name(text: str) -> tuple[PrefixDefinition, UnitDefinition]:
    """Resolve only names to shared definitions, with at most one prefix."""
    unit_symbol = UNIT_NAME_TO_SYMBOL.get(text)
    if unit_symbol is not None:
        return NO_PREFIX, UNIT_DEFINITIONS[unit_symbol]
    prefixed_name = _PREFIXED_NAME_EXPANSIONS.get(text, text)
    for prefix_len in _PREFIX_NAME_LENGTHS:
        prefix_symbol = PREFIX_NAME_TO_SYMBOL.get(prefixed_name[:prefix_len])
        if prefix_symbol is None:
            continue
        name = prefixed_name[prefix_len:]
        unit_symbol = UNIT_NAME_TO_SYMBOL.get(name)
        if unit_symbol is not None:
            unit = UNIT_DEFINITIONS[unit_symbol]
            if unit.allows_prefix(name):
                return PREFIX_DEFINITIONS[prefix_symbol], unit
    raise UnitSymbolError(f'{text!r} is not a valid unit symbol.')


def _resolve_unit(text: str, *, warn: bool = True) -> tuple[PrefixDefinition, UnitDefinition]:
    """Resolve shared definitions; complete symbols precede one prefix split."""
    unitdef = UNIT_DEFINITIONS.get(text)
    if unitdef is not None:
        return NO_PREFIX, unitdef
    unit_symbol = UNIT_SYMBOL_ALIAS_TO_SYMBOL.get(text)
    if unit_symbol is not None:
        return NO_PREFIX, UNIT_DEFINITIONS[unit_symbol]
    for prefix_len in _PREFIX_SYMBOL_LENGTHS:
        prefix = text[:prefix_len]
        prefixdef = PREFIX_DEFINITIONS.get(PREFIX_SYMBOL_ALIAS_TO_SYMBOL.get(prefix, prefix))
        if prefixdef is None:
            continue
        unit = text[prefix_len:]
        unitdef = UNIT_DEFINITIONS.get(UNIT_SYMBOL_ALIAS_TO_SYMBOL.get(unit, unit))
        if unitdef is not None and unitdef.allows_prefix(unit):
            if warn and prefix in PREFIX_SYMBOL_ALIAS_TO_SYMBOL and prefix in UNIT_DEFINITIONS:
                from .singleunit import SingleUnit

                resolved = SingleUnit._from_definitions(prefixdef, unitdef)
                preferred, name = resolved.symbol, resolved.name
                warnings.warn(
                    f'{text!r} is interpreted as {preferred!r} ({name}). '
                    f'Use {preferred!r} for {name}, or {prefix + " " + unit!r} '
                    f'for {UNIT_DEFINITIONS[prefix].name} {unitdef.name}.',
                    UnitSymbolWarning,
                    skip_file_prefixes=_WARNING_SKIP_PREFIXES,
                )
            return prefixdef, unitdef
    return _resolve_unit_name(text)


def _register_unit(definition: UnitDefinition, /) -> None:
    """Validate every binding and generated spelling before updating indexes."""
    symbol, *symbol_aliases = definition.symbols
    if not isinstance(definition.prefixable, (bool, tuple)):
        wrong_type = type(definition.prefixable).__name__
        error_message = f'Prefixable must be bool, str or tuple of spellings, got {wrong_type}.'
        raise TypeError(error_message)
    # Validate every spelling and alias, including generated prefixed spellings.
    for kind, texts in (
        ('symbol', (symbol,)),
        ('name', (definition.name,)),
        ('symbol alias', symbol_aliases),
        ('name alias', definition.names[1:]),
    ):
        for text in texts:
            if not isinstance(text, str):
                error_message = f'Unit {kind} must be str, got {type(text).__name__}.'
                raise TypeError(error_message)
            if text == '' and kind.endswith('alias'):
                continue
            is_name = kind.startswith('name')
            pattern = _UNIT_NAME_RE if is_name else _UNIT_SYMBOL_RE
            if not text.isprintable() or pattern.fullmatch(text) is None:
                if is_name:
                    rule = 'consist of printable unit tokens separated by single spaces or hyphens'
                else:
                    rule = 'be a single printable unit token'
                raise ValueError(f'Unit {kind} must {rule}, got {text!r}.')

    symbols = dict.fromkeys(text for text in (symbol, *symbol_aliases) if text)
    names = dict.fromkeys(name for name in definition.names if name)
    if isinstance(definition.prefixable, tuple):
        for text in definition.prefixable:
            if not isinstance(text, str):
                error_message = f'Prefixable spellings must be str, got {type(text).__name__}.'
                raise TypeError(error_message)
            if text not in symbols and text not in names:
                error_message = f'Prefixable spelling {text!r} is not a registered spelling.'
                raise ValueError(error_message)
    spellings = [(text, '') for text in (*symbols, *names)]
    for prefix in PREFIX_DEFINITIONS.values():
        prefix_symbol = prefix.symbol
        for text in filter(definition.allows_prefix, symbols):
            spellings.append((prefix_symbol + text, prefix_symbol))
            spellings.extend((alias + text, prefix_symbol) for alias in prefix.symbol_aliases)
        for text in filter(definition.allows_prefix, names):
            for prefix_name in prefix.names:
                prefixed_name = prefix_name + text
                spellings.append((prefixed_name, prefix_symbol))
                if exception := _PREFIXED_NAME_EXCEPTIONS.get(prefixed_name):
                    spellings.append((exception, prefix_symbol))

    spelling_prefixes: dict[str, str] = {}
    for text, prefix in spellings:
        if spelling_prefixes.setdefault(text, prefix) != prefix:
            error_message = f'Unit text {text!r} has conflicting prefix meanings.'
            raise UnitRegistryError(error_message)
        try:
            _resolve_unit(text, warn=False)
        except UnitSymbolError:
            pass
        else:
            error_message = f'Unit text {text!r} already resolves to a registered unit.'
            raise UnitRegistryError(error_message)
        if ' ' in text:
            try:
                for word in text.split(' '):
                    _resolve_unit(word, warn=False)
            except UnitSymbolError:
                continue
            error_message = f'Unit text {text!r} already parses as a unit product.'
            raise UnitRegistryError(error_message)

    UNIT_DEFINITIONS[symbol] = definition
    UNIT_NAME_TO_SYMBOL.update(dict.fromkeys(names, symbol))
    UNIT_SYMBOL_ALIAS_TO_SYMBOL.update(
        dict.fromkeys((alias for alias in symbols if alias != symbol), symbol)
    )
