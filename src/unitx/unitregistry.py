"""Unit definitions, spelling resolution, and process-local registration.

``PrefixPolicy`` is a bool or tuple of registered unit spellings. The
``prefixable`` keyword on Unit.define() enables all prefixes for every spelling
with True, disables them with False, or enables only the listed spellings.
"""

from __future__ import annotations

import re
import warnings

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

type PrefixPolicy = bool | tuple[str, ...]

_UNIT_SYMBOL_PATTERN = r'[^\s\d' + re.escape('*^()/+-[]' + SUPERSCRIPTS) + ']+'
_UNIT_SYMBOL_RE = re.compile(_UNIT_SYMBOL_PATTERN)
_UNIT_NAME_RE = re.compile(_UNIT_SYMBOL_PATTERN + '(?:[ -]' + _UNIT_SYMBOL_PATTERN + ')*')
_PREFIXED_NAME_EXCEPTIONS = {('M', 'ohm'): 'megohm', ('k', 'ohm'): 'kilohm'}


class PrefixDefinition:
    """Exact scale and accepted names of a decimal prefix."""

    __slots__ = ('factor', 'names')

    def __init__(self, factor: Factor, name: str, *name_aliases: str) -> None:
        self.factor = factor
        self.names = (name, *name_aliases)

    @property
    def name(self) -> str:
        return self.names[0]


_PREFIX_ENTRIES: dict[str | tuple[str, ...], PrefixDefinition] = {
    # Decimal multiples and the identity prefix.
    'Q': PrefixDefinition(TEN**30, 'quetta'),
    'R': PrefixDefinition(TEN**27, 'ronna'),
    'Y': PrefixDefinition(TEN**24, 'yotta'),
    'Z': PrefixDefinition(TEN**21, 'zetta'),
    'E': PrefixDefinition(TEN**18, 'exa'),
    'P': PrefixDefinition(TEN**15, 'peta'),
    'T': PrefixDefinition(TEN**12, 'tera'),
    'G': PrefixDefinition(TEN**9, 'giga'),
    'M': PrefixDefinition(TEN**6, 'mega'),
    ('k', 'K'): PrefixDefinition(TEN**3, 'kilo'),
    'h': PrefixDefinition(TEN**2, 'hecto'),
    'da': PrefixDefinition(TEN**1, 'deca'),
    '': PrefixDefinition(ONE, ''),
    # Decimal submultiples.
    'd': PrefixDefinition(TEN**-1, 'deci'),
    'c': PrefixDefinition(TEN**-2, 'centi'),
    'm': PrefixDefinition(TEN**-3, 'milli'),
    ('µ', 'μ', 'u'): PrefixDefinition(TEN**-6, 'micro'),  # chr(0xB5), chr(0x3BC)
    'n': PrefixDefinition(TEN**-9, 'nano'),
    'p': PrefixDefinition(TEN**-12, 'pico'),
    'f': PrefixDefinition(TEN**-15, 'femto'),
    'a': PrefixDefinition(TEN**-18, 'atto'),
    'z': PrefixDefinition(TEN**-21, 'zepto'),
    'y': PrefixDefinition(TEN**-24, 'yocto'),
    'r': PrefixDefinition(TEN**-27, 'ronto'),
    'q': PrefixDefinition(TEN**-30, 'quecto'),
}

PREFIX_DEFINITIONS: dict[str, PrefixDefinition] = {
    prefix[0] if isinstance(prefix, tuple) else prefix: definition
    for prefix, definition in _PREFIX_ENTRIES.items()
}
"""Canonical prefix symbol -> prefix definition."""

PREFIX_NAME_TO_SYMBOL: dict[str, str] = {
    name: prefix
    for prefix, definition in PREFIX_DEFINITIONS.items()
    for name in definition.names
    if name
}
"""Prefix name or name alias -> canonical prefix symbol."""

PREFIX_SYMBOL_ALIAS_TO_SYMBOL: dict[str, str] = {
    alias: prefix[0]
    for prefix in _PREFIX_ENTRIES
    if isinstance(prefix, tuple)
    for alias in prefix[1:]
}
"""Prefix symbol alias -> canonical prefix symbol."""


class UnitDefinition:
    """Unit metadata and spelling policies, distinct from a formal Unit."""

    __slots__ = ('factor', 'names', 'dimension', 'prefixable', 'symbols')

    def __init__(
        self,
        factor: Factor,
        name: str,
        *name_aliases: str,
        dimension: Dimension,
        prefixable: PrefixPolicy = True,
    ) -> None:
        self.factor = factor
        self.names = (name, *name_aliases)
        self.dimension = dimension
        self.prefixable = prefixable
        self.symbols: tuple[str, ...] = ()

    def allows_prefix(self, spelling: str) -> bool:
        return self.prefixable if isinstance(self.prefixable, bool) else spelling in self.prefixable

    @property
    def name(self) -> str:
        return self.names[0]


UNIT_DEFINITIONS: dict[str, UnitDefinition] = {
    '': UnitDefinition(ONE, '', dimension=DIMENSIONLESS, prefixable=False),
    's': UnitDefinition(ONE, 'second', dimension=Dimension(T=1)),
    'm': UnitDefinition(ONE, 'metre', 'meter', dimension=Dimension(L=1)),
    'g': UnitDefinition(TEN**-3, 'gram', dimension=Dimension(M=1)),
    'A': UnitDefinition(ONE, 'ampere', dimension=Dimension(I=1)),
    'K': UnitDefinition(ONE, 'kelvin', dimension=Dimension(Theta=1)),
    'mol': UnitDefinition(ONE, 'mole', dimension=Dimension(N=1)),
    'cd': UnitDefinition(ONE, 'candela', dimension=Dimension(J=1)),
}
"""Canonical unit symbol -> unit definition."""

UNIT_NAME_TO_SYMBOL: dict[str, str] = {
    name: unit for unit, definition in UNIT_DEFINITIONS.items() for name in definition.names if name
}
"""Unit name or name alias -> canonical unit symbol."""

UNIT_SYMBOL_ALIAS_TO_SYMBOL: dict[str, str] = {}
"""Unit symbol alias -> canonical unit symbol."""


for _symbol, _definition in UNIT_DEFINITIONS.items():
    _definition.symbols = (_symbol,)


def _format_prefixed_name(prefix_symbol: str, name: str) -> str:
    return _PREFIXED_NAME_EXCEPTIONS.get(
        (prefix_symbol, name), PREFIX_DEFINITIONS[prefix_symbol].name + name
    )


def _resolve_unit_name(text: str) -> tuple[str, str]:
    """Resolve only names, with at most one prefix and no expression fallback."""
    if text in UNIT_NAME_TO_SYMBOL:
        return '', UNIT_NAME_TO_SYMBOL[text]
    for (prefix_symbol, name), spelling in _PREFIXED_NAME_EXCEPTIONS.items():
        if text == spelling and name in UNIT_NAME_TO_SYMBOL:
            unit_symbol = UNIT_NAME_TO_SYMBOL[name]
            if UNIT_DEFINITIONS[unit_symbol].allows_prefix(name):
                return prefix_symbol, unit_symbol
    for prefix_name, prefix_symbol in PREFIX_NAME_TO_SYMBOL.items():
        if text.startswith(prefix_name):
            name = text[len(prefix_name) :]
            if name in UNIT_NAME_TO_SYMBOL:
                unit_symbol = UNIT_NAME_TO_SYMBOL[name]
                if UNIT_DEFINITIONS[unit_symbol].allows_prefix(name):
                    return prefix_symbol, unit_symbol
    raise UnitSymbolError(f'{text!r} is not a valid unit symbol.')


def _resolve_unit_spelling(text: str, *, warn: bool = True) -> tuple[str, str]:
    """Resolve one spelling; complete symbols precede a single prefix split."""
    if text in UNIT_DEFINITIONS:
        return '', text
    if text in UNIT_SYMBOL_ALIAS_TO_SYMBOL:
        return '', UNIT_SYMBOL_ALIAS_TO_SYMBOL[text]
    for prefix_spelling in (*PREFIX_DEFINITIONS, *PREFIX_SYMBOL_ALIAS_TO_SYMBOL):
        if not prefix_spelling or not text.startswith(prefix_spelling):
            continue
        stem = text[len(prefix_spelling) :]
        unit_symbol = UNIT_SYMBOL_ALIAS_TO_SYMBOL.get(stem, stem)
        prefix_symbol = PREFIX_SYMBOL_ALIAS_TO_SYMBOL.get(prefix_spelling, prefix_spelling)
        if (
            unit_symbol in UNIT_DEFINITIONS
            and stem
            and UNIT_DEFINITIONS[unit_symbol].allows_prefix(stem)
        ):
            if (
                warn
                and prefix_spelling in PREFIX_SYMBOL_ALIAS_TO_SYMBOL
                and prefix_spelling in UNIT_DEFINITIONS
            ):
                preferred, _ = _preferred_unit_spelling(prefix_symbol, unit_symbol)
                name, _ = _preferred_unit_spelling(prefix_symbol, unit_symbol, prefer_name=True)
                warnings.warn(
                    f'{text!r} is interpreted as {preferred!r} ({name}). '
                    f'Use {preferred!r} for {name}, or {prefix_spelling + " " + stem!r} '
                    f'for {UNIT_DEFINITIONS[prefix_spelling].name} '
                    f'{UNIT_DEFINITIONS[unit_symbol].name}.',
                    UnitSymbolWarning,
                    skip_file_prefixes=_WARNING_SKIP_PREFIXES,
                )
            return prefix_symbol, unit_symbol
    return _resolve_unit_name(text)


def _preferred_unit_spelling(
    prefix_symbol: str, unit_symbol: str, *, prefer_name: bool = False
) -> tuple[str, bool]:
    """Choose a permitted spelling; the bool identifies a name fallback."""
    definition = UNIT_DEFINITIONS[unit_symbol]
    for is_name in (True, False) if prefer_name else (False, True):
        for text in definition.names if is_name else definition.symbols:
            if text and (not prefix_symbol or definition.allows_prefix(text)):
                spelling = (
                    _format_prefixed_name(prefix_symbol, text) if is_name else prefix_symbol + text
                )
                return spelling, is_name
    # The identity has an empty name and symbol.
    return '', False


def _register_unit(symbol: str, definition: UnitDefinition, /, *symbol_aliases: str) -> None:
    """Validate every binding and generated spelling before updating indexes."""
    if not isinstance(definition.prefixable, (bool, tuple)):
        raise TypeError('Prefixable must be bool or tuple of spellings.')
    for kind, texts in (
        ('symbol', (symbol,)),
        ('name', (definition.name,)),
        ('symbol alias', symbol_aliases),
        ('name alias', definition.names[1:]),
    ):
        for text in texts:
            if not isinstance(text, str):
                raise TypeError(f'Unit {kind} must be str, got {type(text).__name__}.')
            if text == '' and kind.endswith('alias'):
                continue
            is_name = kind.startswith('name')
            pattern = _UNIT_NAME_RE if is_name else _UNIT_SYMBOL_RE
            if not text.isprintable() or pattern.fullmatch(text) is None:
                rule = (
                    'consist of printable unit tokens separated by single spaces or hyphens'
                    if is_name
                    else 'be a single printable unit token'
                )
                raise ValueError(f'Unit {kind} must {rule}, got {text!r}.')

    symbols = dict.fromkeys(text for text in (symbol, *symbol_aliases) if text)
    names = dict.fromkeys(name for name in definition.names if name)
    if isinstance(definition.prefixable, tuple):
        for text in definition.prefixable:
            if not isinstance(text, str):
                raise TypeError('Prefixable spellings must be str.')
            if text not in symbols and text not in names:
                raise ValueError(f'Prefixable spelling {text!r} is not a registered spelling.')
    definition.symbols = tuple(symbols)
    spellings = [(text, '') for text in (*symbols, *names)]
    for prefix in PREFIX_DEFINITIONS:
        if not prefix:
            continue
        for text in symbols:
            if definition.allows_prefix(text):
                spellings.append((prefix + text, prefix))
                spellings.extend(
                    (alias + text, prefix)
                    for alias, canonical in PREFIX_SYMBOL_ALIAS_TO_SYMBOL.items()
                    if canonical == prefix
                )
        for text in names:
            if definition.allows_prefix(text):
                spellings.extend(
                    (prefix_name + text, prefix) for prefix_name in PREFIX_DEFINITIONS[prefix].names
                )
                if exception := _PREFIXED_NAME_EXCEPTIONS.get((prefix, text)):
                    spellings.append((exception, prefix))

    spelling_prefixes: dict[str, str] = {}
    for text, prefix in spellings:
        if spelling_prefixes.setdefault(text, prefix) != prefix:
            raise UnitRegistryError(f'Unit text {text!r} has conflicting prefix meanings.')
        try:
            _resolve_unit_spelling(text, warn=False)
        except UnitSymbolError:
            pass
        else:
            raise UnitRegistryError(f'Unit text {text!r} already resolves to a registered unit.')
        if ' ' in text:
            try:
                for word in text.split(' '):
                    _resolve_unit_spelling(word, warn=False)
            except UnitSymbolError:
                continue
            raise UnitRegistryError(f'Unit text {text!r} already parses as a unit product.')

    UNIT_DEFINITIONS[symbol] = definition
    UNIT_NAME_TO_SYMBOL.update(dict.fromkeys(names, symbol))
    UNIT_SYMBOL_ALIAS_TO_SYMBOL.update(
        dict.fromkeys((alias for alias in symbols if alias != symbol), symbol)
    )
