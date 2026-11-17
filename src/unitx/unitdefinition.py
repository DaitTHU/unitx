"""Immutable unit metadata; definitions retain independent formal identities."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from uuid import UUID, uuid4

from ._formatting.unicode import SUPERSCRIPTS
from ._immutable import _Immutable
from ._registry.prefixes import (
    _PREFIXED_NAME_EXCEPTIONS,
    ALL_PREFIXES,
    BINARY_PREFIXES,
    DECIMAL_PREFIXES,
    PREFIX_DEFINITIONS,
    PREFIX_SYMBOL_ALIAS_TO_SYMBOL,
    PrefixDefinition,
)
from .dimension import DIMENSIONLESS, Dimension
from .exceptions import UnitRegistryError
from .factor import Factor, one

__all__ = ['PrefixFamily', 'PrefixSelection', 'PrefixPolicy', 'UnitDefinition']

_UNIT_SYMBOL_PATTERN = r'[^\s\d' + re.escape('*^()/+-[]' + SUPERSCRIPTS) + ']+'
_UNIT_SYMBOL_RE = re.compile(_UNIT_SYMBOL_PATTERN)
_UNIT_NAME_RE = re.compile(f'{_UNIT_SYMBOL_PATTERN}(?:[ -]{_UNIT_SYMBOL_PATTERN})*')


class PrefixFamily(Enum):
    """Named prefix sets, expanded to the same symbols as explicit selections."""

    DECIMAL = 'decimal'
    BINARY = 'binary'
    ALL = 'all'


type PrefixSelection = bool | PrefixFamily | str | tuple[str, ...] | set[str] | frozenset[str]
type PrefixPolicy = PrefixSelection | Mapping[str, PrefixSelection]
type _StoredPrefixPolicy = frozenset[str] | Mapping[str, frozenset[str]]


def _prefix_symbols(selection: PrefixSelection) -> frozenset[str]:
    """Normalize named sets and explicit symbols, including prefix aliases."""
    if isinstance(selection, (bool, PrefixFamily)):
        prefixes = (
            ()
            if selection is False
            else BINARY_PREFIXES
            if selection is PrefixFamily.BINARY
            else ALL_PREFIXES
            if selection is PrefixFamily.ALL
            else DECIMAL_PREFIXES
        )
        return frozenset(prefix.symbol for prefix in prefixes)
    if isinstance(selection, str):
        # Longest symbol first: da and Ki are single prefixes; kMG is three.
        tokens: list[str] = []
        offset = 0
        while offset < len(selection):
            token = selection[offset : offset + 2]
            if token not in PREFIX_DEFINITIONS:
                token = selection[offset]
            tokens.append(token)
            offset += len(token)
        selection = tuple(tokens)
    if not isinstance(selection, (tuple, set, frozenset)):
        error_message = 'Prefix selection must be bool, PrefixFamily, str, tuple, set or frozenset.'
        raise TypeError(error_message)
    symbols: set[str] = set()
    for symbol in selection:
        if not isinstance(symbol, str):
            error_message = f'Prefix symbols must be str, got {type(symbol).__name__}.'
            raise TypeError(error_message)
        canonical = PREFIX_SYMBOL_ALIAS_TO_SYMBOL.get(symbol, symbol)
        if canonical not in PREFIX_DEFINITIONS:
            error_message = f'Unknown prefix symbol {symbol!r}.'
            raise ValueError(error_message)
        symbols.add(canonical)
    return frozenset(symbols)


type UnitTerms = tuple[tuple[PrefixDefinition, UnitDefinition, Fraction], ...]


@dataclass(frozen=True, slots=True)
class _ConversionData:
    """Exact physical conversion data, independent of unit identity and spelling."""

    factor: Factor = one
    dimension: Dimension = DIMENSIONLESS

    def __post_init__(self) -> None:
        for value, expected, label in (
            (self.factor, Factor, 'factor'),
            (self.dimension, Dimension, 'dimension'),
        ):
            if not isinstance(value, expected):
                error_message = f'Unit definition {label} must be a {expected.__name__}.'
                raise TypeError(error_message)


@dataclass(frozen=True, slots=True)
class _UnitSpellings:
    """Validated names and prefix permissions, with derived accepted spellings."""

    symbols: tuple[str, ...]
    names: tuple[str, ...]
    prefixable: _StoredPrefixPolicy
    _identity: InitVar[bool] = False
    accepted: tuple[str, ...] = field(init=False)

    def __post_init__(self, _identity: bool) -> None:
        if isinstance(self.prefixable, Mapping):
            object.__setattr__(self, 'prefixable', MappingProxyType(dict(self.prefixable)))
        object.__setattr__(self, 'accepted', () if _identity else _validate_spellings(self))

    @property
    def symbol(self) -> str:
        return self.symbols[0]

    @property
    def name(self) -> str:
        return self.names[0]

    def _prefixes(self) -> tuple[PrefixDefinition, ...]:
        """Enumerate permitted symbols in the standard prefix order."""
        policy = self.prefixable
        symbols = policy if isinstance(policy, frozenset) else frozenset().union(*policy.values())
        return tuple(prefix for prefix in ALL_PREFIXES if prefix.symbol in symbols)

    def allows_prefix(self, spelling: str, prefix: PrefixDefinition | None = None) -> bool:
        if spelling not in self.symbols and spelling not in self.names:
            return False
        policy = self.prefixable
        symbols = policy if isinstance(policy, frozenset) else policy.get(spelling, frozenset())
        return bool(symbols) if prefix is None else prefix.symbol in symbols


@dataclass(frozen=True, slots=True)
class _UnitProduct:
    """Complete bound unit expression, including its extra exact scale."""

    scale: Factor
    terms: UnitTerms


class UnitDefinition(_Immutable):
    """One immutable unit identity with spellings and linear conversion data.

    The definition object itself is the formal identity. Equal dimensions and
    factors do not merge independently defined units. Ordinary definitions need
    only a symbol; the name defaults to it, the factor to one, and the dimension
    to dimensionless. Derived units and coordinates copy their conversion data
    through internal constructors rather than exposing unrelated options here.
    """

    __slots__ = (
        '__weakref__',
        '_key',
        '_spelling',
        '_conversion',
    )

    _key: UUID
    _spelling: _UnitSpellings
    _conversion: _ConversionData

    def __init__(
        self,
        symbol: str,
        name: str | None = None,
        *name_aliases: str,
        factor: Factor = one,
        dimension: Dimension = DIMENSIONLESS,
        symbol_aliases: str | tuple[str, ...] = (),
        prefixable: PrefixPolicy = True,
    ) -> None:
        self._initialize(
            symbol,
            name,
            name_aliases,
            conversion=_ConversionData(factor, dimension),
            symbol_aliases=symbol_aliases,
            prefixable=prefixable,
        )

    def _initialize(
        self,
        symbol: str,
        name: str | None,
        name_aliases: tuple[str, ...],
        *,
        conversion: _ConversionData,
        _identity: bool = False,
        symbol_aliases: str | tuple[str, ...] = (),
        prefixable: PrefixPolicy = True,
    ) -> None:
        if hasattr(self, '_key'):
            error_message = 'UnitDefinition objects are immutable.'
            raise AttributeError(error_message)
        if not isinstance(symbol_aliases, (str, tuple)):
            error_message = (
                f'Symbol aliases must be str or tuple of str, got {type(symbol_aliases).__name__}.'
            )
            raise TypeError(error_message)
        if isinstance(symbol_aliases, str):
            symbol_aliases = (symbol_aliases,)
        spelling = _UnitSpellings(
            (symbol, *symbol_aliases),
            (symbol if name is None else name, *name_aliases),
            {text: _prefix_symbols(rule) for text, rule in prefixable.items()}
            if isinstance(prefixable, Mapping)
            else _prefix_symbols(prefixable),
            _identity=_identity,
        )
        object.__setattr__(self, '_spelling', spelling)
        object.__setattr__(self, '_conversion', conversion)
        object.__setattr__(self, '_key', uuid4())
        from ._registry.reconstruction import _remember_definition

        _remember_definition(self)

    @classmethod
    def _derived(
        cls,
        symbol: str,
        name: str | None,
        *name_aliases: str,
        conversion: _ConversionData,
        symbol_aliases: str | tuple[str, ...] = (),
        prefixable: PrefixPolicy = True,
    ) -> UnitDefinition:
        definition = object.__new__(cls)
        definition._initialize(
            symbol,
            name,
            name_aliases,
            conversion=conversion,
            symbol_aliases=symbol_aliases,
            prefixable=prefixable,
        )
        return definition

    @classmethod
    def _identity(cls) -> UnitDefinition:
        definition = object.__new__(cls)
        definition._initialize(
            '',
            '',
            (),
            conversion=_ConversionData(),
            prefixable=False,
            _identity=True,
        )
        return definition

    def _dependencies(self) -> tuple[UnitDefinition, ...]:
        return ()

    def __copy__(self) -> UnitDefinition:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> UnitDefinition:
        return self

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        from ._registry.reconstruction import (
            _definition_location,
            _definition_state,
            _restore_definition,
        )

        return _restore_definition, (
            self._key.int,
            _definition_location(self),
            _definition_state(self),
        )

    @property
    def symbol(self) -> str:
        return self._spelling.symbol

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._spelling.symbols

    @property
    def names(self) -> tuple[str, ...]:
        return self._spelling.names

    @property
    def prefixable(self) -> _StoredPrefixPolicy:
        return self._spelling.prefixable

    @property
    def factor(self) -> Factor:
        return self._conversion.factor

    @property
    def dimension(self) -> Dimension:
        return self._conversion.dimension

    @property
    def _spellings(self) -> tuple[str, ...]:
        return self._spelling.accepted

    def allows_prefix(self, spelling: str, prefix: PrefixDefinition | None = None) -> bool:
        """Check a spelling and, when supplied, an individual prefix."""
        return self._spelling.allows_prefix(spelling, prefix)

    def _prefixes(self) -> tuple[PrefixDefinition, ...]:
        return self._spelling._prefixes()

    @property
    def name(self) -> str:
        return self._spelling.name


def _validate_spellings(definition: _UnitSpellings) -> tuple[str, ...]:
    """Validate a definition's own spellings without consulting an environment."""
    symbol, *symbol_aliases = definition.symbols
    policy = definition.prefixable
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
                error_message = f'Unit {kind} must {rule}, got {text!r}.'
                raise ValueError(error_message)

    symbols = dict.fromkeys(text for text in (symbol, *symbol_aliases) if text)
    names = dict.fromkeys(name for name in definition.names if name)
    if isinstance(policy, Mapping):
        for text in policy:
            if not isinstance(text, str):
                error_message = f'Prefixable spellings must be str, got {type(text).__name__}.'
                raise TypeError(error_message)
            if text not in symbols and text not in names:
                error_message = f'Prefixable spelling {text!r} is not a registered spelling.'
                raise ValueError(error_message)
    spellings = [(text, '') for text in (*symbols, *names)]
    for prefix in definition._prefixes():
        prefix_symbol = prefix.symbol
        for text in symbols:
            if not definition.allows_prefix(text, prefix):
                continue
            spellings.append((prefix_symbol + text, prefix_symbol))
            spellings.extend((alias + text, prefix_symbol) for alias in prefix.symbol_aliases)
        for text in names:
            if not definition.allows_prefix(text, prefix):
                continue
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
    return tuple(spelling_prefixes)
