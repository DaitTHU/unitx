"""Immutable unit metadata; definitions retain independent formal identities."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from typing import Literal
from uuid import UUID, uuid4

from ._formatting.unicode import SUPERSCRIPTS
from ._immutable import _Immutable
from ._registry.prefixes import (
    _PREFIXED_NAME_EXCEPTIONS,
    ALL_PREFIXES,
    BINARY_PREFIXES,
    DECIMAL_PREFIXES,
    PrefixDefinition,
)
from .dimension import DIMENSIONLESS, Dimension
from .exceptions import UnitRegistryError
from .factor import Factor, one

__all__ = ['PrefixFamily', 'PrefixPolicy', 'UnitDefinition']

_UNIT_SYMBOL_PATTERN = r'[^\s\d' + re.escape('*^()/+-[]' + SUPERSCRIPTS) + ']+'
_UNIT_SYMBOL_RE = re.compile(_UNIT_SYMBOL_PATTERN)
_UNIT_NAME_RE = re.compile(f'{_UNIT_SYMBOL_PATTERN}(?:[ -]{_UNIT_SYMBOL_PATTERN})*')


class PrefixFamily(Enum):
    """Allowed prefix families, distinct from unit spelling strings."""

    DECIMAL = 'decimal'
    BINARY = 'binary'
    ALL = 'all'


type PrefixPolicy = (
    bool | PrefixFamily | str | tuple[str, ...] | Mapping[str, PrefixFamily | Literal[True]]
)
type _StoredPrefixPolicy = (
    bool | PrefixFamily | tuple[str, ...] | Mapping[str, PrefixFamily | Literal[True]]
)
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
        """Select candidate families once, before iterating individual prefixes."""
        policy = self.prefixable
        if policy is True:
            return DECIMAL_PREFIXES
        if policy is False:
            return ()
        if isinstance(policy, tuple):
            return DECIMAL_PREFIXES if policy else ()
        if isinstance(policy, PrefixFamily):
            if policy is PrefixFamily.DECIMAL:
                return DECIMAL_PREFIXES
            if policy is PrefixFamily.BINARY:
                return BINARY_PREFIXES
            return ALL_PREFIXES
        rules = policy.values()
        decimal = any(r is True or r in (PrefixFamily.DECIMAL, PrefixFamily.ALL) for r in rules)
        binary = any(r in (PrefixFamily.BINARY, PrefixFamily.ALL) for r in rules)
        if decimal and binary:
            return ALL_PREFIXES
        if decimal:
            return DECIMAL_PREFIXES
        if binary:
            return BINARY_PREFIXES
        return ()

    def allows_prefix(self, spelling: str, prefix: PrefixDefinition | None = None) -> bool:
        policy = self.prefixable
        if policy is True:
            return prefix is None or prefix.family == 'decimal'
        if policy is False:
            return False
        if isinstance(policy, tuple):
            return (prefix is None or prefix.family == 'decimal') and spelling in policy
        rule = policy if isinstance(policy, PrefixFamily) else policy.get(spelling, False)
        if rule is True:
            return prefix is None or prefix.family == 'decimal'
        return rule is not False and (
            prefix is None or rule is PrefixFamily.ALL or prefix.family == rule.value
        )


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
            (prefixable,) if isinstance(prefixable, str) else prefixable,
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
        """Check a spelling and, when supplied, the prefix's family."""
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
    if not isinstance(policy, (bool, PrefixFamily, tuple, Mapping)):
        wrong_type = type(policy).__name__
        error_message = (
            'Prefixable must be bool, str or tuple of spellings, a PrefixFamily, '
            f'or a spelling mapping, got {wrong_type}.'
        )
        raise TypeError(error_message)
    if isinstance(policy, Mapping):
        for rule in policy.values():
            if rule is not True and not isinstance(rule, PrefixFamily):
                error_message = 'Prefix mapping values must be True or a PrefixFamily.'
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
                error_message = f'Unit {kind} must {rule}, got {text!r}.'
                raise ValueError(error_message)

    symbols = dict.fromkeys(text for text in (symbol, *symbol_aliases) if text)
    names = dict.fromkeys(name for name in definition.names if name)
    if isinstance(policy, (tuple, Mapping)):
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
