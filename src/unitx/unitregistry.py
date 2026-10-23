"""Explicit spelling environments and the default SI parser bindings.

Definitions are already valid before binding. Registry owns spelling lookup and
checks conflicts against the receiving environment. Registration during concurrent
reads or writes requires external synchronization.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._registry.builtin_units import _UNIT_ENTRIES
from ._registry.prefixes import (
    _PREFIX_NAME_LENGTHS,
    _PREFIX_SYMBOL_LENGTHS,
    _PREFIXED_NAME_EXPANSIONS,
    NO_PREFIX,
    PREFIX_DEFINITIONS,
    PREFIX_NAME_TO_SYMBOL,
    PREFIX_SYMBOL_ALIAS_TO_SYMBOL,
    PrefixDefinition,
)
from ._registry.reconstruction import _publish_definitions
from .exceptions import UnitRegistryError, UnitSymbolError
from .unitdefinition import PrefixPolicy, UnitDefinition

if TYPE_CHECKING:
    from ._coordinates.resolve import _Coordinate
    from ._numerics import Fractionable
    from .quantity import Quantity
    from .unit import Unit

__all__ = ['Registry', 'default_registry']


_DEFINITION_SCOPE: ContextVar[Registry | None] = ContextVar('unitx_definition_scope', default=None)


def _definition_target() -> Registry | None:
    return _DEFINITION_SCOPE.get()


def _collect_definitions(units: Iterable[Unit | _Coordinate]) -> tuple[UnitDefinition, ...]:
    """Adapt public unit objects to immutable records before entering the binder."""
    from ._coordinates.resolve import _Coordinate
    from .unit import Unit

    entries: list[UnitDefinition] = []
    for unit in units:
        if isinstance(unit, _Coordinate):
            if unit._definition is None:
                error_message = 'Name a coordinate with .define() before including it.'
                raise TypeError(error_message)
            entries.append(unit._definition)
        elif isinstance(unit, Unit):
            entries.extend(base._unit for base, _ in unit._monomial.components())
        else:
            error_message = 'Registry.include requires Unit or named AffineUnit or LogUnit objects.'
            raise TypeError(error_message)
    return tuple(entries)


@dataclass(slots=True)
class _Bindings:
    """Own spelling indexes and their lookup, conflict and staging rules."""

    definitions: dict[str, UnitDefinition]
    names: dict[str, str]
    aliases: dict[str, str]

    def copy(self) -> _Bindings:
        return _Bindings(self.definitions.copy(), self.names.copy(), self.aliases.copy())

    def update(self, candidate: _Bindings) -> None:
        self.definitions.update(candidate.definitions)
        self.names.update(candidate.names)
        self.aliases.update(candidate.aliases)

    def resolve_name(self, text: str, /) -> tuple[PrefixDefinition, UnitDefinition]:
        """Resolve only names to shared definitions, with at most one prefix."""
        definitions, names = self.definitions, self.names
        unit_symbol = names.get(text)
        if unit_symbol is not None:
            return NO_PREFIX, definitions[unit_symbol]
        prefixed_name = _PREFIXED_NAME_EXPANSIONS.get(text, text)
        for prefix_len in _PREFIX_NAME_LENGTHS:
            prefix_symbol = PREFIX_NAME_TO_SYMBOL.get(prefixed_name[:prefix_len])
            if prefix_symbol is None:
                continue
            name = prefixed_name[prefix_len:]
            unit_symbol = names.get(name)
            if unit_symbol is not None:
                unit = definitions[unit_symbol]
                if unit.allows_prefix(name):
                    return PREFIX_DEFINITIONS[prefix_symbol], unit
        error_message = f'{text!r} is not a valid unit symbol.'
        raise UnitSymbolError(error_message)

    def resolve(self, text: str, /) -> tuple[PrefixDefinition, UnitDefinition]:
        """Resolve shared definitions; complete symbols precede one prefix split."""
        definitions, aliases = self.definitions, self.aliases
        unitdef = definitions.get(text)
        if unitdef is not None:
            return NO_PREFIX, unitdef
        unit_symbol = aliases.get(text)
        if unit_symbol is not None:
            return NO_PREFIX, definitions[unit_symbol]
        for prefix_len in _PREFIX_SYMBOL_LENGTHS:
            prefix = text[:prefix_len]
            prefixdef = PREFIX_DEFINITIONS.get(PREFIX_SYMBOL_ALIAS_TO_SYMBOL.get(prefix, prefix))
            if prefixdef is None:
                continue
            unit = text[prefix_len:]
            unitdef = definitions.get(aliases.get(unit, unit))
            if unitdef is not None and unitdef.allows_prefix(unit):
                return prefixdef, unitdef
        return self.resolve_name(text)

    def register(self, definition: UnitDefinition, /) -> None:
        """Validate all spellings before updating shared indexes, without locking."""
        definitions, names_index, aliases_index = self.definitions, self.names, self.aliases
        symbol = definition.symbol
        symbols = dict.fromkeys(text for text in definition.symbols if text)
        names = dict.fromkeys(text for text in definition.names if text)
        for text in definition._spellings:
            try:
                self.resolve(text)
            except UnitSymbolError:
                pass
            else:
                error_message = f'Unit text {text!r} already resolves to a registered unit.'
                raise UnitRegistryError(error_message)
            if ' ' in text:
                try:
                    for word in text.split(' '):
                        self.resolve(word)
                except UnitSymbolError:
                    continue
                error_message = f'Unit text {text!r} already parses as a unit product.'
                raise UnitRegistryError(error_message)

        definitions[symbol] = definition
        names_index.update(dict.fromkeys(names, symbol))
        aliases_index.update(dict.fromkeys((alias for alias in symbols if alias != symbol), symbol))


class Registry:
    """An explicit spelling environment, independent of the default environment.

    Start with only the multiplicative identity. Include existing Unit objects
    to share their definitions, or define private units. No fallback to the
    default table, inheritance, or implicit activation occurs. Initialize before
    concurrent use; registration and parsing during registration need external
    synchronization.

    `prefixable` supplies the default for definitions made by this registry
    or within its defining() blocks. Included units retain their own policy.
    """

    __slots__ = ('_bindings', '_prefixable')

    def __init__(
        self, units: Iterable[Unit | _Coordinate] = (), *, prefixable: bool = True
    ) -> None:
        if type(prefixable) is not bool:
            error_message = 'Registry prefixable default must be bool.'
            raise TypeError(error_message)
        self._prefixable = prefixable
        self._bindings = _Bindings({'': _UNIT_ENTRIES[0]}, {}, {})
        units = tuple(units)
        if units:
            self.include(*units)

    @contextmanager
    def defining(self) -> Iterator[Registry]:
        """Bind `.define()` calls in this block to this registry.

        Unit, Quantity and coordinate definitions, including helper calls, use
        the selected target. `as` returns this Registry. Bare string parsing
        still uses the default environment. An inner block supplies its own scope;
        exit restores the enclosing selection even when an exception is raised.
        Completed registrations are not rolled back.
        ContextVar isolation follows asyncio task contexts; it does not
        synchronize writes to a shared Registry.
        """
        token = _DEFINITION_SCOPE.set(self)
        try:
            yield self
        finally:
            _DEFINITION_SCOPE.reset(token)

    def include(self, *units: Unit | _Coordinate) -> None:
        """Bind an entire batch of existing units without creating new identities.

        Include coordinate dependencies as well. Validate the whole batch before
        updating spelling indexes; a conflict leaves this Registry unchanged.
        """
        self._bind(_collect_definitions(units))

    def _bind(self, definitions: Iterable[UnitDefinition]) -> tuple[UnitDefinition, ...]:
        """Bind validated records and their dependencies; return the explicit closure."""
        candidate = self._bindings.copy()
        bound: dict[UnitDefinition, None] = {}

        def include(definition: UnitDefinition) -> None:
            if definition in bound:
                return
            for dependency in definition._dependencies():
                include(dependency)
            if candidate.definitions.get(definition.symbol) is not definition:
                candidate.register(definition)
            bound[definition] = None

        for definition in definitions:
            include(definition)
        self._bindings.update(candidate)
        return tuple(bound)

    def _resolve(self, text: str, /) -> tuple[PrefixDefinition, UnitDefinition]:
        return self._bindings.resolve(text)

    def _resolve_name(self, text: str, /) -> tuple[PrefixDefinition, UnitDefinition]:
        return self._bindings.resolve_name(text)

    def parse(self, expression: str, /) -> Unit:
        """Parse using only this environment and return a bound Unit."""
        from .unit import Unit
        from .unitparser import UnitParser

        return Unit._from_monomial(UnitParser(expression, registry=self).parse())

    def define(
        self,
        source: Unit | Quantity[Fractionable],
        symbol: str,
        /,
        name: str | None = None,
        *name_aliases: str,
        symbol_aliases: str | tuple[str, ...] = (),
        prefixable: PrefixPolicy | None = None,
    ) -> Unit:
        """Define and bind a linear unit using the ordinary `.define()` contract."""
        from .quantity import Quantity
        from .unit import Unit

        if not isinstance(source, (Unit, Quantity)):
            error_message = 'Registry.define requires a Unit or positive real Quantity.'
            raise TypeError(error_message)
        with self.defining():
            return source.define(
                symbol,
                name,
                *name_aliases,
                symbol_aliases=symbol_aliases,
                prefixable=prefixable,
            )


default_registry = Registry()
"""The environment used by bare unit strings; extra catalogues do not change it."""
default_registry._bind(_UNIT_ENTRIES)
_publish_definitions(__name__, '', default_registry._bindings.definitions.values())
