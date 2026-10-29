"""Read-only unit catalogues collected after ordinary class-body execution."""

from collections.abc import Iterable, Iterator
from typing import ClassVar

from ._constnamespace import ConstNamespace, _ConstNamespaceMeta
from ._coordinates.resolve import _Coordinate
from ._registry.prefixes import NO_PREFIX, PREFIX_DEFINITIONS, PREFIX_SYMBOL_ALIAS_TO_SYMBOL
from ._registry.reconstruction import _publish_definitions
from .monomial import Monomial
from .singleunit import SingleUnit
from .unit import Unit
from .unitregistry import Registry, _collect_definitions

__all__ = ['UnitCatalog']


def _prefixed_units(unit: Unit, prefixes: str | tuple[str, ...]) -> Iterator[Unit]:
    """Replace a single unit's prefix with each listed prefix.

    Use `prefixes` as a string when every prefix is an individual character.
    Unit without prefix can be included in the string as a space character.
    Otherwise use a tuple of strings for multi-character prefixes.
    """
    [(base, exponent)] = unit._monomial.components()
    assert exponent == 1
    for spelling in prefixes:
        prefix = (
            PREFIX_DEFINITIONS[PREFIX_SYMBOL_ALIAS_TO_SYMBOL.get(spelling, spelling)]
            if spelling != ' '
            else NO_PREFIX
        )
        yield Unit._from_monomial(Monomial({SingleUnit._from_definitions(prefix, base._unit): 1}))


class _UnitCatalogMeta(_ConstNamespaceMeta):
    def __new__(
        metacls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, object],
        *,
        units: Iterable[Unit | _Coordinate] = (),
        registry: Registry | None = None,
    ) -> type:
        target = Registry() if registry is None else registry
        namespace['_registry'] = target
        catalogue = super().__new__(metacls, name, bases, namespace)
        exports = tuple(
            value
            for attr, value in namespace.items()
            if not attr.startswith('_') and isinstance(value, Unit | _Coordinate)
        )
        definitions = _collect_definitions((*units, *exports))
        bound = target._bind(definitions)
        _publish_definitions(catalogue.__module__, catalogue.__qualname__, bound)
        return catalogue


class UnitCatalog(ConstNamespace, metaclass=_UnitCatalogMeta):
    """An immutable attribute namespace with its own parsing environment.

    After the class body finishes, collect its public Unit and named coordinate
    attributes and optional `units=` into a fresh Registry, or the supplied
    `registry=`. Include canonical spellings and aliases; attribute names do not
    become parser aliases. Inherited attributes do not populate the parser.
    Namespace immutability does not freeze the Registry. Later registrations in
    a supplied Registry remain visible through `unit()`; its synchronization
    requirements apply to catalogue parsing as well.

    Use unit objects to build later definitions in the class body. New spellings
    become parseable only after collection; bare strings always use the default
    environment. `Registry.defining()` remains an ordinary explicit block scope.
    Module-level catalogues supply persistence locations automatically.
    """

    _registry: ClassVar[Registry]

    def __init_subclass__(
        cls,
        *,
        units: Iterable[Unit | _Coordinate] = (),
        registry: Registry | None = None,
    ) -> None:
        super().__init_subclass__()

    @classmethod
    def unit(cls, expression: str, /) -> Unit:
        """Resolve an expression in this catalogue and return a bound Unit."""
        return cls._registry.parse(expression)
