"""Named difference definitions retaining one immutable coordinate mapping."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._log_mapping import _LogMapping
from .unitdefinition import UnitDefinition

if TYPE_CHECKING:
    from ._affine_mapping import _AffineMapping


class _CoordinateDefinition(UnitDefinition):
    """Extend a linear identity with a shared, already validated mapping."""

    __slots__ = ('_mapping',)
    _mapping: _AffineMapping | _LogMapping

    def __init__(
        self,
        mapping: _AffineMapping | _LogMapping,
        symbol: str,
        name: str | None = None,
        *name_aliases: str,
        symbol_aliases: str | tuple[str, ...] = (),
    ) -> None:
        self._initialize(
            symbol,
            name,
            name_aliases,
            conversion=mapping.delta_unit._conversion,
            symbol_aliases=symbol_aliases,
            prefixable=False,
        )
        object.__setattr__(self, '_mapping', mapping)

    def _dependencies(self) -> tuple[UnitDefinition, ...]:
        mapping = self._mapping
        reference = mapping.reference if isinstance(mapping, _LogMapping) else mapping.origin
        units = reference.unit, mapping.delta_unit
        return tuple(base._unit for unit in units for base, _ in unit._monomial.components())
