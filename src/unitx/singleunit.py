"""Bound prefix/definition identities and their permitted canonical spellings.

Public construction resolves one spelling in the default registry. Compound
expressions belong to UnitParser; physical conversion belongs to quantities.
"""

from __future__ import annotations

from ._immutable import _Immutable
from ._registry.prefixes import NO_PREFIX, PrefixDefinition, _format_prefixed_name
from ._registry.reconstruction import _definition_by_id
from .dimension import Dimension
from .exceptions import UnitSymbolError, UnitSyntaxError
from .factor import Factor
from .unitdefinition import UnitDefinition
from .unitregistry import default_registry

__all__ = ['SingleUnit']


class SingleUnit(_Immutable):
    """An immutable unit identified by its canonical prefix and shared definition object.

    Hold shared immutable prefix and unit definitions resolved at construction.

    Accept a registered symbol or singular English name, optionally with one
    permitted decimal or binary prefix: `'m'`, `'metre'`, `'km'`, and
    `'kilometre'`. Binary prefixes require an explicit definition permission. Inputs are
    case-sensitive. Micro prefix aliases `'u'` and `'μ'` normalize to
    `'µ'`; kilo uses lowercase `'k'`, while `'K'` denotes kelvin.
    The empty string represents a dimensionless unit.

    Registered names may contain words separated by single ASCII spaces,
    as in `astronomical unit`, or hyphens as in `pascal-second`.
    Prefix names attach directly to the first word, as in
    `kiloastronomical unit`. Names are matched exactly.
    Prefix permissions are checked on the spelling used, before alias
    normalization. `symbol` prefers a permitted symbol, then a name;
    `name` prefers a permitted name, then a symbol. This keeps reconstruction
    valid even when only names or only symbol aliases accept a given prefix.

    The default table includes SI and common scientific units. Extend it
    explicitly with `default_registry.define()` or `include()`. Compound expressions and
    powers are not accepted. Equality and hashing compare
    canonical unit identity, so aliases compare equal but m and mm do not.

    `factor` scales a magnitude to the corresponding coherent SI unit.
    Mass is referenced to kg: g has factor 1/1000 and kg has factor 1.
    `without_prefix()` returns a unit without its prefix; it does not convert
    a magnitude. `name` returns the primary name with its prefix, or an empty
    string for the multiplicative identity.

    >>> unit = SingleUnit('kilogram')
    >>> unit.symbol, unit.name, unit.factor.value
    ('kg', 'kilogram', 1)
    >>> unit.prefix_symbol, unit.unprefixed_symbol
    ('k', 'g')
    >>> unit.without_prefix()
    SingleUnit('g')
    >>> unit.prefix_factor.value, unit.unprefixed_factor.value
    (1000, Fraction(1, 1000))
    >>> SingleUnit('um') == SingleUnit('micrometre')
    True
    """

    __slots__ = ('_prefix', '_unit')
    _prefix: PrefixDefinition
    _unit: UnitDefinition

    def __new__(cls, text: str, /) -> SingleUnit:
        if not isinstance(text, str):
            error_message = f'Unit symbol or name must be str, got {type(text).__name__}.'
            raise TypeError(error_message)
        prefix, unit = default_registry._resolve(text)
        return cls._from_definitions(prefix, unit)

    @classmethod
    def from_name(cls, name: str, /) -> SingleUnit:
        """Construct from a registered name or name alias, optionally prefixed.

        Match names exactly, with the same spelling-specific prefix permissions
        as the ordinary constructor. Symbols, expressions, and the empty string
        raise UnitSymbolError; non-string inputs raise TypeError.

        >>> SingleUnit.from_name('kilometre') == SingleUnit('km')
        True
        >>> SingleUnit.from_name('meter') == SingleUnit('m')
        True
        """
        if not isinstance(name, str):
            error_message = f'Unit name must be str, got {type(name).__name__}.'
            raise TypeError(error_message)
        prefix, unit = default_registry._resolve_name(name)
        return cls._from_definitions(prefix, unit)

    @classmethod
    def _from_definitions(cls, prefix: PrefixDefinition, unit: UnitDefinition) -> SingleUnit:
        """Construct from shared immutable definitions. Internal use only."""
        obj = super().__new__(cls)
        object.__setattr__(obj, '_prefix', prefix)
        object.__setattr__(obj, '_unit', unit)
        return obj

    def __repr__(self) -> str:
        try:
            resolved = SingleUnit(self.symbol)
        except (UnitSymbolError, UnitSyntaxError):
            resolved = None
        if resolved == self:
            return f'{type(self).__name__}({self.symbol!r})'
        return f'SingleUnit._from_identity({self._unit._key.int!r}, {self.prefix_symbol!r})'

    def __str__(self) -> str:
        return self.symbol

    def __hash__(self) -> int:
        return hash((self.prefix_symbol, self._unit))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SingleUnit):
            return NotImplemented
        return self.prefix_symbol == other.prefix_symbol and self._unit is other._unit

    def __copy__(self) -> SingleUnit:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> SingleUnit:
        return self

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return _restore_single_unit, (self._unit, self.prefix_symbol)

    @classmethod
    def _from_identity(cls, identity: int, prefix: str) -> SingleUnit:
        from ._registry.prefixes import PREFIX_DEFINITIONS

        return cls._from_definitions(
            PREFIX_DEFINITIONS[prefix] if prefix else NO_PREFIX,
            _definition_by_id(identity),
        )

    @property
    def prefix_symbol(self) -> str:
        return self._prefix.symbol

    @property
    def unprefixed_symbol(self) -> str:
        return self._unit.symbol

    @property
    def symbol(self) -> str:
        spelling = self._get_symbol()
        if spelling is None:
            spelling = self._get_name()
        assert spelling is not None
        return spelling

    @property
    def prefix_name(self) -> str:
        return self._prefix.name

    @property
    def unprefixed_name(self) -> str:
        return self._unit.name

    @property
    def name(self) -> str:
        spelling = self._get_name()
        if spelling is None:
            spelling = self._get_symbol()
        assert spelling is not None
        return spelling

    def _expression(self) -> str:
        symbol = self._get_symbol()
        if symbol is not None:
            return symbol
        name = self._get_name()
        assert name is not None
        return f'[{name}]'

    def _get_symbol(self) -> str | None:
        """Return a permitted symbol, or None if this prefix requires a name."""
        if not self.has_prefix():
            return self._unit.symbol
        for symbol in self._unit.symbols:
            if not self._unit.allows_prefix(symbol, self._prefix):
                continue
            return self._prefix.symbol + symbol
        return None

    def _get_name(self) -> str | None:
        """Return a permitted name, or None if this prefix requires a symbol."""
        if not self.has_prefix():
            return self._unit.name
        for name in self._unit.names:
            if not self._unit.allows_prefix(name, self._prefix):
                continue
            return _format_prefixed_name(self._prefix, name)
        return None

    @property
    def prefix_factor(self) -> Factor:
        return self._prefix.factor

    @property
    def unprefixed_factor(self) -> Factor:
        """Scale from the unprefixed unit to its coherent SI unit."""
        return self._unit.factor

    @property
    def factor(self) -> Factor:
        """Scale from this unit to its coherent SI unit, including the prefix."""
        return self.prefix_factor * self.unprefixed_factor

    @property
    def dimension(self) -> Dimension:
        return self._unit.dimension

    def without_prefix(self) -> SingleUnit:
        """Return the unit without its prefix; no magnitude is converted."""
        return SingleUnit._from_definitions(NO_PREFIX, self._unit) if self.has_prefix() else self

    def has_prefix(self) -> bool:
        return self.prefix_symbol != ''


def _restore_single_unit(definition: UnitDefinition, prefix: str) -> SingleUnit:
    from ._registry.prefixes import PREFIX_DEFINITIONS

    return SingleUnit._from_definitions(
        PREFIX_DEFINITIONS[prefix] if prefix else NO_PREFIX,
        definition,
    )
