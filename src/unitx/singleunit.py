from __future__ import annotations

from fractions import Fraction

from ._immutable import _Immutable
from .dimension import Dimension
from .factor import Factor
from .unitregistry import (
    NO_PREFIX,
    PrefixDefinition,
    UnitDefinition,
    _format_prefixed_name,
    _resolve_unit,
)

__all__ = ['SingleUnit']


class SingleUnit(_Immutable):
    """An immutable unit identified by its canonical prefix and unit symbols.

    Hold shared immutable prefix and unit definitions resolved at construction.

    Accept a registered symbol or singular English name, optionally with one
    SI prefix: `'m'`, `'metre'`, `'km'`, and `'kilometre'`. Inputs are
    case-sensitive. Micro prefix aliases `'u'` and `'μ'` normalize to
    `'µ'`; kilo uses lowercase `'k'`, while `'K'` denotes kelvin.
    The empty string represents a dimensionless unit.

    Registered names may contain words separated by single ASCII spaces,
    as in ``astronomical unit``, or hyphens as in ``pascal-second``.
    Prefix names attach directly to the first word, as in
    ``kiloastronomical unit``. Names are matched exactly.
    Prefix permissions are checked on the spelling used, before alias
    normalization. ``symbol`` prefers a permitted symbol, then a name;
    ``name`` prefers a permitted name, then a symbol. This keeps reconstruction
    valid even when only names or only symbol aliases accept a given prefix.

    The built-in unit table contains s, m, g, A, K, mol, and cd; Unit.define()
    can register more units in the current process. Compound expressions and
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
        prefix, unit = _resolve_unit(text)
        return cls._from_definitions(prefix, unit)

    @classmethod
    def _from_definitions(cls, prefix: PrefixDefinition, unit: UnitDefinition) -> SingleUnit:
        """Construct from shared immutable definitions. Internal use only."""
        obj = super().__new__(cls)
        object.__setattr__(obj, '_prefix', prefix)
        object.__setattr__(obj, '_unit', unit)
        return obj

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.symbol!r})'

    def __str__(self) -> str:
        return self.symbol

    def __hash__(self) -> int:
        return hash((self.prefix_symbol, self.unprefixed_symbol))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SingleUnit):
            return NotImplemented
        return (
            self.prefix_symbol == other.prefix_symbol
            and self.unprefixed_symbol == other.unprefixed_symbol
        )

    def __copy__(self) -> SingleUnit:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> SingleUnit:
        return self

    def __reduce__(self) -> tuple[type[SingleUnit], tuple[str]]:
        return SingleUnit, (self.symbol,)

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
        for symbol in filter(self._unit.allows_prefix, self._unit.symbols):
            return self._prefix.symbol + symbol
        return None

    def _get_name(self) -> str | None:
        """Return a permitted name, or None if this prefix requires a symbol."""
        if not self.has_prefix():
            return self._unit.name
        for name in filter(self._unit.allows_prefix, self._unit.names):
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
    def log_degree(self) -> Fraction:
        return self._unit.log_degree

    @property
    def dimension(self) -> Dimension:
        return self._unit.dimension

    def without_prefix(self) -> SingleUnit:
        """Return the unit without its prefix; no magnitude is converted."""
        return SingleUnit._from_definitions(NO_PREFIX, self._unit) if self.has_prefix() else self

    def has_prefix(self) -> bool:
        return self.prefix_symbol != ''
