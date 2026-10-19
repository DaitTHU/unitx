from __future__ import annotations

from .dimension import Dimension
from .factor import Factor
from .unitregistry import (
    PREFIX_DEFINITIONS,
    UNIT_DEFINITIONS,
    _preferred_unit_spelling,
    _resolve_unit_spelling,
)

__all__ = ['SingleUnit']


class SingleUnit:
    """An immutable unit identified by its canonical prefix and unit symbols.

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

    __slots__ = ('_unit_symbol', '_prefix_symbol')
    _unit_symbol: str
    _prefix_symbol: str

    def __new__(cls, text: str, /) -> SingleUnit:
        if not isinstance(text, str):
            raise TypeError(f'Unit symbol or name must be str, got {type(text).__name__}.')
        prefix_symbol, unit_symbol = _resolve_unit_spelling(text)
        return cls._from_symbols(prefix_symbol, unit_symbol)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError('SingleUnit objects are immutable.')

    def __delattr__(self, name: str) -> None:
        raise AttributeError('SingleUnit objects are immutable.')

    @classmethod
    def _from_symbols(cls, prefix_symbol: str, unit_symbol: str) -> SingleUnit:
        """Construct from canonical symbols. Internal use only."""
        obj = super().__new__(cls)
        object.__setattr__(obj, '_prefix_symbol', prefix_symbol)
        object.__setattr__(obj, '_unit_symbol', unit_symbol)
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
        return self._prefix_symbol

    @property
    def unprefixed_symbol(self) -> str:
        return self._unit_symbol

    @property
    def symbol(self) -> str:
        return _preferred_unit_spelling(self.prefix_symbol, self.unprefixed_symbol)[0]

    def _expression(self) -> str:
        spelling, is_name = _preferred_unit_spelling(self.prefix_symbol, self.unprefixed_symbol)
        return f'[{spelling}]' if is_name else spelling

    @property
    def prefix_name(self) -> str:
        return PREFIX_DEFINITIONS[self.prefix_symbol].name

    @property
    def unprefixed_name(self) -> str:
        return UNIT_DEFINITIONS[self.unprefixed_symbol].name

    @property
    def name(self) -> str:
        return _preferred_unit_spelling(
            self.prefix_symbol, self.unprefixed_symbol, prefer_name=True
        )[0]

    @property
    def prefix_factor(self) -> Factor:
        return PREFIX_DEFINITIONS[self.prefix_symbol].factor

    @property
    def unprefixed_factor(self) -> Factor:
        """Scale from the unprefixed unit to its coherent SI unit."""
        return UNIT_DEFINITIONS[self.unprefixed_symbol].factor

    @property
    def factor(self) -> Factor:
        """Scale from this unit to its coherent SI unit, including the prefix."""
        return self.prefix_factor * self.unprefixed_factor

    @property
    def dimension(self) -> Dimension:
        return UNIT_DEFINITIONS[self.unprefixed_symbol].dimension

    def without_prefix(self) -> SingleUnit:
        """Return the unit without its prefix; no magnitude is converted."""
        return SingleUnit._from_symbols('', self._unit_symbol) if self.has_prefix() else self

    def has_prefix(self) -> bool:
        return self.prefix_symbol != ''
