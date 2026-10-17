from __future__ import annotations

import warnings

from ._unit_data import (
    PREFIX_NAME_TO_SYMBOL,
    PREFIX_SYMBOL_ALIASES,
    PREFIXES,
    UNIT_NAME_TO_SYMBOL,
    UNIT_SYMBOL_ALIASES,
    UNITS,
)
from .dimension import Dimension
from .exceptions import _WARNING_SKIP_PREFIXES, UnitSymbolError, UnitSymbolWarning
from .factor import Factor

__all__ = ['SingleUnit']

_PREFIX_MAXLEN = max(map(len, PREFIXES))
_PREFIX_NAME_MINLEN = min(map(len, PREFIX_NAME_TO_SYMBOL))
_PREFIX_NAME_MAXLEN = max(map(len, PREFIX_NAME_TO_SYMBOL))


def _resolve_single(text: str, *, warn: bool = True) -> tuple[str, str]:
    """Resolve a symbol or name; registration checks can suppress alias warnings."""
    # unit symbol without prefix
    if text in UNITS:
        return '', text
    # unit alias symbol without prefix
    if text in UNIT_SYMBOL_ALIASES:
        return '', UNIT_SYMBOL_ALIASES[text]
    # unit (alias) symbol with prefix symbol
    if len(text) > 1:
        for prefix_length in range(1, _PREFIX_MAXLEN + 1):
            prefix_symbol, unit_symbol = text[:prefix_length], text[prefix_length:]
            if prefix_symbol in PREFIX_SYMBOL_ALIASES:
                prefix_symbol_alias = prefix_symbol
                prefix_symbol = PREFIX_SYMBOL_ALIASES[prefix_symbol]
            else:
                prefix_symbol_alias = None
            if unit_symbol in UNIT_SYMBOL_ALIASES:
                unit_symbol = UNIT_SYMBOL_ALIASES[unit_symbol]
            if prefix_symbol in PREFIXES and unit_symbol in UNITS:
                if UNITS[unit_symbol].prefixable:
                    if warn and prefix_symbol_alias in UNITS:
                        symbol = prefix_symbol + unit_symbol
                        name = f'{PREFIXES[prefix_symbol].name}{UNITS[unit_symbol].name}'
                        ambiguous_symbol = f'{prefix_symbol_alias} {unit_symbol}'
                        ambiguous = f'{UNITS[prefix_symbol_alias].name} {UNITS[unit_symbol].name}'
                        warning_message = (
                            f'{text!r} is interpreted as {symbol!r} ({name}). '
                            f'Use {symbol!r} for {name}, or {ambiguous_symbol!r} for {ambiguous}.'
                        )
                        warnings.warn(
                            warning_message,
                            UnitSymbolWarning,
                            skip_file_prefixes=_WARNING_SKIP_PREFIXES,
                        )
                    return prefix_symbol, unit_symbol
    # unit name without prefix
    if text in UNIT_NAME_TO_SYMBOL:
        return '', UNIT_NAME_TO_SYMBOL[text]
    # unit name with prefix name
    if len(text) > _PREFIX_NAME_MINLEN:
        for prefix_length in range(_PREFIX_NAME_MINLEN, _PREFIX_NAME_MAXLEN + 1):
            prefix_name, unit_name = text[:prefix_length], text[prefix_length:]
            if prefix_name in PREFIX_NAME_TO_SYMBOL and unit_name in UNIT_NAME_TO_SYMBOL:
                if UNITS[UNIT_NAME_TO_SYMBOL[unit_name]].prefixable:
                    return PREFIX_NAME_TO_SYMBOL[prefix_name], UNIT_NAME_TO_SYMBOL[unit_name]
    error_message = f'{text!r} is not a valid unit symbol.'
    raise UnitSymbolError(error_message)


class SingleUnit:
    """An immutable unit identified by its canonical prefix and unit symbols.

    Accept a registered symbol or singular English name, optionally with one
    SI prefix: `'m'`, `'metre'`, `'km'`, and `'kilometre'`. Inputs are
    case-sensitive. Micro prefix aliases `'u'` and `'μ'` normalize to
    `'µ'`; kilo uses lowercase `'k'`, while `'K'` denotes kelvin.
    The empty string represents a dimensionless unit.

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

    __slots__ = ('_unit', '_prefix')
    _unit: str
    _prefix: str

    def __new__(cls, text: str, /) -> SingleUnit:
        if not isinstance(text, str):
            raise TypeError(f'Unit symbol or name must be str, got {type(text).__name__}.')
        prefix, unit = _resolve_single(text)
        return cls._from_symbol(prefix, unit)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError('SingleUnit objects are immutable.')

    def __delattr__(self, name: str) -> None:
        raise AttributeError('SingleUnit objects are immutable.')

    @classmethod
    def _from_symbol(cls, prefix: str, unit: str) -> SingleUnit:
        """Construct from canonical symbols. Internal use only."""
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
        return self._prefix

    @property
    def unprefixed_symbol(self) -> str:
        return self._unit

    @property
    def symbol(self) -> str:
        return self.prefix_symbol + self.unprefixed_symbol

    @property
    def prefix_name(self) -> str:
        return PREFIXES[self.prefix_symbol].name

    @property
    def unprefixed_name(self) -> str:
        return UNITS[self.unprefixed_symbol].name

    @property
    def name(self) -> str:
        return self.prefix_name + self.unprefixed_name

    @property
    def prefix_factor(self) -> Factor:
        return PREFIXES[self.prefix_symbol].factor

    @property
    def unprefixed_factor(self) -> Factor:
        """Scale from the unprefixed unit to its coherent SI unit."""
        return UNITS[self.unprefixed_symbol].factor

    @property
    def factor(self) -> Factor:
        """Scale from this unit to its coherent SI unit, including the prefix."""
        return self.prefix_factor * self.unprefixed_factor

    @property
    def dimension(self) -> Dimension:
        return UNITS[self.unprefixed_symbol].dimension

    def without_prefix(self) -> SingleUnit:
        """Return the unit without its prefix; no magnitude is converted."""
        return SingleUnit._from_symbol('', self._unit) if self.has_prefix() else self

    def has_prefix(self) -> bool:
        return self.prefix_symbol != ''
