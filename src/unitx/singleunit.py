from __future__ import annotations

from ._data.prefix import PREFIX, PREFIX_ALIAS
from ._data.unit import UNIT, UNIT_ALIAS
from .dimension import Dimension

_PREFIX_MAXLEN = max(map(len, PREFIX))
_UNIT_MAXLEN = max(map(len, UNIT))
_SYMBOL_MAXLEN = _PREFIX_MAXLEN + _UNIT_MAXLEN


def _split_prefix_unit(symbol: str) -> tuple[str, str]:
    '''Split a symbol into prefix and unit.'''
    # unit without prefix
    if symbol in UNIT:
        return '', symbol
    # unit alias without prefix
    if symbol in UNIT_ALIAS:
        return '', UNIT_ALIAS[symbol]
    # unit with prefix
    if 1 < len(symbol) <= _SYMBOL_MAXLEN:
        for plen in range(1, _PREFIX_MAXLEN + 1):
            prefix, unit = symbol[:plen], symbol[plen:]
            if prefix in PREFIX_ALIAS:
                prefix = PREFIX_ALIAS[prefix]
            if unit in UNIT_ALIAS:
                unit = UNIT_ALIAS[unit]
            if prefix in PREFIX and unit in UNIT and UNIT[unit].prefixable:
                return prefix, unit
    raise UnitSymbolError(f'{symbol!r} is not a valid unit symbol.')


class SingleUnit:
    '''
    Single unit with optional prefix. Immutable.

    Attributes
    ---
    - `prefix`: prefix symbol, e.g. 'k'
    - `unit`: unit symbol, e.g. 'm'
    - `symbol`: full symbol, e.g. 'km'
    - `prefix_name`: prefix name, e.g. 'kilo'
    - `unit_name`: unit name, e.g. 'meter'
    - `name`: full name, e.g. 'kilometer'
    - `prefix_factor`: prefix factor, e.g. 1e3
    - `unit_factor`: unit factor, e.g. 1.0
    - `factor`: total factor, e.g. 1e3
    - `dimension`: dimension of the unit, e.g. Dimension(L=1)

    Construct
    ---
    Currently, only the symbol constructor is supported.
    >>> u = SingleUnit('km')
    >>> v = SingleUnit('meter')  # Not supported, use symbol 'm' instead.
    '''

    __slots__ = ('_prefix', '_unit')

    def __init__(self, symbol: str, /) -> None:
        if not isinstance(symbol, str):
            raise TypeError(f'symbol must be str, got {type(symbol)}.')
        prefix, unit = _split_prefix_unit(symbol)
        object.__setattr__(self, '_prefix', prefix)
        object.__setattr__(self, '_unit', unit)

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError('SingleUnit is immutable and cannot set attribute.')

    @classmethod
    def _from_prefix_unit(cls, prefix: str, unit: str, /) -> SingleUnit:
        '''Construct from prefix and unit.'''
        if prefix not in PREFIX:
            raise ValueError(f'{prefix!r} is not a valid prefix symbol.')
        if unit not in UNIT:
            raise ValueError(f'{unit!r} is not a valid unit symbol.')
        if not UNIT[unit].prefixable and prefix != '':
            raise ValueError(f'{unit!r} cannot be prefixed.')
        obj = object.__new__(cls)
        object.__setattr__(obj, '_prefix', prefix)
        object.__setattr__(obj, '_unit', unit)
        return obj

    @property
    def prefix(self) -> str: return self._prefix
    @property
    def unit(self) -> str: return self._unit
    @property
    def symbol(self) -> str: return self._prefix + self._unit
    @property
    def prefix_name(self) -> str: return PREFIX[self._prefix].name
    @property
    def unit_name(self) -> str: return UNIT[self._unit].name
    @property
    def name(self) -> str: return self.prefix_name + self.unit_name
    @property
    def prefix_factor(self) -> float: return PREFIX[self._prefix].factor
    @property
    def unit_factor(self) -> float: return UNIT[self._unit].factor
    @property
    def factor(self) -> float: return self.prefix_factor * self.unit_factor
    @property
    def dimension(self) -> Dimension: return UNIT[self._unit].dimension

    def deprefix(self) -> SingleUnit:
        '''Return a new SingleUnit without prefix.'''
        return self._from_prefix_unit('', self._unit) if self.has_prefix() else self

    def has_prefix(self) -> bool: return self._prefix != ''

    def __repr__(self) -> str: return f'SingleUnit({self.symbol!r})'

    def __str__(self) -> str: return self.symbol

    def __hash__(self) -> int: return hash((self._prefix, self._unit))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SingleUnit):
            return NotImplemented
        return self._prefix == other._prefix and self._unit == other._unit


class UnitSymbolError(ValueError):
    pass
