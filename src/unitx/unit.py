from __future__ import annotations

from fractions import Fraction

from .dimension import Dimension, DIMENSIONLESS
from .factor import Factor, ONE
from .monomial import Monomial
from .singleunit import SingleUnit
from .unitparser import UnitParser
from .utils.number import common_fraction


class Unit:

    __slots__ = ('_monomial', '_dimension', '_factor', '_symbol')
    _monomial: Monomial[SingleUnit]
    _dimension: Dimension
    _factor: Factor
    _symbol: str

    def __new__(cls, symbol: str = '', /) -> Unit:
        if not isinstance(symbol, str):
            raise TypeError(f'Unit symbol must be str, got {type(symbol)}.')
        return cls._from_monomial(UnitParser(symbol).parse())

    @classmethod
    def _from_monomial(cls, monomial: Monomial[SingleUnit], /) -> Unit:
        '''Direct constructor from Monomial without copy.'''
        obj = object.__new__(cls)
        obj._monomial = monomial
        obj._dimension = DIMENSIONLESS
        obj._factor = ONE
        for unit, exponent in monomial.components():
            obj._dimension *= unit.dimension ** exponent
            obj._factor *= unit.factor ** exponent
        obj._symbol = str(monomial)
        return obj

    @property
    def dimension(self) -> Dimension: return self._dimension
    @property
    def factor(self) -> Factor: return self._factor
    @property
    def symbol(self) -> str: return self._symbol

    def __repr__(self) -> str:
        s = self._monomial.format(mul=' * ', exp='**', frac=False)
        return f'{type(self).__name__}({s!r})'

    def __str__(self) -> str: return self.symbol

    def __hash__(self) -> int: return hash(tuple(self._monomial.components()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Unit):
            return NotImplemented
        return self._monomial == other._monomial

    def is_dimensionless(self) -> bool: return self.dimension is DIMENSIONLESS

    def __len__(self) -> int: return len(self._monomial)

    def __bool__(self) -> bool: return len(self._monomial) > 0

    def __mul__(self, other: Unit) -> Unit:
        from .quantity import Quantity
        if not isinstance(other, Unit):
            return Quantity(other, self)
        return self._from_monomial(self._monomial * other._monomial)

    def __truediv__(self, other: Unit) -> Unit:
        from .quantity import Quantity
        if not isinstance(other, Unit):
            return Quantity(other, self)
        return self._from_monomial(self._monomial / other._monomial)

    def __pow__(self, exponent: int | Fraction | float) -> Unit:
        if isinstance(exponent, float):
            exponent = common_fraction(exponent, stacklevel=2)
        if not isinstance(exponent, (int, Fraction)):
            return NotImplemented
        return self._from_monomial(self._monomial ** exponent)

    def __rtruediv__(self, other):
        from .quantity import Quantity
        return Quantity(other, self._from_monomial(self._monomial ** -1))
