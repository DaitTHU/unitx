from __future__ import annotations

from fractions import Fraction

from .dimension import Dimension, DIMENSIONLESS
from .monomial import Monomial
from .singleunit import SingleUnit
from .unitparser import UnitParser
from .utils.number import common_fraction


class Unit:

    __slots__ = ('_elements', '_dimension', '_factor', '_symbol')

    def __init__(self, symbol: str = '', /) -> None:
        if not isinstance(symbol, str):
            raise TypeError(f'symbol must be str, got {type(symbol)}.')
        self.__derive_properties(UnitParser(symbol).parse())

    @classmethod
    def _from_monomial(cls, elements: Monomial[SingleUnit], /) -> Unit:
        '''Direct constructor from Monomial without copy.'''
        obj = object.__new__(cls)
        obj.__derive_properties(elements)
        return obj

    def __derive_properties(self, elements: Monomial[SingleUnit], /) -> None:
        self._elements = elements
        self._dimension = DIMENSIONLESS
        self._factor = 1
        for unit, exp in elements.components():
            self._dimension *= unit.dimension ** exp
            self._factor *= unit.factor ** exp
        self._symbol = str(elements)

    @property
    def dimension(self) -> Dimension: return self._dimension
    @property
    def factor(self) -> float: return self._factor
    @property
    def symbol(self) -> str: return self._symbol

    def __repr__(self) -> str: return f'{type(self).__name__}({self.symbol!r})'

    def __str__(self) -> str: return self.symbol

    def __hash__(self) -> int: return hash(tuple(self._elements.components()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._elements == other._elements

    def is_dimensionless(self) -> bool: return self.dimension is DIMENSIONLESS

    def __len__(self) -> int: return len(self._elements)

    def __bool__(self) -> bool: return len(self._elements) > 0

    def __mul__(self, other: Unit) -> Unit:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._from_monomial(self._elements * other._elements)

    def __truediv__(self, other: Unit) -> Unit:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._from_monomial(self._elements / other._elements)

    def __pow__(self, exponent: int | Fraction | float) -> Unit:
        if isinstance(exponent, float):
            exponent = common_fraction(exponent, floatwarning_stacklevel=3)
        if not isinstance(exponent, (int, Fraction)):
            return NotImplemented
        return self._from_monomial(self._elements ** exponent)
