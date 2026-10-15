from __future__ import annotations

from ._numerics import Fractionable
from .dimension import DIMENSIONLESS, Dimension
from .factor import ONE, Factor
from .monomial import Monomial
from .singleunit import SingleUnit
from .unitparser import UnitParser


class Unit:
    __slots__ = ('_monomial', '_dimension', '_factor', '_symbol')
    _monomial: Monomial[SingleUnit]
    _symbol: str
    _factor: Factor
    _dimension: Dimension

    def __new__(cls, symbol: str = '', /) -> Unit:
        if not isinstance(symbol, str):
            raise TypeError(f'Unit symbol must be str, got {type(symbol)}.')
        return cls._from_monomial(UnitParser(symbol).parse())

    @classmethod
    def _from_monomial(cls, monomial: Monomial[SingleUnit], /) -> Unit:
        """Direct constructor from Monomial without copy. Internal use only."""
        obj = object.__new__(cls)
        obj._monomial = monomial
        obj._symbol = str(monomial)
        obj._factor = ONE
        obj._dimension = DIMENSIONLESS
        for unit, exponent in monomial.components():
            obj._factor *= unit.factor**exponent
            obj._dimension *= unit.dimension**exponent
        return obj

    def __repr__(self) -> str:
        expression = ' * '.join(
            f'{unit}'
            if exponent == 1
            else f'{unit}**{exponent}'
            if exponent.denominator == 1
            else f'{unit}**({exponent})'
            for unit, exponent in self._monomial.components()
        )
        return f'{type(self).__name__}({expression!r})'

    def __str__(self) -> str:
        return self.symbol

    def __hash__(self) -> int:
        return hash(self._monomial)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Unit):
            return NotImplemented
        return self._monomial == other._monomial

    def __mul__(self, other: object) -> Unit:
        if not isinstance(other, Unit):
            return NotImplemented
        return self._from_monomial(self._monomial * other._monomial)

    def __truediv__(self, other: object) -> Unit:
        if not isinstance(other, Unit):
            return NotImplemented
        return self._from_monomial(self._monomial / other._monomial)

    def __pow__(self, exponent: Fractionable) -> Unit:
        if not isinstance(exponent, Fractionable):
            return NotImplemented
        return self._from_monomial(self._monomial**exponent)

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def factor(self) -> Factor:
        return self._factor

    @property
    def dimension(self) -> Dimension:
        return self._dimension

    def is_dimensionless(self) -> bool:
        return self.dimension is DIMENSIONLESS
