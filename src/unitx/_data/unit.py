from ..dimension import Dimension
from ..factor import Factor, ONE, TEN


class UnitData:
    '''
    Attributes:
        factor (Factor): 1/1000 for gram
        names (tuple[str, ...]): liter or litre
        dimension (Dimension): L³ for liter
        prefixable (bool): whether the unit can be prefixed
    '''

    __slots__ = ('factor', 'names', 'dimension', 'prefixable')

    def __init__(self, factor: Factor, *names: str, dimension: Dimension, prefixable=True) -> None:
        self.factor = factor
        self.names = names
        self.dimension = dimension
        self.prefixable = prefixable

    @property
    def name(self) -> str: return self.names[0]


UNIT: dict[str, UnitData] = {
    '': UnitData(ONE, '', dimension=Dimension(), prefixable=False),
    's': UnitData(ONE, 'second', dimension=Dimension(T=1)),
    'm': UnitData(ONE, 'meter', 'metre', dimension=Dimension(L=1)),
    'g': UnitData(TEN ** -3, 'gram', dimension=Dimension(M=1)),
    'A': UnitData(ONE, 'ampere', dimension=Dimension(I=1)),
    'K': UnitData(ONE, 'kelvin', dimension=Dimension(Theta=1)),
    'mol': UnitData(ONE, 'mole', dimension=Dimension(N=1)),
    'cd': UnitData(ONE, 'candela', dimension=Dimension(J=1)),
}
'''unit {symbol: data}'''

UNIT_NAME: dict[str, str] = {
    name: unit for unit, data in UNIT.items() for name in data.names if name
}
'''unit {name: symbol}'''

UNIT_ALIAS: dict[str, str] = {}
'''unit {alias: symbol}'''
