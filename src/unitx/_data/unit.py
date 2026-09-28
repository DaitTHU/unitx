from ..dimension import Dimension

class UnitData:
    '''
    Attributes:
        factor (float): 1e-3 for gram
        names (tuple[str, ...]): liter or litre
        dimension (Dimension): L³ for liter
        prefixable (bool): whether the unit can be prefixed
    '''

    __slots__ = ('factor', 'names', 'dimension', 'prefixable')

    def __init__(self, factor: float, *names: str, dimension: Dimension, prefixable=True) -> None:
        self.factor = factor
        self.names = names
        self.dimension = dimension
        self.prefixable = prefixable

    @property
    def name(self) -> str: return self.names[0]


__UNIT_LIB: dict[str | tuple[str, ...], UnitData] = {
    '': UnitData(1, '', dimension=Dimension(), prefixable=False),
    's': UnitData(1, 'second', dimension=Dimension(T=1)),
    'm': UnitData(1, 'meter', 'metre', dimension=Dimension(L=1)),
    'g': UnitData(1e-3, 'gram', dimension=Dimension(M=1)),
    'A': UnitData(1, 'ampere', dimension=Dimension(I=1)),
    'K': UnitData(1, 'kelvin', dimension=Dimension(Theta=1)),
    'mol': UnitData(1, 'mole', dimension=Dimension(N=1)),
    'cd': UnitData(1, 'candela', dimension=Dimension(J=1)),
}

UNIT: dict[str, UnitData] = {
    unit[0] if isinstance(unit, tuple) else unit: data
    for unit, data in __UNIT_LIB.items()
}
'''unit {symbol: data}'''

UNIT_NAME: dict[str, str] = {
    name: unit for unit, data in UNIT.items() for name in data.names if name
}
'''unit {name: symbol}'''

UNIT_ALIAS: dict[str, str] = {
    alias: unit[0] for unit in __UNIT_LIB if isinstance(unit, tuple) for alias in unit[1:]
}
'''unit {alias: symbol}'''
