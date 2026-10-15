from .dimension import DIMENSIONLESS, Dimension
from .factor import ONE, TEN, Factor


class PrefixData:
    __slots__ = ('factor', 'names')

    def __init__(self, factor: Factor, name: str, *name_alias: str) -> None:
        self.factor = factor
        self.names = (name, *name_alias)

    @property
    def name(self) -> str:
        return self.names[0]


_PREFIX_ENTRIES: dict[str | tuple[str, ...], PrefixData] = {
    # whole unit
    'Q': PrefixData(TEN**30, 'quetta'),
    'R': PrefixData(TEN**27, 'ronna'),
    'Y': PrefixData(TEN**24, 'yotta'),
    'Z': PrefixData(TEN**21, 'zetta'),
    'E': PrefixData(TEN**18, 'exa'),
    'P': PrefixData(TEN**15, 'peta'),
    'T': PrefixData(TEN**12, 'tera'),
    'G': PrefixData(TEN**9, 'giga'),
    'M': PrefixData(TEN**6, 'mega'),
    ('k', 'K'): PrefixData(TEN**3, 'kilo'),
    'h': PrefixData(TEN**2, 'hecto'),
    'da': PrefixData(TEN**1, 'deca'),
    '': PrefixData(ONE, ''),
    # sub unit
    'd': PrefixData(TEN**-1, 'deci'),
    'c': PrefixData(TEN**-2, 'centi'),
    'm': PrefixData(TEN**-3, 'milli'),
    ('µ', 'μ', 'u'): PrefixData(TEN**-6, 'micro'),  # chr(0xB5), chr(0x3BC)
    'n': PrefixData(TEN**-9, 'nano'),
    'p': PrefixData(TEN**-12, 'pico'),
    'f': PrefixData(TEN**-15, 'femto'),
    'a': PrefixData(TEN**-18, 'atto'),
    'z': PrefixData(TEN**-21, 'zepto'),
    'y': PrefixData(TEN**-24, 'yocto'),
    'r': PrefixData(TEN**-27, 'ronto'),
    'q': PrefixData(TEN**-30, 'quecto'),
}

PREFIXES: dict[str, PrefixData] = {
    prefix[0] if isinstance(prefix, tuple) else prefix: data
    for prefix, data in _PREFIX_ENTRIES.items()
}
"""prefix {symbol: data}"""

PREFIX_NAME_TO_SYMBOL: dict[str, str] = {
    name: prefix for prefix, data in PREFIXES.items() for name in data.names if name
}
"""prefix {name: symbol}"""

PREFIX_SYMBOL_ALIASES: dict[str, str] = {
    alias: prefix[0]
    for prefix in _PREFIX_ENTRIES
    if isinstance(prefix, tuple)
    for alias in prefix[1:]
}
"""prefix {symbol alias: symbol}"""


class UnitData:
    __slots__ = ('factor', 'names', 'dimension', 'prefixable')

    def __init__(
        self,
        factor: Factor,
        name: str,
        *name_alias: str,
        dimension: Dimension,
        prefixable: bool = True,
    ) -> None:
        self.factor = factor
        self.names = (name, *name_alias)
        self.dimension = dimension
        self.prefixable = prefixable

    @property
    def name(self) -> str:
        return self.names[0]


_UNIT_ENTRIES: dict[str | tuple[str, ...], UnitData] = {
    '': UnitData(ONE, '', dimension=DIMENSIONLESS, prefixable=False),
    's': UnitData(ONE, 'second', dimension=Dimension(T=1)),
    'm': UnitData(ONE, 'metre', 'meter', dimension=Dimension(L=1)),
    'g': UnitData(TEN**-3, 'gram', dimension=Dimension(M=1)),
    'A': UnitData(ONE, 'ampere', dimension=Dimension(I=1)),
    'K': UnitData(ONE, 'kelvin', dimension=Dimension(Theta=1)),
    'mol': UnitData(ONE, 'mole', dimension=Dimension(N=1)),
    'cd': UnitData(ONE, 'candela', dimension=Dimension(J=1)),
}

UNITS: dict[str, UnitData] = {
    unit[0] if isinstance(unit, tuple) else unit: data for unit, data in _UNIT_ENTRIES.items()
}
"""unit {symbol: data}"""

UNIT_NAME_TO_SYMBOL: dict[str, str] = {
    name: unit for unit, data in UNITS.items() for name in data.names if name
}
"""unit {name: symbol}"""

UNIT_SYMBOL_ALIASES: dict[str, str] = {
    alias: unit[0] for unit in _UNIT_ENTRIES if isinstance(unit, tuple) for alias in unit[1:]
}
"""unit {symbol alias: symbol}"""
