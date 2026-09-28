class PrefixData:
    '''
    Attributes:
        factor (float): 1000 for kilo-
        names (tuple[str, ...]): kilo
    '''

    __slots__ = ('factor', 'names')

    def __init__(self, facotr: float, *names: str) -> None:
        self.factor = facotr
        self.names = names

    @property
    def name(self) -> str: return self.names[0]


__PREFIX_LIB: dict[str | tuple[str, ...], PrefixData] = {
    # whole unit
    'Q': PrefixData(1e30, 'quetta'),
    'R': PrefixData(1e27, 'ronna'),
    'Y': PrefixData(1e24, 'yotta'),
    'Z': PrefixData(1e21, 'zetta'),
    'E': PrefixData(1e18, 'exa'),
    'P': PrefixData(1e15, 'peta'),
    'T': PrefixData(1e12, 'tera'),
    'G': PrefixData(1e9, 'giga'),
    'M': PrefixData(1e6, 'mega'),
    ('k', 'K'): PrefixData(1e3, 'kilo'),
    'h': PrefixData(1e2, 'hecto'),
    'da': PrefixData(1e1, 'deca'),
    '': PrefixData(1, ''),
    # sub-unit
    'd': PrefixData(1e-1, 'deci'),
    'c': PrefixData(1e-2, 'centi'),
    'm': PrefixData(1e-3, 'milli'),
    ('µ', 'μ', 'u'): PrefixData(1e-6, 'micro'),  # chr(0xB5), chr(0x03BC)
    'n': PrefixData(1e-9, 'nano'),
    'p': PrefixData(1e-12, 'pico'),
    'f': PrefixData(1e-15, 'femto'),
    'a': PrefixData(1e-18, 'atto'),
    'z': PrefixData(1e-21, 'zepto'),
    'y': PrefixData(1e-24, 'yocto'),
    'r': PrefixData(1e-27, 'ronto'),
    'q': PrefixData(1e-30, 'quecto'),
}

PREFIX: dict[str, PrefixData] = {
    prefix[0] if isinstance(prefix, tuple) else prefix: data
    for prefix, data in __PREFIX_LIB.items()
}
'''prefix {symbol: data}'''

PREFIX_NAME: dict[str, str] = {
    name: prefix for prefix, data in PREFIX.items() for name in data.names if name
}
'''prefix {name: symbol}'''

PREFIX_ALIAS: dict[str, str] = {
    alias: prefix[0] for prefix in __PREFIX_LIB if isinstance(prefix, tuple) for alias in prefix[1:]
}
'''prefix {alias: symbol}'''
