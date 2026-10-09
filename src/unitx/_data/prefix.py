from ..factor import Factor, ONE, TEN


class PrefixData:
    '''
    Attributes:
        factor (Factor): 1000 for kilo-
        names (tuple[str, ...]): kilo
    '''

    __slots__ = ('factor', 'names')

    def __init__(self, facotr: Factor, *names: str) -> None:
        self.factor = facotr
        self.names = names

    @property
    def name(self) -> str: return self.names[0]


__PREFIX_LIB: dict[str | tuple[str, ...], PrefixData] = {
    # whole unit
    'Q': PrefixData(TEN ** 30, 'quetta'),
    'R': PrefixData(TEN ** 27, 'ronna'),
    'Y': PrefixData(TEN ** 24, 'yotta'),
    'Z': PrefixData(TEN ** 21, 'zetta'),
    'E': PrefixData(TEN ** 18, 'exa'),
    'P': PrefixData(TEN ** 15, 'peta'),
    'T': PrefixData(TEN ** 12, 'tera'),
    'G': PrefixData(TEN ** 9, 'giga'),
    'M': PrefixData(TEN ** 6, 'mega'),
    ('k', 'K'): PrefixData(TEN ** 3, 'kilo'),
    'h': PrefixData(TEN ** 2, 'hecto'),
    'da': PrefixData(TEN ** 1, 'deca'),
    '': PrefixData(ONE, ''),
    # sub-unit
    'd': PrefixData(TEN ** -1, 'deci'),
    'c': PrefixData(TEN ** -2, 'centi'),
    'm': PrefixData(TEN ** -3, 'milli'),
    ('µ', 'μ', 'u'): PrefixData(TEN ** -6, 'micro'),  # chr(0xB5), chr(0x03BC)
    'n': PrefixData(TEN ** -9, 'nano'),
    'p': PrefixData(TEN ** -12, 'pico'),
    'f': PrefixData(TEN ** -15, 'femto'),
    'a': PrefixData(TEN ** -18, 'atto'),
    'z': PrefixData(TEN ** -21, 'zepto'),
    'y': PrefixData(TEN ** -24, 'yocto'),
    'r': PrefixData(TEN ** -27, 'ronto'),
    'q': PrefixData(TEN ** -30, 'quecto'),
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
