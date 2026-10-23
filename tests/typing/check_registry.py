from decimal import Decimal
from typing import Final, assert_type

from unitx import Quantity, Registry, Unit, UnitCatalog
from unitx.imperial import imperial
from unitx.serialization import from_dict, to_dict
from unitx.si import si
from unitx.unitparser import UnitParser

assert_type(imperial.ft, Unit)
assert_type(imperial.parse('ft/s'), Unit)
assert_type(3 * imperial.ft, Quantity[int])
private = Registry([si.m, si.s])
assert_type(private.include(si.m, si.s), None)
assert_type(private.parse('m/s'), Unit)
assert_type(private.define(Decimal('0.5') * si.m, 'span', 'private span'), Unit)
UnitParser('m', registry=private).parse()
from_dict(to_dict(3 * imperial.ft), registry=Registry([imperial.ft]))

assert_type(si.parse('m/s'), Unit)


class laboratory(UnitCatalog, units=(si.m,)):
    span: Final = (2 * si.m).define('span', 'survey span')


assert_type(laboratory.span, Unit)
assert_type(laboratory.parse('span'), Unit)
assert_type(3 * laboratory.span, Quantity[int])


class scoped_laboratory(UnitCatalog, units=(si.m,)):
    length: Final = (Decimal('0.5') * si.m).define('scspan', 'scope span')


assert_type(scoped_laboratory.length, Unit)
assert_type(scoped_laboratory.parse('scspan'), Unit)
assert_type(3 * scoped_laboratory.length, Quantity[int])

with private.defining() as selected:
    assert_type(selected, Registry)
    assert_type((2 * si.s).define('wtime', 'with time'), Unit)

with private.defining():
    assert_type(si.m.define('wlength', 'with length'), Unit)


class unprefixed(UnitCatalog):
    span: Final = si.m.define('span', prefixable=False)


assert_type(unprefixed.span, Unit)
assert_type(private.define(si.m, 'length'), Unit)
assert_type(Registry(prefixable=False), Registry)
assert_type(si.m.define('length'), Unit)
assert_type((2 * si.m).define('doubleLength'), Unit)
