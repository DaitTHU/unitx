"""Check the installed wheel before optional NumPy is installed."""

import os
import pickle
import sys
from decimal import Decimal
from fractions import Fraction
from importlib.metadata import distribution
from importlib.util import find_spec
from pathlib import Path
from typing import Final, assert_type

import unitx
from unitx import AffineQuantity, LogQuantity, Quantity, Registry, Unit, UnitCatalog
from unitx.factor import Factor
from unitx.imperial import imperial
from unitx.serialization import from_dict, to_dict
from unitx.si import si

installed = distribution('unitx')
package_root = Path(str(installed.locate_file('unitx')))
assert installed.version == os.environ['EXPECTED_VERSION']
assert Path(unitx.__file__).resolve() == (package_root / '__init__.py').resolve()
assert (package_root / 'py.typed').is_file()
assert find_spec('numpy') is None

distance = Quantity(Decimal('1.25'), si.m)
assert_type(distance.to(si.cm), Quantity[Decimal])
assert distance.value_in(si.cm) == Decimal(125)
assert Quantity(1, si.cm).value_in(si.m) == Fraction(1, 100)
assert 2 * Unit(si.m, scale=Factor(3)) == 6 * si.m

celsius = 20 @ si.degC
level = 30 @ si.dBm
assert_type(celsius, AffineQuantity[int])
assert_type(level, LogQuantity[int])
assert celsius.to_quantity(si.K) == Fraction(5863, 20) * si.K
assert (32 @ imperial.degF).to(si.degC) == 0 @ si.degC
assert level.to_quantity(si.W) == 1 * si.W
assert LogQuantity.from_quantity(1 * si.W, si.dBm) == level

registry = Registry([si.m])
with registry.defining():
    span = (2 * si.m).define('installedSpan')


class laboratory(UnitCatalog, units=(span,)):
    length: Final = span


assert_type(laboratory.parse('installedSpan'), Unit)
assert laboratory.parse('installedSpan') == span
quantity = 3 * span
assert from_dict(to_dict(quantity), registry=registry) == quantity
for value in (quantity, celsius, level):
    assert pickle.loads(pickle.dumps(value)) == value
assert 'numpy' not in sys.modules
assert 'sympy' not in sys.modules
