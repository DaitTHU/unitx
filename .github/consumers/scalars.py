"""Check the installed wheel before optional NumPy is installed."""

import os
import pickle
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
from importlib import import_module
from importlib.metadata import distribution
from importlib.util import find_spec
from pathlib import Path
from typing import Final, assert_type

import unitx
from unitx import (
    AffineQuantity,
    LogQuantity,
    LogUnit,
    Quantity,
    Registry,
    Unit,
    UnitCatalog,
    math,
    time,
)
from unitx.cgs import cgs, emu, esu, gaussian, hl
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
assert import_module('unitx.math') is math
assert import_module('unitx.time') is time

distance = Quantity(Decimal('1.25'), si.m)
assert_type(distance.to(si.cm), Quantity[Decimal])
assert distance.value_in(si.cm) == Decimal(125)
assert Quantity(1, si.cm).value_in(si.m) == Fraction(1, 100)
assert 2 * Unit(si.m, scale=Factor(3)) == 6 * si.m

assert (1 * cgs.erg).value_in(si.J) == Fraction(1, 10**7)
charge = gaussian.convert(1 * si.C, gaussian.statC)
assert_type(charge, Quantity[int | Fraction | float])
assert abs(gaussian.convert(charge, si.C).value - 1) < 1e-15
assert_type(si.em.k1, Quantity[float])
assert_type(gaussian.em.k2, Quantity[float])
assert isinstance(si.c, Unit) and si.em.c == 1 * si.c
assert gaussian.em.c is si.em.c
assert issubclass(gaussian, cgs) and gaussian.dyn is cgs.dyn
assert gaussian.parse('erg/statC') == cgs.erg / gaussian.statC
assert from_dict(to_dict(gaussian.statC), registry=Registry([gaussian.statC])) == gaussian.statC
assert pickle.loads(pickle.dumps(gaussian.G)) == gaussian.G
for catalogue, charge_unit in ((esu, esu.statC), (emu, emu.abC), (hl, hl.charge)):
    converted = catalogue.convert(1 * si.C, charge_unit)
    assert_type(converted, Quantity[int | Fraction | float])
    assert abs(catalogue.convert(converted, si.C).value - 1) < 1e-15
    assert catalogue.parse(charge_unit.expression()) == charge_unit
    assert pickle.loads(pickle.dumps(charge_unit)) == charge_unit
    assert from_dict(to_dict(charge_unit), registry=Registry([charge_unit])) == charge_unit

root = math.sqrt(9 * si.m**2)
assert_type(root, Quantity[float])
assert root == 3 * si.m
assert math.sin(90 * si.deg).value == 1.0
assert math.hypot(3 * si.m, 400 * si.cm) == 5 * si.m

stamp = time.time()
start = time.perf_counter()
elapsed = time.perf_counter() - start
assert_type(stamp, AffineQuantity[int])
assert_type(start, AffineQuantity[int])
assert_type(elapsed.unit, Unit)
assert type(stamp.value) is int and stamp.unit.delta_unit == si.ns
assert type(elapsed.value) is int and elapsed.unit == si.ns and elapsed.value >= 0
local_zone = timezone(timedelta(hours=8))
instant = datetime(2000, 1, 1, 12, 34, 56, 123456, tzinfo=local_zone)
point = time.from_datetime(instant)
assert_type(point, AffineQuantity[int])
assert_type(time.to_datetime(point), datetime)
utc_instant = time.to_datetime(point)
local_instant = time.to_datetime(point, tz=local_zone)
assert utc_instant == instant and utc_instant.tzinfo is timezone.utc
assert local_instant == instant and local_instant.tzinfo is local_zone
assert time.to_datetime(AffineQuantity(1500, si.ns)) == datetime(
    1970, 1, 1, 0, 0, 0, 2, tzinfo=timezone.utc
)
duration = time.from_timedelta(timedelta(microseconds=123456))
assert_type(duration, Quantity[int])
assert duration == 123456 * si.us
assert time.to_timedelta(Quantity(Decimal('0.0000015'), si.s)) == timedelta(microseconds=2)

celsius = 20 @ si.degC
level = 30 @ si.dBm
assert_type(celsius, AffineQuantity[int])
assert_type(level, LogQuantity[int])
assert celsius.to_quantity(si.K) == Fraction(5863, 20) * si.K
assert (32 @ imperial.degF).to(si.degC) == 0 @ si.degC
assert level.to_quantity(si.W) == 1 * si.W
assert LogQuantity.from_quantity(1 * si.W, si.dBm) == level
assert_type(si.dB(si.V), LogUnit)
assert si.dB(si.mW) is si.dBm
assert Unit('decibel')(si.V) is si.dBV
millivolt_level = Decimal(20) @ si.dB(si.mV)
assert_type(millivolt_level, LogQuantity[Decimal])
assert millivolt_level.to_quantity(si.mV).value == Decimal(10)

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
