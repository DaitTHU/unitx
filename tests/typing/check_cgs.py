"""Scalar CGS consumers, also checked with NumPy absent."""

from decimal import Decimal
from fractions import Fraction
from typing import assert_type

from unitx import Quantity, Unit
from unitx.cgs import cgs, emu, esu, gaussian, hl
from unitx.si import si

assert_type(cgs.parse('erg/cm'), Unit)
assert_type(gaussian.parse('erg/statC'), Unit)
assert_type(gaussian.dyn, Unit)
assert_type(gaussian.statC, Unit)
assert_type(3 * cgs.dyn, Quantity[int])
assert_type(gaussian.convert(1 * gaussian.statC, gaussian.Mx), Quantity[int | Fraction | float])
assert_type(gaussian.convert((1 + 2j) * gaussian.statC, gaussian.Mx), Quantity[complex])
assert_type(gaussian.convert(Decimal(1) * cgs.dyn, cgs.dyn), Quantity[Decimal])
assert_type(si.em.k1, Quantity[float])
assert_type(si.em.alpha, Quantity[float])
assert_type(si.em.k2, Quantity[float])
assert_type(si.em.c, Quantity[float])
assert_type(si.c, Unit)
assert_type(gaussian.em.alpha, Quantity[float])
assert_type(gaussian.em.k2, Quantity[float])
assert_type(esu.parse('esuB'), Unit)
assert_type(emu.parse('abcoulomb'), Unit)
assert_type(hl.parse('hlC'), Unit)
assert_type(esu.statC, Unit)
assert_type(emu.abA, Unit)
assert_type(hl.charge, Unit)
assert_type(esu.convert(1 * si.C, esu.statC), Quantity[int | Fraction | float])
assert_type(emu.convert(1.0 * si.C, emu.abC), Quantity[int | Fraction | float])
assert_type(hl.convert(Fraction(2, 3) * si.C, hl.charge), Quantity[int | Fraction | float])
assert_type(hl.convert((1 + 2j) * si.C, hl.charge), Quantity[complex])
assert_type(esu.convert(Decimal(1) * cgs.dyn, cgs.dyn), Quantity[Decimal])
assert_type(esu.em.k2, Quantity[float])
assert_type(emu.em.k1, Quantity[float])
assert_type(hl.em.alpha, Quantity[float])
