"""Check that the ready-to-use SI namespaces retain concrete public types."""

from decimal import Decimal
from typing import assert_type

from unitx.quantity import Quantity
from unitx.si import SI, si
from unitx.unit import Unit


def test_si_unit_types() -> None:
    assert_type(si.m, Unit)
    assert_type(si.MeV, Unit)
    assert_type(si.au, Unit)
    assert_type(2 * si.m, Quantity[int])
    assert_type(Decimal('0.5') * si.s, Quantity[Decimal])


def test_si_constant_types() -> None:
    assert_type(SI.c, Quantity[int])
    assert_type(SI.h, Quantity[Decimal])
    assert_type(SI.hbar, Quantity[Decimal])
    assert_type(SI.e, Quantity[Decimal])
    assert_type(SI.h.to(si.J * si.s), Quantity[Decimal])
